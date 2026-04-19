# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerun bridge for logging pubsub messages with to_rerun() methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    cast,
    runtime_checkable,
)
from urllib.parse import quote

from reactivex.disposable import Disposable
from toolz import pipe  # type: ignore[import-untyped]
import typer

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.protocol.pubsub.patterns import Glob, pattern_matches
from dimos.utils.logging_config import setup_logger

# Message types with large payloads that need rate-limiting.
# Image (~1 MB/frame at 30 fps) and PointCloud2 (~600-800 KB/frame)
# can cause viewer OOM if logged at full rate. OccupancyGrid is converted
# into a textured mesh, so it is also treated as a heavy visualization payload.
_HEAVY_MSG_TYPES: tuple[type, ...] = (Image, PointCloud2, OccupancyGrid)

RERUN_GRPC_PORT = 9876
RERUN_WEB_PORT = 9090

# TODO OUT visual annotations
#
# In the future it would be nice if modules can annotate their individual OUTs with (general or rerun specific)
# hints related to their visualization
#
# so stuff like color, update frequency etc (some Image needs to be rendered on the 3d floor like occupancy grid)
# some other image is an image to be streamed into a specific 2D view etc.
#
# To achieve this we'd feed a full blueprint into the rerun bridge.
#
# rerun bridge can then inspect all transports used, all modules with their outs,
# automatically spy an all the transports and read visualization hints
#
# Temporarily we are using these "sideloading" visual_override={} dict on the bridge
# to define custom visualizations for specific topics
#
# as well as pubsubs={} to specify which protocols to listen to.


# TODO better TF processing
#
# this is rerun bridge specific, rerun has a specific (better) way of handling TFs
# using entity path conventions, each of these nodes in a path are TF frames:
#
# /world/robot1/base_link/camera/optical
#
# While here since we are just listening on TFMessage messages which optionally contain
# just a subset of full TF tree we don't know the full tree structure to build full entity
# path for a transform being published
#
# This is easy to reconstruct but a service/tf.py already does this so should be integrated here
#
# we have decoupled entity paths and actual transforms (like ROS TF frames)
# https://rerun.io/docs/concepts/logging-and-ingestion/transforms
#
# tf#/world
# tf#/base_link
# tf#/camera
#
# In order to solve this, bridge needs to own it's own tf service
# and render it's tf tree into correct rerun entity paths


logger = setup_logger()

if TYPE_CHECKING:
    from collections.abc import Callable

    from rerun._baseclasses import Archetype
    from rerun.blueprint import Blueprint

    from dimos.protocol.pubsub.spec import SubscribeAllCapable

BlueprintFactory: TypeAlias = "Callable[[], Blueprint]"

# to_rerun() can return a single archetype or a list of (entity_path, archetype) tuples
RerunMulti: TypeAlias = "list[tuple[str, Archetype]]"
RerunData: TypeAlias = "Archetype | RerunMulti"


def is_rerun_multi(data: Any) -> TypeGuard[RerunMulti]:
    """Check if data is a list of (entity_path, archetype) tuples."""
    from rerun._baseclasses import Archetype

    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], tuple)
        and len(data[0]) == 2
        and isinstance(data[0][0], str)
        and isinstance(data[0][1], Archetype)
    )


@runtime_checkable
class RerunConvertible(Protocol):
    """Protocol for messages that can be converted to Rerun data."""

    def to_rerun(self) -> RerunData: ...


ViewerMode = Literal["native", "web", "connect", "none"]


def _default_blueprint() -> Blueprint:
    """Default blueprint with black background and raised grid."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(  # type: ignore[no-any-return]
        rrb.Spatial3DView(
            origin="world",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(
                plane=rr.components.Plane3D.XY.with_distance(0.2),
            ),
        ),
    )


# Maps global_config.viewer -> bridge viewer_mode.
# Evaluated at blueprint construction time (main process), not in start() (worker process).
_BACKEND_TO_MODE: dict[str, ViewerMode] = {
    "rerun": "native",
    "rerun-web": "web",
    "rerun-connect": "connect",
    "none": "none",
}


def _resolve_viewer_mode() -> ViewerMode:
    from dimos.core.global_config import global_config

    return _BACKEND_TO_MODE.get(global_config.viewer, "native")


def _web_viewer_url(
    grpc_port: int = RERUN_GRPC_PORT,
    web_port: int = RERUN_WEB_PORT,
    server_uri: str | None = None,
) -> str:
    proxy_url = quote(server_uri or f"rerun+http://localhost:{grpc_port}/proxy", safe="")
    return f"http://localhost:{web_port}/?url={proxy_url}"


@dataclass
class Config(ModuleConfig):
    """Configuration for RerunBridgeModule."""

    pubsubs: list[SubscribeAllCapable[Any, Any]] = field(default_factory=lambda: [LCM()])

    visual_override: dict[Glob | str, Callable[[Any], Archetype]] = field(default_factory=dict)

    # Static items logged once after start. Maps entity_path -> callable(rr) returning Archetype
    static: dict[str, Callable[[Any], Archetype]] = field(default_factory=dict)

    min_interval_sec: float = 0.1  # Rate-limit per entity path (default: 10 Hz max)
    entity_prefix: str = "world"
    topic_to_entity: Callable[[Any], str] | None = None
    viewer_mode: ViewerMode = field(default_factory=_resolve_viewer_mode)
    connect_url: str = "rerun+http://127.0.0.1:9877/proxy"
    memory_limit: str = "25%"
    # Optional per-entity throttle for Rerun logging only. Keys may be exact
    # entity paths or Glob patterns, and values are minimum intervals in seconds.
    min_interval_by_entity: dict[Glob | str, float] = field(default_factory=dict)
    # TF is produced at high rate by robot odometry. This only throttles Rerun
    # visualization and does not affect odometry, planning, or control.
    tf_min_interval_sec: float = 0.0
    # Root TF frame that maps to entity_prefix in Rerun. Frames below this root
    # are logged as real entity paths, e.g. world/base_link/camera_link.
    tf_root_frame: str = "world"

    # Blueprint factory: callable(rrb) -> Blueprint for viewer layout configuration
    # Set to None to disable default blueprint
    blueprint: BlueprintFactory | None = _default_blueprint


@dataclass
class _TfEdge:
    parent_frame: str
    transform: Transform


class RerunBridgeModule(Module):
    """Bridge that logs messages from pubsubs to Rerun.

    Spawns its own Rerun viewer and subscribes to all topics on each provided
    pubsub. Any message that has a to_rerun() method is automatically logged.

    Example:
        from dimos.protocol.pubsub.impl.lcmpubsub import LCM

        lcm = LCM()
        bridge = RerunBridgeModule(pubsubs=[lcm])
        bridge.start()
        # All messages with to_rerun() are now logged to Rerun
        bridge.stop()
    """

    default_config = Config
    config: Config

    @lru_cache(maxsize=256)
    def _visual_override_for_entity_path(
        self, entity_path: str
    ) -> Callable[[Any], RerunData | None]:
        """Return a composed visual override for the entity path.

        Chains matching overrides from config, ending with final_convert
        which handles .to_rerun() or passes through Archetypes.
        """
        from rerun._baseclasses import Archetype

        # find all matching converters for this entity path
        matches = [
            fn
            for pattern, fn in self.config.visual_override.items()
            if pattern_matches(pattern, entity_path)
        ]

        # None means "suppress this topic entirely"
        if any(fn is None for fn in matches):
            return lambda msg: None

        # final step (ensures we return Archetype or None)
        def final_convert(msg: Any) -> RerunData | None:
            if isinstance(msg, Archetype):
                return msg
            if is_rerun_multi(msg):
                return msg
            if isinstance(msg, RerunConvertible):
                return msg.to_rerun()
            return None

        # compose all converters
        return lambda msg: pipe(msg, *matches, final_convert)

    def _get_entity_path(self, topic: Any) -> str:
        """Convert a topic to a Rerun entity path."""
        if self.config.topic_to_entity:
            return self.config.topic_to_entity(topic)

        # Default: use topic.name if available (LCM Topic), else str
        topic_str = getattr(topic, "name", None) or str(topic)
        # Strip everything after # (LCM topic suffix)
        topic_str = topic_str.split("#")[0]
        return f"{self.config.entity_prefix}{topic_str}"

    def _min_interval_for_message(self, entity_path: str, msg: Any) -> float:
        """Return the Rerun-only throttle interval for this entity/message."""
        entity_intervals = [
            interval_sec
            for pattern, interval_sec in self.config.min_interval_by_entity.items()
            if pattern_matches(pattern, entity_path)
        ]
        if entity_intervals:
            return max(entity_intervals)

        if entity_path == f"{self.config.entity_prefix}/tf":
            return self.config.tf_min_interval_sec

        if isinstance(msg, _HEAVY_MSG_TYPES):
            return self.config.min_interval_sec

        return 0.0

    @staticmethod
    def _timestamp_for_message(msg: Any) -> float:
        """Return a wall-clock timestamp for a pubsub message."""
        if isinstance(msg, TFMessage) and msg.transforms:
            return max(transform.ts for transform in msg.transforms)

        for attr in ("ts", "timestamp"):
            value = getattr(msg, attr, None)
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            if timestamp > 0:
                return timestamp

        return time.time()

    def _set_rerun_time_for_message(self, msg: Any) -> None:
        """Attach dynamic logs to explicit Rerun timelines.

        Without an explicit timeline, Rerun Web can display the first logged
        value for live streams while later updates exist in the recording but
        are not shown in the active time selection.
        """
        import rerun as rr

        self._rerun_log_seq += 1
        rr.set_time(
            "dimos_time",
            timestamp=self._timestamp_for_message(msg),
            recording=self._rr_recording,
        )
        rr.set_time("dimos_seq", sequence=self._rerun_log_seq, recording=self._rr_recording)

    def _flush_rerun_periodically(self) -> None:
        """Flush Rerun batches often enough for live web visualization."""
        now = time.monotonic()
        if now - self._last_rerun_flush < 0.5:
            return
        self._last_rerun_flush = now
        self._rr_recording.flush(timeout_sec=0.1)

    @staticmethod
    def _normalize_tf_frame(frame: str) -> str:
        """Normalize ROS/LCM TF frame IDs for stable lookup and entity paths."""
        return frame.strip().strip("/")

    @staticmethod
    def _tf_frame_segment(frame: str) -> str:
        """Return a safe single Rerun entity path segment for a TF frame."""
        normalized = RerunBridgeModule._normalize_tf_frame(frame)
        return normalized.replace("/", "_") or "unnamed"

    def _tf_root_path(self) -> str:
        return self.config.entity_prefix.strip("/") or "world"

    def _resolve_tf_entity_path(self, frame: str, seen: set[str] | None = None) -> str | None:
        """Resolve a TF frame into the corresponding Rerun entity path.

        Rerun transforms are attached to entity hierarchy edges, so the TF edge
        world -> base_link should be logged at world/base_link, not at an
        unrelated world/tf/base_link entity with named-frame metadata.
        """
        normalized = self._normalize_tf_frame(frame)
        root_frame = self._normalize_tf_frame(self.config.tf_root_frame)
        root_path = self._tf_root_path()

        if normalized in {root_frame, root_path}:
            return root_path

        if seen is None:
            seen = set()
        if normalized in seen:
            logger.warning("Ignoring cyclic TF tree", frame=normalized)
            return None
        seen.add(normalized)

        edge = self._tf_edges.get(normalized)
        if edge is None:
            return None

        parent_path = self._resolve_tf_entity_path(edge.parent_frame, seen)
        if parent_path is None:
            return None

        return f"{parent_path}/{self._tf_frame_segment(normalized)}"

    @staticmethod
    def _transform_to_rerun_entity(transform: Transform) -> Any:
        """Convert a TF transform to an entity-hierarchy Rerun transform."""
        import rerun as rr

        return rr.Transform3D(
            translation=[
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            rotation=transform.rotation.to_rerun(),
        )

    def _update_tf_tree(self, msg: TFMessage) -> None:
        """Update the bridge-owned TF tree from a TFMessage."""
        for transform in msg.transforms:
            parent_frame = self._normalize_tf_frame(transform.frame_id)
            child_frame = self._normalize_tf_frame(transform.child_frame_id)
            if not parent_frame or not child_frame:
                logger.warning(
                    "Ignoring TF with empty frame id",
                    parent_frame=transform.frame_id,
                    child_frame=transform.child_frame_id,
                )
                continue
            if parent_frame == child_frame:
                logger.warning("Ignoring self-referential TF", frame=child_frame)
                continue
            self._tf_edges[child_frame] = _TfEdge(
                parent_frame=parent_frame,
                transform=transform,
            )

    def _log_tf_tree(self) -> None:
        """Log the currently resolvable TF tree into Rerun entity hierarchy."""
        import rerun as rr

        resolved: list[tuple[str, Transform]] = []
        for child_frame, edge in self._tf_edges.items():
            entity_path = self._resolve_tf_entity_path(child_frame)
            if entity_path is None:
                continue
            resolved.append((entity_path, edge.transform))

        for entity_path, transform in sorted(resolved, key=lambda item: item[0].count("/")):
            rr.log(
                entity_path,
                self._transform_to_rerun_entity(transform),
                recording=self._rr_recording,
            )

    def _on_tf_message(self, msg: TFMessage, entity_path: str) -> None:
        """Handle TF separately from generic to_rerun conversion."""
        self._update_tf_tree(msg)

        min_interval_sec = self._min_interval_for_message(entity_path, msg)
        if min_interval_sec > 0:
            now = time.monotonic()
            last = self._last_log.get(entity_path, 0.0)
            if now - last < min_interval_sec:
                return
            self._last_log[entity_path] = now

        self._set_rerun_time_for_message(msg)
        self._log_tf_tree()

    def _on_message(self, msg: Any, topic: Any) -> None:
        """Handle incoming message - log to rerun."""
        import rerun as rr

        # convert a potentially complex topic object into an str rerun entity path
        entity_path: str = self._get_entity_path(topic)

        if isinstance(msg, TFMessage):
            self._on_tf_message(msg, entity_path)
            return
        if isinstance(msg, Transform) and entity_path == f"{self.config.entity_prefix}/tf":
            self._on_tf_message(TFMessage(msg), entity_path)
            return

        # Rate-limit high-bandwidth visualization streams before converting to
        # Rerun archetypes. This does not affect upstream pubsub consumers.
        min_interval_sec = self._min_interval_for_message(entity_path, msg)
        if min_interval_sec > 0:
            now = time.monotonic()
            last = self._last_log.get(entity_path, 0.0)
            if now - last < min_interval_sec:
                return
            self._last_log[entity_path] = now

        # apply visual overrides (including final_convert which handles .to_rerun())
        rerun_data: RerunData | None = self._visual_override_for_entity_path(entity_path)(msg)

        # converters can also suppress logging by returning None
        if not rerun_data:
            return

        self._set_rerun_time_for_message(msg)

        # TFMessage for example returns list of (entity_path, archetype) tuples
        if is_rerun_multi(rerun_data):
            for path, archetype in rerun_data:
                rr.log(path, archetype, recording=self._rr_recording)
        else:
            rr.log(entity_path, cast("Archetype", rerun_data), recording=self._rr_recording)
        self._flush_rerun_periodically()

    @rpc
    def start(self) -> None:
        import rerun as rr

        super().start()

        self._last_log: dict[str, float] = {}
        self._tf_edges: dict[str, _TfEdge] = {}
        self._rerun_log_seq = 0
        self._last_rerun_flush = 0.0
        logger.info("Rerun bridge starting", viewer_mode=self.config.viewer_mode)

        # Own one explicit RecordingStream. This avoids relying on
        # thread-local defaults from pubsub callback threads.
        self._rr_recording = rr.RecordingStream("dimos", make_default=True)
        rr.set_global_data_recording(self._rr_recording)

        if self.config.viewer_mode == "native":
            try:
                import rerun_bindings

                rerun_bindings.spawn(
                    port=RERUN_GRPC_PORT,
                    executable_name="dimos-viewer",
                    memory_limit=self.config.memory_limit,
                )
            except ImportError:
                pass  # dimos-viewer not installed
            except Exception:
                logger.warning(
                    "dimos-viewer found but failed to spawn, falling back to stock rerun",
                    exc_info=True,
                )
            rr.spawn(
                connect=True,
                memory_limit=self.config.memory_limit,
                recording=self._rr_recording,
            )
        elif self.config.viewer_mode == "web":
            try:
                server_uri = rr.serve_grpc(
                    grpc_port=RERUN_GRPC_PORT,
                    recording=self._rr_recording,
                    server_memory_limit=self.config.memory_limit,
                    newest_first=True,
                )
            except OSError as exc:
                logger.warning(
                    "Rerun gRPC port already in use, assuming an existing viewer is running",
                    port=RERUN_GRPC_PORT,
                    exc_info=exc,
                )
                server_uri = f"rerun+http://127.0.0.1:{RERUN_GRPC_PORT}/proxy"

            try:
                rr.serve_web_viewer(
                    web_port=RERUN_WEB_PORT,
                    connect_to=server_uri,
                    open_browser=False,
                )
            except OSError as exc:
                logger.warning(
                    "Rerun web viewer port already in use; reuse the existing web UI",
                    port=RERUN_WEB_PORT,
                    url=_web_viewer_url(),
                    exc_info=exc,
                )
            logger.info(
                "Rerun web viewer available",
                url=_web_viewer_url(server_uri=server_uri),
                grpc_uri=server_uri,
            )
        elif self.config.viewer_mode == "connect":
            rr.connect_grpc(self.config.connect_url, recording=self._rr_recording)
        # "none" - just init, no viewer (connect externally)

        if self.config.blueprint:
            rr.send_blueprint(self.config.blueprint(), recording=self._rr_recording)

        # Start pubsubs and subscribe to all messages
        for pubsub in self.config.pubsubs:
            logger.info(f"bridge listening on {pubsub.__class__.__name__}")
            if hasattr(pubsub, "start"):
                pubsub.start()  # type: ignore[union-attr]
            unsub = pubsub.subscribe_all(self._on_message)
            self._disposables.add(Disposable(unsub))

        # Add pubsub stop as disposable
        for pubsub in self.config.pubsubs:
            if hasattr(pubsub, "stop"):
                self._disposables.add(Disposable(pubsub.stop))  # type: ignore[union-attr]

        self._log_static()

    def _log_static(self) -> None:
        import rerun as rr

        for entity_path, factory in self.config.static.items():
            data = factory(rr)
            if isinstance(data, list):
                for archetype in data:
                    rr.log(
                        entity_path,
                        archetype,
                        static=True,
                        recording=self._rr_recording,
                    )
            else:
                rr.log(entity_path, data, static=True, recording=self._rr_recording)
        self._rr_recording.flush(timeout_sec=1.0)

    @rpc
    def stop(self) -> None:
        if hasattr(self, "_rr_recording"):
            self._rr_recording.flush(timeout_sec=1.0)
        super().stop()


def run_bridge(
    viewer_mode: str = "native",
    memory_limit: str = "25%",
) -> None:
    """Start a RerunBridgeModule with default LCM config and block until interrupted."""
    import signal

    from dimos.protocol.service.lcmservice import autoconf

    autoconf(check_only=True)

    bridge = RerunBridgeModule(
        viewer_mode=viewer_mode,
        memory_limit=memory_limit,
        # any pubsub that supports subscribe_all and topic that supports str(topic)
        # is acceptable here
        pubsubs=[LCM()],
    )

    bridge.start()

    signal.signal(signal.SIGINT, lambda *_: bridge.stop())
    signal.pause()


app = typer.Typer()


@app.command()
def cli(
    viewer_mode: str = typer.Option(
        "native", help="Viewer mode: native (desktop), web (browser), none (headless)"
    ),
    memory_limit: str = typer.Option(
        "25%", help="Memory limit for Rerun viewer (e.g., '4GB', '16GB', '25%')"
    ),
) -> None:
    """Rerun bridge for LCM messages."""
    run_bridge(viewer_mode=viewer_mode, memory_limit=memory_limit)


if __name__ == "__main__":
    app()

# you don't need to include this in your blueprint if you are not creating a
# custom rerun configuration for your deployment, you can also run rerun-bridge standalone
rerun_bridge = RerunBridgeModule.blueprint
