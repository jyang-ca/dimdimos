# Copyright 2026 Dimensional Inc.
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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Protocol

from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.autonomy.zones import (
    DEFAULT_ZONE_DEFINITIONS,
    ZoneDefinition,
    find_zone,
)
from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs import PoseStamped, Vector3
from dimos.msgs.sensor_msgs import Image
from dimos.navigation.base import NavigationState
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_ZONE_INSPECTION_OUTPUT_DIR = DIMOS_PROJECT_ROOT / "temp" / "dimos_zone_inspections"


class NavigatorSpec(Spec, Protocol):
    def set_goal(self, goal: PoseStamped) -> bool: ...
    def get_state(self) -> NavigationState: ...
    def is_goal_reached(self) -> bool: ...
    def cancel_goal(self) -> bool: ...


@dataclass
class ZoneInspectionConfig(ModuleConfig):
    zones: tuple[ZoneDefinition, ...] = field(default_factory=lambda: DEFAULT_ZONE_DEFINITIONS)
    output_dir: str = str(DEFAULT_ZONE_INSPECTION_OUTPUT_DIR)
    default_radius_m: float = 1.0
    default_views: int = 8
    navigation_timeout_s: float = 60.0
    settle_time_s: float = 0.5


@dataclass(frozen=True)
class InspectionCapture:
    view_index: int
    pose: PoseStamped
    image_path: str | None


class ZoneInspectionSkillContainer(Module[ZoneInspectionConfig]):
    """Composite skill for deterministic zone inspection captures."""

    default_config = ZoneInspectionConfig
    config: ZoneInspectionConfig

    color_image: In[Image]
    odom: In[PoseStamped]

    _navigator: NavigatorSpec

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._latest_image: Image | None = None
        self._latest_odom: PoseStamped | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.color_image.subscribe(self._on_color_image)))
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_color_image(self, image: Image) -> None:
        self._latest_image = image

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom

    @skill
    def inspect_zone(
        self,
        zone_name: str,
        reason: str,
        radius_m: float = 0.0,
        views: int = 0,
    ) -> str:
        """Inspect a known zone by navigating to it and capturing one camera view.

        Args:
            zone_name: Name of the known zone to inspect, for example "Zone A".
            reason: Short reason for the inspection, for example "temperature=100 C".
            radius_m: Deprecated compatibility argument. The inspection now captures one view.
            views: Deprecated compatibility argument. The inspection now captures one view.
        """

        zone = find_zone(zone_name, self.config.zones)
        if zone is None:
            known_zones = ", ".join(zone.name for zone in self.config.zones)
            return f"Unknown zone '{zone_name}'. Known zones: {known_zones}"

        session_dir = _inspection_session_dir(
            Path(self.config.output_dir),
            zone.name,
            reason,
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        pose = inspection_goal_pose(zone)
        logger.info("Navigating to zone inspection location", zone=zone.name)
        if not self._navigator.set_goal(pose):
            self._navigator.cancel_goal()
            return f"Failed to start inspection for '{zone.name}': navigator rejected the zone goal."

        if not self._wait_for_goal():
            self._navigator.cancel_goal()
            return (
                f"Inspection for '{zone.name}' stopped: failed to reach the zone within "
                f"{self.config.navigation_timeout_s:g}s."
            )

        if self.config.settle_time_s > 0:
            time.sleep(self.config.settle_time_s)
        image_path = self._capture_view(session_dir, zone)
        saved_count = 1 if image_path else 0
        return (
            f"Completed inspection for '{zone.name}' because {reason}. "
            f"Captured {saved_count}/1 image"
            + (" (missing due to unavailable camera frame)" if saved_count == 0 else "")
            + f". Output directory: {session_dir}"
        )

    def _wait_for_goal(self) -> bool:
        deadline = time.monotonic() + self.config.navigation_timeout_s
        saw_active = False
        while time.monotonic() < deadline:
            state = self._navigator.get_state()
            if state != NavigationState.IDLE:
                saw_active = True
            if state == NavigationState.IDLE and (saw_active or self._navigator.is_goal_reached()):
                return self._navigator.is_goal_reached()
            time.sleep(0.25)
        return False

    def _capture_view(self, session_dir: Path, zone: ZoneDefinition) -> str | None:
        image = self._latest_image
        if image is None:
            logger.warning("No camera frame available for zone inspection", zone=zone.name)
            return None

        image_path = session_dir / f"{_slugify(zone.name)}.jpg"
        if not image.save(str(image_path)):
            logger.warning("Failed to save zone inspection image", path=str(image_path))
            return None
        return str(image_path)


def inspection_goal_pose(zone: ZoneDefinition) -> PoseStamped:
    return PoseStamped(
        frame_id=zone.frame_id,
        position=Vector3(zone.center_x, zone.center_y, zone.center_z),
    )


def _inspection_session_dir(base_dir: Path, zone_name: str, reason: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = _slugify(f"{zone_name}_{reason}")[:80]
    return base_dir / f"{timestamp}_{suffix}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "zone_inspection"


zone_inspection_skill = ZoneInspectionSkillContainer.blueprint

__all__ = [
    "DEFAULT_ZONE_INSPECTION_OUTPUT_DIR",
    "InspectionCapture",
    "NavigatorSpec",
    "ZoneInspectionConfig",
    "ZoneInspectionSkillContainer",
    "inspection_goal_pose",
    "zone_inspection_skill",
]
