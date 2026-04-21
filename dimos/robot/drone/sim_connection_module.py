#!/usr/bin/env python3

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

"""Lightweight simulated drone connection for LiDAR mapping tests and demos."""

from dataclasses import dataclass
from functools import lru_cache
import threading
import time
from typing import Any, cast

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Transform, Twist, Vector3
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.data import get_data
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _sample_plane(
    center: np.ndarray[Any, Any],
    rotation: np.ndarray[Any, Any],
    half_size: np.ndarray[Any, Any],
    spacing: float,
) -> np.ndarray[Any, Any]:
    xs = np.arange(-half_size[0], half_size[0] + spacing, spacing, dtype=np.float32)
    ys = np.arange(-half_size[1], half_size[1] + spacing, spacing, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local = np.column_stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            np.zeros(xx.size, dtype=np.float32),
        ]
    )
    return cast("np.ndarray[Any, Any]", (local @ rotation.T + center).astype(np.float32))


def _sample_mesh_surface(
    vertices: np.ndarray[Any, Any],
    faces: np.ndarray[Any, Any],
    spacing: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    valid = areas > 1e-8
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    triangles = triangles[valid]
    areas = areas[valid]
    total_area = float(areas.sum())
    n_points = min(max_points, max(1, int(total_area / max(spacing * spacing, 1e-4))))
    probabilities = areas / total_area
    chosen = rng.choice(len(triangles), size=n_points, replace=True, p=probabilities)

    u = rng.random(n_points, dtype=np.float32)
    v = rng.random(n_points, dtype=np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    sampled = (
        triangles[chosen, 0]
        + u[:, None] * (triangles[chosen, 1] - triangles[chosen, 0])
        + v[:, None] * (triangles[chosen, 2] - triangles[chosen, 0])
    )
    return cast("np.ndarray[Any, Any]", sampled.astype(np.float32))


def _mujoco_scene_assets(room: str) -> dict[str, bytes]:
    data_dir = get_data("mujoco_sim")
    scene_dir = data_dir / f"scene_{room}"
    assets: dict[str, bytes] = {}

    for path in data_dir.glob("*.xml"):
        assets[path.name] = path.read_bytes()

    if scene_dir.exists():
        for path in scene_dir.rglob("*"):
            if path.suffix.lower() in {".obj", ".png"}:
                assets[path.name] = path.read_bytes()

    return assets


def _mujoco_scene_xml(room: str, room_from_occupancy: str | None) -> str:
    if room_from_occupancy:
        from pathlib import Path

        from dimos.mapping.occupancy.extrude_occupancy import generate_mujoco_scene
        from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

        return generate_mujoco_scene(OccupancyGrid.from_path(Path(room_from_occupancy)))

    data_dir = get_data("mujoco_sim")
    return (data_dir / f"scene_{room}.xml").read_text()


@lru_cache(maxsize=8)
def _build_mujoco_scene_points(
    room: str,
    room_from_occupancy: str | None,
    spacing: float,
    max_points: int,
    seed: int,
) -> np.ndarray[Any, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        _mujoco_scene_xml(room, room_from_occupancy),
        assets=_mujoco_scene_assets(room),
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(seed)
    sampled_groups: list[np.ndarray[Any, Any]] = []
    per_mesh_cap = max(1_000, max_points // 8)

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        center = data.geom_xpos[geom_id].astype(np.float32)
        rotation = data.geom_xmat[geom_id].reshape(3, 3).astype(np.float32)

        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            sampled_groups.append(
                _sample_plane(center, rotation, model.geom_size[geom_id], spacing)
            )
            continue

        if geom_type != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_convex_" in geom_name:
            continue

        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0:
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) or ""
            if mat_name == "mat_invisible":
                continue

        mesh_id = int(model.geom_dataid[geom_id])
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])

        local_vertices = model.mesh_vert[vert_start : vert_start + vert_count]
        world_vertices = local_vertices @ rotation.T + center
        faces = model.mesh_face[face_start : face_start + face_count]
        sampled_groups.append(
            _sample_mesh_surface(world_vertices, faces, spacing, per_mesh_cap, rng)
        )

    if not sampled_groups:
        raise RuntimeError(f"No sampleable MuJoCo geometry found for room '{room}'")

    points = np.vstack([group for group in sampled_groups if len(group) > 0]).astype(np.float32)
    if len(points) > max_points:
        selected = rng.choice(len(points), size=max_points, replace=False)
        points = points[selected]

    return points


@dataclass
class DroneSimConfig(ModuleConfig):
    """Configuration for the lightweight drone LiDAR simulator."""

    world_frame_id: str = "world"
    base_frame_id: str = "drone/base_link"
    lidar_frame_id: str = "drone/lidar_link"
    update_hz: float = 20.0
    lidar_hz: float = 5.0
    sensor_range: float = 3.0
    lidar_min_world_z: float | None = None
    lidar_max_world_z: float | None = 2.5
    start_x: float | None = None
    start_y: float | None = None
    start_z: float = 1.2
    use_mujoco_scene: bool = True
    mujoco_room: str | None = None
    mujoco_room_from_occupancy: str | None = None
    scene_sample_spacing: float = 0.14
    scene_max_points: int = 120_000
    scene_seed: int = 7
    floor_extent: float = 5.0
    floor_spacing: float = 0.2
    obstacle_spacing: float = 0.15


class DroneSimConnectionModule(Module[DroneSimConfig]):
    """Simulated drone source that publishes odometry and world-frame LiDAR."""

    default_config = DroneSimConfig
    config: DroneSimConfig

    cmd_vel: In[Twist]
    movecmd_twist: In[Twist]

    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]

    _running: bool
    _sim_thread: threading.Thread | None
    _velocity_lock: threading.Lock
    _velocity_cmd: Twist
    _position: Vector3
    _yaw: float
    _world_points: np.ndarray[Any, Any]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._running = False
        self._sim_thread = None
        self._velocity_lock = threading.Lock()
        self._velocity_cmd = Twist.zero()
        default_x, default_y = global_config.mujoco_start_pos_float
        start_x = self.config.start_x if self.config.start_x is not None else default_x
        start_y = self.config.start_y if self.config.start_y is not None else default_y
        self._position = Vector3(start_x, start_y, self.config.start_z)
        self._yaw = 0.0
        self._world_points = self._build_world_points()

    @rpc
    def start(self) -> None:
        super().start()
        if self._running:
            return

        self._running = True
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self._on_twist)))
        self._disposables.add(Disposable(self.movecmd_twist.subscribe(self._on_twist)))
        self._sim_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._sim_thread.start()
        logger.info("DroneSimConnectionModule started")

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        super().stop()

    def _on_twist(self, twist: Twist) -> None:
        with self._velocity_lock:
            self._velocity_cmd = Twist(twist)

    def _run_loop(self) -> None:
        update_period = 1.0 / self.config.update_hz
        lidar_period = 1.0 / self.config.lidar_hz
        last_t = time.monotonic()
        last_lidar_t = 0.0

        while self._running:
            now_mono = time.monotonic()
            dt = max(0.0, now_mono - last_t)
            last_t = now_mono

            self._integrate(dt)
            now = time.time()
            pose = self._pose(now)
            self._publish_tf(pose)
            self.odom.publish(pose)

            if now_mono - last_lidar_t >= lidar_period:
                last_lidar_t = now_mono
                self.lidar.publish(self._lidar_frame(now))

            elapsed = time.monotonic() - now_mono
            time.sleep(max(0.0, update_period - elapsed))

    def _integrate(self, dt: float) -> None:
        with self._velocity_lock:
            twist = Twist(self._velocity_cmd)

        self._yaw += twist.angular.z * dt
        cos_yaw = float(np.cos(self._yaw))
        sin_yaw = float(np.sin(self._yaw))

        # Treat Twist.linear as body-frame velocity: x=forward, y=left, z=up.
        world_vx = twist.linear.x * cos_yaw - twist.linear.y * sin_yaw
        world_vy = twist.linear.x * sin_yaw + twist.linear.y * cos_yaw
        self._position = Vector3(
            self._position.x + world_vx * dt,
            self._position.y + world_vy * dt,
            max(0.2, self._position.z + twist.linear.z * dt),
        )

    def _pose(self, ts: float) -> PoseStamped:
        return PoseStamped(
            ts=ts,
            frame_id=self.config.world_frame_id,
            position=self._position,
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, self._yaw)),
        )

    def _publish_tf(self, pose: PoseStamped) -> None:
        base_link = Transform(
            translation=pose.position,
            rotation=pose.orientation,
            frame_id=self.config.world_frame_id,
            child_frame_id=self.config.base_frame_id,
            ts=pose.ts,
        )
        lidar_link = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(),
            frame_id=self.config.base_frame_id,
            child_frame_id=self.config.lidar_frame_id,
            ts=pose.ts,
        )
        self.tf.publish(base_link, lidar_link)

    def _lidar_frame(self, ts: float) -> PointCloud2:
        center = np.array([self._position.x, self._position.y, self._position.z], dtype=np.float32)
        deltas = self._world_points - center
        distances = np.linalg.norm(deltas, axis=1)
        points = self._world_points[distances <= self.config.sensor_range]
        if self.config.lidar_min_world_z is not None:
            points = points[points[:, 2] >= self.config.lidar_min_world_z]
        if self.config.lidar_max_world_z is not None:
            # The sim LiDAR samples MuJoCo mesh surfaces and then applies a range gate; it is
            # not a true raycast sensor with occlusion/FOV. Without this navigation-height gate,
            # ceiling and upper-wall surfaces are accumulated as floating layers in Rerun/global_map
            # even though the 2D costmap ignores points above this height.
            points = points[points[:, 2] <= self.config.lidar_max_world_z]
        return PointCloud2.from_numpy(
            points.astype(np.float32),
            frame_id=self.config.world_frame_id,
            timestamp=ts,
        )

    def _build_world_points(self) -> np.ndarray[Any, Any]:
        if self.config.use_mujoco_scene:
            room = self.config.mujoco_room or global_config.mujoco_room or "office1"
            room_from_occupancy = (
                self.config.mujoco_room_from_occupancy or global_config.mujoco_room_from_occupancy
            )
            try:
                points = _build_mujoco_scene_points(
                    room,
                    room_from_occupancy,
                    self.config.scene_sample_spacing,
                    self.config.scene_max_points,
                    self.config.scene_seed,
                )
                logger.info(
                    "Loaded MuJoCo scene for drone sim",
                    room=room,
                    point_count=len(points),
                    bbox_min=np.min(points, axis=0).round(3).tolist(),
                    bbox_max=np.max(points, axis=0).round(3).tolist(),
                )
                return points
            except Exception as e:
                logger.warning(
                    "Falling back to procedural drone sim world",
                    room=room,
                    error=str(e),
                )

        return self._build_procedural_world_points()

    def _build_procedural_world_points(self) -> np.ndarray[Any, Any]:
        floor_axis = np.arange(
            -self.config.floor_extent,
            self.config.floor_extent + self.config.floor_spacing,
            self.config.floor_spacing,
            dtype=np.float32,
        )
        floor_x, floor_y = np.meshgrid(floor_axis, floor_axis)
        floor = np.column_stack(
            [
                floor_x.reshape(-1),
                floor_y.reshape(-1),
                np.zeros(floor_x.size, dtype=np.float32),
            ]
        )

        wall_y = np.arange(-2.2, 2.21, self.config.obstacle_spacing, dtype=np.float32)
        wall_z = np.arange(0.2, 2.01, self.config.obstacle_spacing, dtype=np.float32)
        wall_yy, wall_zz = np.meshgrid(wall_y, wall_z)
        wall = np.column_stack(
            [
                np.full(wall_yy.size, 2.2, dtype=np.float32),
                wall_yy.reshape(-1),
                wall_zz.reshape(-1),
            ]
        )

        box_axis = np.arange(-0.4, 0.41, self.config.obstacle_spacing, dtype=np.float32)
        box_x, box_y, box_z = np.meshgrid(box_axis, box_axis, np.arange(0.1, 1.1, 0.15))
        box = np.column_stack(
            [
                box_x.reshape(-1) - 1.3,
                box_y.reshape(-1) + 1.4,
                box_z.reshape(-1),
            ]
        )

        return np.vstack([floor, wall, box]).astype(np.float32)


drone_sim_connection = DroneSimConnectionModule.blueprint

__all__ = [
    "DroneSimConfig",
    "DroneSimConnectionModule",
    "drone_sim_connection",
]
