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

"""Drone-specific costmap projection for fixed-altitude flight."""

from dataclasses import dataclass
import threading
from typing import Any, Literal

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import Pose, PoseStamped
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2

HighObstaclePolicy = Literal["ignore", "occupied"]


@dataclass
class DroneFlightCostConfig(ModuleConfig):
    """Configuration for flight-band occupancy projection."""

    resolution: float = 0.1
    bounds_padding: float = 1.0
    default_flight_z: float = 1.2
    clearance_below: float = 0.25
    clearance_above: float = 0.25
    high_obstacle_policy: HighObstaclePolicy = "ignore"
    odom_republish_z_delta: float = 0.05


class DroneFlightCostMapper(Module[DroneFlightCostConfig]):
    """Build a 2D costmap for the drone's current flight altitude.

    The map treats points below the drone body band as passable and only marks
    cells occupied when points intersect the current vertical flight band.
    """

    default_config = DroneFlightCostConfig
    config: DroneFlightCostConfig

    global_map: In[PointCloud2]
    odom: In[PoseStamped]
    global_costmap: Out[OccupancyGrid]

    _latest_global_map: PointCloud2 | None
    _latest_flight_z: float
    _last_published_z: float | None
    _lock: threading.Lock

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_global_map = None
        self._latest_flight_z = self.config.default_flight_z
        self._last_published_z = None
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.global_map.subscribe(self._on_global_map)))
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

    def _on_global_map(self, msg: PointCloud2) -> None:
        with self._lock:
            self._latest_global_map = msg
            flight_z = self._latest_flight_z
        self._publish_costmap(msg, flight_z)

    def _on_odom(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_flight_z = msg.position.z
            global_map = self._latest_global_map
            last_published_z = self._last_published_z

        if global_map is None:
            return
        if last_published_z is not None:
            if abs(msg.position.z - last_published_z) < self.config.odom_republish_z_delta:
                return

        self._publish_costmap(global_map, msg.position.z)

    def _publish_costmap(self, msg: PointCloud2, flight_z: float) -> None:
        grid = self._calculate_costmap(msg, flight_z=flight_z)
        self.global_costmap.publish(grid)
        with self._lock:
            self._last_published_z = flight_z

    def _calculate_costmap(self, msg: PointCloud2, *, flight_z: float) -> OccupancyGrid:
        if self.config.high_obstacle_policy not in {"ignore", "occupied"}:
            raise ValueError(
                "high_obstacle_policy must be either 'ignore' or 'occupied', "
                f"got {self.config.high_obstacle_policy!r}"
            )

        points, _ = msg.as_numpy()
        points = points.astype(np.float64)
        if points.size == 0:
            return OccupancyGrid(
                width=1,
                height=1,
                resolution=self.config.resolution,
                frame_id=msg.frame_id,
                ts=msg.ts,
            )

        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        if len(points) == 0:
            return OccupancyGrid(
                width=1,
                height=1,
                resolution=self.config.resolution,
                frame_id=msg.frame_id,
                ts=msg.ts,
            )

        padding = self.config.bounds_padding
        min_x = float(np.min(points[:, 0])) - padding
        max_x = float(np.max(points[:, 0])) + padding
        min_y = float(np.min(points[:, 1])) - padding
        max_y = float(np.max(points[:, 1])) + padding
        width = max(1, int(np.ceil((max_x - min_x) / self.config.resolution)))
        height = max(1, int(np.ceil((max_y - min_y) / self.config.resolution)))

        origin = Pose()
        origin.position.x = min_x
        origin.position.y = min_y
        origin.position.z = 0.0
        origin.orientation.w = 1.0

        grid = np.full((height, width), int(CostValues.UNKNOWN), dtype=np.int8)
        inv_res = 1.0 / self.config.resolution
        grid_x = ((points[:, 0] - min_x) * inv_res + 0.5).astype(np.int64)
        grid_y = ((points[:, 1] - min_y) * inv_res + 0.5).astype(np.int64)
        valid = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)

        if not np.any(valid):
            return OccupancyGrid(
                grid=grid,
                resolution=self.config.resolution,
                origin=origin,
                frame_id=msg.frame_id,
                ts=msg.ts,
            )

        grid_x = grid_x[valid]
        grid_y = grid_y[valid]
        z = points[valid, 2]
        lower_z = flight_z - self.config.clearance_below
        upper_z = flight_z + self.config.clearance_above

        free_mask = z < lower_z
        grid[grid_y[free_mask], grid_x[free_mask]] = int(CostValues.FREE)

        occupied_mask = (z >= lower_z) & (z <= upper_z)
        if self.config.high_obstacle_policy == "occupied":
            occupied_mask |= z > upper_z
        grid[grid_y[occupied_mask], grid_x[occupied_mask]] = int(CostValues.OCCUPIED)

        return OccupancyGrid(
            grid=grid,
            resolution=self.config.resolution,
            origin=origin,
            frame_id=msg.frame_id,
            ts=msg.ts,
        )


drone_flight_cost_mapper = DroneFlightCostMapper.blueprint

__all__ = [
    "DroneFlightCostConfig",
    "DroneFlightCostMapper",
    "drone_flight_cost_mapper",
]
