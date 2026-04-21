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

"""Costmap fusion helpers for the Go2 + drone simulation blueprint."""

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.pointclouds.occupancy import simple_occupancy
from dimos.msgs.geometry_msgs import Pose
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2


@dataclass
class DroneObstacleLayerConfig(ModuleConfig):
    """Config for projecting a drone point cloud into Go2 obstacle semantics."""

    resolution: float = 0.1
    min_obstacle_height: float = 0.12
    max_obstacle_height: float = 1.6
    frame_id: str | None = None


class DroneMapToGo2ObstacleLayer(Module[DroneObstacleLayerConfig]):
    """Project the drone global map into a Go2 traversability layer."""

    default_config = DroneObstacleLayerConfig
    config: DroneObstacleLayerConfig

    global_map: In[PointCloud2]
    drone_obstacle_layer: Out[OccupancyGrid]

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.global_map.subscribe(self._on_global_map)))

    def _on_global_map(self, msg: PointCloud2) -> None:
        self.drone_obstacle_layer.publish(self._calculate_obstacle_layer(msg))

    def _calculate_obstacle_layer(self, msg: PointCloud2) -> OccupancyGrid:
        return simple_occupancy(
            msg,
            resolution=self.config.resolution,
            min_height=self.config.min_obstacle_height,
            max_height=self.config.max_obstacle_height,
            frame_id=self.config.frame_id or msg.frame_id,
        )


class Go2CostmapFusion(Module):
    """Fuse Go2's local costmap with the drone-derived traversability layer."""

    global_costmap: In[OccupancyGrid]
    drone_obstacle_layer: In[OccupancyGrid]
    fused_global_costmap: Out[OccupancyGrid]

    _latest_global_costmap: OccupancyGrid | None
    _latest_drone_obstacle_layer: OccupancyGrid | None
    _lock: threading.Lock

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_global_costmap = None
        self._latest_drone_obstacle_layer = None
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.global_costmap.subscribe(self._on_global_costmap)))
        self._disposables.add(
            Disposable(self.drone_obstacle_layer.subscribe(self._on_drone_obstacle_layer))
        )

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._latest_global_costmap = msg
            obstacle_layer = self._latest_drone_obstacle_layer
        if obstacle_layer is not None:
            self.fused_global_costmap.publish(self._fuse_costmaps(msg, obstacle_layer))

    def _on_drone_obstacle_layer(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._latest_drone_obstacle_layer = msg
            global_costmap = self._latest_global_costmap
        if global_costmap is not None:
            self.fused_global_costmap.publish(self._fuse_costmaps(global_costmap, msg))

    def _fuse_costmaps(
        self,
        base: OccupancyGrid,
        drone_obstacle_layer: OccupancyGrid,
    ) -> OccupancyGrid:
        if _same_grid_geometry(base, drone_obstacle_layer):
            fused_grid = _merge_drone_layer(base.grid.copy(), drone_obstacle_layer.grid)
            return OccupancyGrid(
                grid=fused_grid,
                resolution=base.resolution,
                origin=base.origin,
                frame_id=base.frame_id,
                ts=max(base.ts, drone_obstacle_layer.ts),
            )

        resolution = base.resolution
        origin_x = min(base.origin.position.x, drone_obstacle_layer.origin.position.x)
        origin_y = min(base.origin.position.y, drone_obstacle_layer.origin.position.y)
        max_x = max(
            base.origin.position.x + base.width * base.resolution,
            drone_obstacle_layer.origin.position.x
            + drone_obstacle_layer.width * drone_obstacle_layer.resolution,
        )
        max_y = max(
            base.origin.position.y + base.height * base.resolution,
            drone_obstacle_layer.origin.position.y
            + drone_obstacle_layer.height * drone_obstacle_layer.resolution,
        )
        width = max(1, int(np.ceil((max_x - origin_x) / resolution)))
        height = max(1, int(np.ceil((max_y - origin_y) / resolution)))

        result_grid = np.full((height, width), int(CostValues.UNKNOWN), dtype=np.int8)
        base_x = int(np.floor((base.origin.position.x - origin_x) / resolution + 0.5))
        base_y = int(np.floor((base.origin.position.y - origin_y) / resolution + 0.5))
        result_grid[
            base_y : base_y + base.height,
            base_x : base_x + base.width,
        ] = base.grid

        _project_drone_layer(result_grid, drone_obstacle_layer, origin_x, origin_y, resolution)

        origin = Pose()
        origin.position.x = origin_x
        origin.position.y = origin_y
        origin.position.z = base.origin.position.z
        origin.orientation.x = base.origin.orientation.x
        origin.orientation.y = base.origin.orientation.y
        origin.orientation.z = base.origin.orientation.z
        origin.orientation.w = base.origin.orientation.w

        return OccupancyGrid(
            grid=result_grid,
            resolution=resolution,
            origin=origin,
            frame_id=base.frame_id,
            ts=max(base.ts, drone_obstacle_layer.ts),
        )


def _same_grid_geometry(a: OccupancyGrid, b: OccupancyGrid) -> bool:
    return bool(
        a.grid.shape == b.grid.shape
        and np.isclose(a.resolution, b.resolution)
        and np.isclose(a.origin.position.x, b.origin.position.x)
        and np.isclose(a.origin.position.y, b.origin.position.y)
    )


def _merge_drone_layer(
    base_grid: np.ndarray,  # type: ignore[type-arg]
    drone_grid: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    drone_free = drone_grid == CostValues.FREE
    base_grid[(base_grid == CostValues.UNKNOWN) & drone_free] = int(CostValues.FREE)
    base_grid[drone_grid >= CostValues.OCCUPIED] = int(CostValues.OCCUPIED)
    return base_grid


def _project_drone_layer(
    result_grid: np.ndarray,  # type: ignore[type-arg]
    drone_layer: OccupancyGrid,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    traversable_y, traversable_x = np.nonzero(drone_layer.grid == CostValues.FREE)
    _project_drone_cells(
        result_grid,
        drone_layer,
        traversable_x,
        traversable_y,
        origin_x,
        origin_y,
        resolution,
        int(CostValues.FREE),
        only_unknown=True,
    )

    obstacle_y, obstacle_x = np.nonzero(drone_layer.grid >= CostValues.OCCUPIED)
    _project_drone_cells(
        result_grid,
        drone_layer,
        obstacle_x,
        obstacle_y,
        origin_x,
        origin_y,
        resolution,
        int(CostValues.OCCUPIED),
        only_unknown=False,
    )


def _project_drone_cells(
    result_grid: np.ndarray,  # type: ignore[type-arg]
    drone_layer: OccupancyGrid,
    cell_x: np.ndarray,  # type: ignore[type-arg]
    cell_y: np.ndarray,  # type: ignore[type-arg]
    origin_x: float,
    origin_y: float,
    resolution: float,
    value: int,
    *,
    only_unknown: bool,
) -> None:
    if cell_x.size == 0:
        return

    world_x = drone_layer.origin.position.x + cell_x.astype(np.float64) * drone_layer.resolution
    world_y = drone_layer.origin.position.y + cell_y.astype(np.float64) * drone_layer.resolution
    result_x = ((world_x - origin_x) / resolution + 0.5).astype(np.int64)
    result_y = ((world_y - origin_y) / resolution + 0.5).astype(np.int64)
    height, width = result_grid.shape
    valid = (
        (result_x >= 0)
        & (result_x < width)
        & (result_y >= 0)
        & (result_y < height)
    )
    valid_x = result_x[valid]
    valid_y = result_y[valid]

    if only_unknown:
        unknown = result_grid[valid_y, valid_x] == CostValues.UNKNOWN
        result_grid[valid_y[unknown], valid_x[unknown]] = value
    else:
        result_grid[valid_y, valid_x] = value


drone_map_to_go2_obstacle_layer = DroneMapToGo2ObstacleLayer.blueprint
go2_costmap_fusion = Go2CostmapFusion.blueprint

__all__ = [
    "DroneMapToGo2ObstacleLayer",
    "DroneObstacleLayerConfig",
    "Go2CostmapFusion",
    "drone_map_to_go2_obstacle_layer",
    "go2_costmap_fusion",
]
