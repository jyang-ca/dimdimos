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

"""Validation tests for the drone LiDAR mapping blueprint."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.health import (
    MappingHealthMonitor,
    compute_costmap_health,
    compute_global_map_health,
)
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.robot.drone.blueprints.mapping.drone_lidar_mapping import drone_lidar_mapping
from dimos.robot.drone.camera_module import DroneCameraModule
from dimos.robot.drone.connection_module import DroneConnectionModule

if TYPE_CHECKING:
    from dimos.core.module import Module


def _synthetic_drone_lidar_frame(ts: float = 100.0) -> PointCloud2:
    xs = np.linspace(-1.0, 1.0, 21)
    ys = np.linspace(-1.0, 1.0, 21)
    ground_x, ground_y = np.meshgrid(xs, ys)
    ground = np.column_stack(
        [
            ground_x.reshape(-1),
            ground_y.reshape(-1),
            np.zeros(ground_x.size),
        ]
    )

    wall_y = np.linspace(-0.75, 0.75, 16)
    wall_z = np.linspace(0.3, 1.2, 10)
    obstacle_y, obstacle_z = np.meshgrid(wall_y, wall_z)
    obstacle = np.column_stack(
        [
            np.full(obstacle_y.size, 0.5),
            obstacle_y.reshape(-1),
            obstacle_z.reshape(-1),
        ]
    )

    points = np.vstack([ground, obstacle]).astype(np.float32)
    return PointCloud2.from_numpy(points, frame_id="world", timestamp=ts)


def test_drone_lidar_mapping_blueprint_reuses_existing_mapping_modules() -> None:
    modules = {bp.module for bp in drone_lidar_mapping.blueprints}
    fastlio2_module = cast("type[Module[Any]]", FastLio2)

    assert DroneConnectionModule in modules
    assert DroneCameraModule in modules
    assert FastLio2 in modules
    assert VoxelGridMapper in modules
    assert CostMapper in modules
    assert MappingHealthMonitor in modules
    assert (
        drone_lidar_mapping.remapping_map[(fastlio2_module, "global_map")] == "fastlio_global_map"
    )


def test_drone_lidar_mapping_health_metrics_from_reused_pipeline() -> None:
    frame = _synthetic_drone_lidar_frame(ts=100.0)
    voxel_mapper = VoxelGridMapper(
        voxel_size=0.1,
        block_count=20_000,
        device="CPU:0",
        carve_columns=False,
    )
    cost_mapper = CostMapper(
        algo="simple",
        config=SimpleOccupancyConfig(resolution=0.1, min_height=0.2, max_height=1.5),
    )

    try:
        voxel_mapper.add_frame(frame)
        global_map = voxel_mapper.get_global_pointcloud2()
        global_map_metrics = compute_global_map_health(global_map, now=100.25)

        costmap = cost_mapper._calculate_costmap(global_map)
        costmap_metrics = compute_costmap_health(costmap, now=100.25)
    finally:
        voxel_mapper.stop()
        cost_mapper.stop()

    assert global_map_metrics.voxel_count == len(global_map)
    assert global_map_metrics.voxel_count > 400
    assert global_map_metrics.finite_point_ratio == pytest.approx(1.0)
    assert global_map_metrics.bbox_extent[0] > 1.9
    assert global_map_metrics.bbox_extent[1] > 1.9
    assert global_map_metrics.bbox_extent[2] > 1.0
    assert global_map_metrics.timestamp_lag_ms == pytest.approx(250.0)

    assert costmap_metrics.width > 10
    assert costmap_metrics.height > 10
    assert 0.0 < costmap_metrics.unknown_percent < 100.0
    assert 0.0 < costmap_metrics.occupied_percent < 100.0
    assert costmap_metrics.free_percent > 0.0
    assert costmap_metrics.bbox_extent[0] > global_map_metrics.bbox_extent[0]
    assert costmap_metrics.bbox_extent[1] > global_map_metrics.bbox_extent[1]
    assert costmap_metrics.timestamp_lag_ms == pytest.approx(250.0)
