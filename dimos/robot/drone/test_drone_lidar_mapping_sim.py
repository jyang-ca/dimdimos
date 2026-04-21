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

"""Validation tests for the simulated drone LiDAR mapping blueprint."""

import numpy as np
import pytest

from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.health import (
    MappingHealthMonitor,
    compute_costmap_health,
    compute_global_map_health,
)
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.geometry_msgs import PoseStamped, Twist, Vector3
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.drone.blueprints.sim.drone_lidar_mapping_sim import drone_lidar_mapping_sim
from dimos.robot.drone.connection_module import DroneConnectionModule
from dimos.robot.drone.sim_connection_module import DroneSimConnectionModule
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


def test_drone_lidar_mapping_sim_blueprint_uses_simulated_connection() -> None:
    modules = {bp.module for bp in drone_lidar_mapping_sim.blueprints}

    assert DroneSimConnectionModule in modules
    assert VoxelGridMapper in modules
    assert CostMapper in modules
    assert ReplanningAStarPlanner in modules
    assert MappingHealthMonitor in modules
    assert DroneConnectionModule not in modules
    assert FastLio2 not in modules


def test_drone_lidar_mapping_sim_wires_click_goals_to_sim_velocity() -> None:
    endpoints = {
        (stream.name, stream.type, stream.direction, bp.module)
        for bp in drone_lidar_mapping_sim.blueprints
        for stream in bp.streams
    }

    assert ("goal_request", PoseStamped, "out", WebsocketVisModule) in endpoints
    assert ("goal_request", PoseStamped, "in", ReplanningAStarPlanner) in endpoints
    assert ("cmd_vel", Twist, "out", ReplanningAStarPlanner) in endpoints
    assert ("cmd_vel", Twist, "out", WebsocketVisModule) in endpoints
    assert ("cmd_vel", Twist, "in", DroneSimConnectionModule) in endpoints


def test_drone_sim_uses_go2_mujoco_office_frame_defaults() -> None:
    sim = DroneSimConnectionModule(sensor_range=4.0)
    try:
        start_x, start_y = global_config.mujoco_start_pos_float
        assert sim._position.x == pytest.approx(start_x)
        assert sim._position.y == pytest.approx(start_y)
        assert sim._position.z == pytest.approx(sim.config.start_z)

        bbox_min = sim._world_points.min(axis=0)
        bbox_max = sim._world_points.max(axis=0)
        bbox_extent = bbox_max - bbox_min

        assert sim.config.world_frame_id == "world"
        assert bbox_extent[0] > 15.0
        assert bbox_extent[1] > 15.0
        assert bbox_extent[2] > 3.0
        assert bbox_min[2] < 0.05
        assert bbox_max[2] > 3.0
    finally:
        sim.stop()


def test_drone_sim_lidar_filters_ceiling_layer_from_navigation_map() -> None:
    sim = DroneSimConnectionModule(use_mujoco_scene=False, sensor_range=10.0)
    try:
        sim._position = Vector3(0.0, 0.0, sim.config.start_z)
        sim._world_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.2],
                [2.0, 0.0, 2.5],
                [3.0, 0.0, 2.75],
                [4.0, 0.0, 3.2],
            ],
            dtype=np.float32,
        )

        frame = sim._lidar_frame(ts=10.0)
        points, _ = frame.as_numpy()
    finally:
        sim.stop()

    assert points[:, 2].max() <= 2.5 + 1e-6
    assert not np.any(np.isclose(points[:, 2], 2.75))
    assert not np.any(np.isclose(points[:, 2], 3.2))
    assert np.any(np.all(np.isclose(points, [2.0, 0.0, 2.5]), axis=1))


def test_drone_lidar_mapping_sim_metrics_from_reused_pipeline() -> None:
    sim = DroneSimConnectionModule(sensor_range=4.0)
    voxel_mapper = VoxelGridMapper(
        voxel_size=0.1,
        block_count=50_000,
        device="CPU:0",
        carve_columns=False,
    )
    cost_mapper = CostMapper(
        algo="simple",
        config=SimpleOccupancyConfig(resolution=0.1, min_height=0.15, max_height=2.5),
    )

    try:
        frame = sim._lidar_frame(ts=100.0)
        voxel_mapper.add_frame(frame)
        global_map = voxel_mapper.get_global_pointcloud2()
        global_map_metrics = compute_global_map_health(global_map, now=100.2)

        costmap = cost_mapper._calculate_costmap(global_map)
        costmap_metrics = compute_costmap_health(costmap, now=100.2)
    finally:
        sim.stop()
        voxel_mapper.stop()
        cost_mapper.stop()

    assert global_map_metrics.voxel_count == len(global_map)
    assert global_map_metrics.voxel_count > 500
    assert global_map_metrics.finite_point_ratio == pytest.approx(1.0)
    assert global_map_metrics.bbox_extent[0] > 4.0
    assert global_map_metrics.bbox_extent[1] > 4.0
    assert global_map_metrics.bbox_extent[2] > 1.5
    assert global_map_metrics.timestamp_lag_ms == pytest.approx(200.0)

    assert costmap_metrics.width > 20
    assert costmap_metrics.height > 20
    assert 0.0 < costmap_metrics.unknown_percent < 100.0
    assert 0.0 < costmap_metrics.occupied_percent < 100.0
    assert costmap_metrics.free_percent > 0.0
    assert costmap_metrics.bbox_extent[0] > global_map_metrics.bbox_extent[0]
    assert costmap_metrics.bbox_extent[1] > global_map_metrics.bbox_extent[1]
    assert costmap_metrics.timestamp_lag_ms == pytest.approx(200.0)
