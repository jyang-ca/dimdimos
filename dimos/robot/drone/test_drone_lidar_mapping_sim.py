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
from dimos.mapping.health import (
    MappingHealthMonitor,
    compute_costmap_health,
    compute_global_map_health,
)
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.geometry_msgs import PoseStamped, Twist, Vector3
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.drone.blueprints.sim.drone_lidar_mapping_sim import drone_lidar_mapping_sim
from dimos.robot.drone.connection_module import DroneConnectionModule
from dimos.robot.drone.flight_costmapper import DroneFlightCostMapper
from dimos.robot.drone.sim_connection_module import DroneSimConnectionModule
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


def _pointcloud(points: list[list[float]], ts: float = 100.0) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray(points, dtype=np.float32), frame_id="world", timestamp=ts
    )


def _grid_value_at(grid: OccupancyGrid, x: float, y: float) -> int:
    gx = int((x - grid.origin.position.x) / grid.resolution + 0.5)
    gy = int((y - grid.origin.position.y) / grid.resolution + 0.5)
    return int(grid.grid[gy, gx])


def test_drone_lidar_mapping_sim_blueprint_uses_simulated_connection() -> None:
    modules = {bp.module for bp in drone_lidar_mapping_sim.blueprints}

    assert DroneSimConnectionModule in modules
    assert VoxelGridMapper in modules
    assert DroneFlightCostMapper in modules
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


def test_drone_lidar_mapping_sim_namespaces_mapping_outputs() -> None:
    assert (
        drone_lidar_mapping_sim.remapping_map[(VoxelGridMapper, "global_map")] == "drone/global_map"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(DroneFlightCostMapper, "global_map")]
        == "drone/global_map"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(MappingHealthMonitor, "global_map")]
        == "drone/global_map"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(DroneFlightCostMapper, "global_costmap")]
        == "drone/global_costmap"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(ReplanningAStarPlanner, "global_costmap")]
        == "drone/global_costmap"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(MappingHealthMonitor, "global_costmap")]
        == "drone/global_costmap"
    )
    assert (
        drone_lidar_mapping_sim.remapping_map[(WebsocketVisModule, "global_costmap")]
        == "drone/global_costmap"
    )


def test_drone_flight_costmap_marks_low_obstacles_free() -> None:
    mapper = DroneFlightCostMapper(clearance_below=0.25, clearance_above=0.25)
    cloud = _pointcloud([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    try:
        grid = mapper._calculate_costmap(cloud, flight_z=1.2)
    finally:
        mapper.stop()

    assert _grid_value_at(grid, 0.0, 0.0) == int(CostValues.FREE)


def test_drone_flight_costmap_marks_current_height_band_occupied() -> None:
    mapper = DroneFlightCostMapper(clearance_below=0.25, clearance_above=0.25)
    cloud = _pointcloud([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])

    try:
        grid = mapper._calculate_costmap(cloud, flight_z=1.2)
    finally:
        mapper.stop()

    assert _grid_value_at(grid, 0.0, 0.0) == int(CostValues.OCCUPIED)


def test_drone_flight_costmap_high_obstacle_policy_is_configurable() -> None:
    cloud = _pointcloud([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])
    ignore_mapper = DroneFlightCostMapper(
        clearance_below=0.25,
        clearance_above=0.25,
        high_obstacle_policy="ignore",
    )
    occupied_mapper = DroneFlightCostMapper(
        clearance_below=0.25,
        clearance_above=0.25,
        high_obstacle_policy="occupied",
    )

    try:
        ignore_grid = ignore_mapper._calculate_costmap(cloud, flight_z=1.2)
        occupied_grid = occupied_mapper._calculate_costmap(cloud, flight_z=1.2)
    finally:
        ignore_mapper.stop()
        occupied_mapper.stop()

    assert _grid_value_at(ignore_grid, 0.0, 0.0) == int(CostValues.FREE)
    assert _grid_value_at(occupied_grid, 0.0, 0.0) == int(CostValues.OCCUPIED)


def test_drone_flight_costmap_changes_with_odom_height() -> None:
    mapper = DroneFlightCostMapper(clearance_below=0.25, clearance_above=0.25)
    cloud = _pointcloud([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])

    try:
        low_flight_grid = mapper._calculate_costmap(cloud, flight_z=1.2)
        high_flight_grid = mapper._calculate_costmap(cloud, flight_z=1.8)
    finally:
        mapper.stop()

    assert _grid_value_at(low_flight_grid, 0.0, 0.0) == int(CostValues.OCCUPIED)
    assert _grid_value_at(high_flight_grid, 0.0, 0.0) == int(CostValues.FREE)


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
    cost_mapper = DroneFlightCostMapper(resolution=0.1, clearance_below=0.25, clearance_above=0.25)

    try:
        frame = sim._lidar_frame(ts=100.0)
        voxel_mapper.add_frame(frame)
        global_map = voxel_mapper.get_global_pointcloud2()
        global_map_metrics = compute_global_map_health(global_map, now=100.2)

        costmap = cost_mapper._calculate_costmap(global_map, flight_z=sim._position.z)
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
