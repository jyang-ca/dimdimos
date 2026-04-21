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

"""Tests for the Go2 + drone LiDAR fusion costmap pipeline."""

import numpy as np

from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.geometry_msgs import Pose
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.robot.drone.sim_connection_module import DroneSimConnectionModule
from dimos.robot.unitree.go2.blueprints.sim.drone_go2_lidar_fusion_sim import (
    DroneFusionPlanner,
    DroneFusionVoxelGridMapper,
    Go2FusionCostMapper,
    Go2FusionPlanner,
    Go2FusionVoxelGridMapper,
    drone_go2_lidar_fusion_sim,
)
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.drone_fusion_costmap import (
    DroneMapToGo2ObstacleLayer,
    Go2CostmapFusion,
)
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


def _pointcloud(points: list[list[float]], ts: float = 100.0) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray(points, dtype=np.float32), frame_id="world", timestamp=ts
    )


def _origin(x: float = 0.0, y: float = 0.0) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.orientation.w = 1.0
    return pose


def _grid(
    values: np.ndarray,  # type: ignore[type-arg]
    *,
    resolution: float = 0.5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    ts: float = 100.0,
) -> OccupancyGrid:
    return OccupancyGrid(
        grid=values.astype(np.int8),
        resolution=resolution,
        origin=_origin(origin_x, origin_y),
        frame_id="world",
        ts=ts,
    )


def _grid_value_at(grid: OccupancyGrid, x: float, y: float) -> int:
    gx = int((x - grid.origin.position.x) / grid.resolution + 0.5)
    gy = int((y - grid.origin.position.y) / grid.resolution + 0.5)
    return int(grid.grid[gy, gx])


def test_drone_map_to_go2_obstacle_layer_marks_go2_height_obstacles() -> None:
    mapper = DroneMapToGo2ObstacleLayer(
        resolution=0.1,
        min_obstacle_height=0.12,
        max_obstacle_height=1.6,
    )
    cloud = _pointcloud(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.8],
            [2.0, 0.0, 2.2],
        ]
    )

    try:
        layer = mapper._calculate_obstacle_layer(cloud)
    finally:
        mapper.stop()

    assert _grid_value_at(layer, 0.0, 0.0) == int(CostValues.FREE)
    assert _grid_value_at(layer, 1.0, 0.0) == int(CostValues.OCCUPIED)
    assert _grid_value_at(layer, 2.0, 0.0) == int(CostValues.UNKNOWN)


def test_drone_map_to_go2_obstacle_layer_publishes_each_drone_map_update() -> None:
    mapper = DroneMapToGo2ObstacleLayer(
        resolution=0.1,
        min_obstacle_height=0.12,
        max_obstacle_height=1.6,
    )
    published: list[OccupancyGrid] = []
    mapper.drone_obstacle_layer.subscribe(published.append)

    try:
        mapper._on_global_map(_pointcloud([[0.0, 0.0, 0.8]], ts=100.0))
        mapper._on_global_map(_pointcloud([[0.5, 0.0, 0.8]], ts=101.0))
    finally:
        mapper.stop()

    assert len(published) == 2
    assert published[0].ts == 100.0
    assert published[1].ts == 101.0


def test_go2_costmap_fusion_overlays_drone_obstacles_without_clearing_base() -> None:
    base_values = np.full((5, 5), int(CostValues.FREE), dtype=np.int8)
    base_values[1, 1] = int(CostValues.OCCUPIED)
    base = _grid(base_values, ts=100.0)

    drone_values = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[1, 1] = int(CostValues.FREE)
    drone_values[3, 3] = int(CostValues.OCCUPIED)
    drone_layer = _grid(drone_values, ts=101.0)

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert int(fused.grid[1, 1]) == int(CostValues.OCCUPIED)
    assert int(fused.grid[3, 3]) == int(CostValues.OCCUPIED)
    assert int(fused.grid[0, 0]) == int(CostValues.FREE)
    assert fused.ts == 101.0


def test_go2_costmap_fusion_promotes_drone_free_unknown_space() -> None:
    base_values = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    base_values[1, 1] = int(CostValues.OCCUPIED)
    base_values[2, 2] = int(CostValues.FREE)
    base = _grid(base_values, ts=100.0)

    drone_values = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[1, 1] = int(CostValues.FREE)
    drone_values[2, 2] = int(CostValues.FREE)
    drone_values[3, 3] = int(CostValues.FREE)
    drone_layer = _grid(drone_values, ts=101.0)

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert int(fused.grid[1, 1]) == int(CostValues.OCCUPIED)
    assert int(fused.grid[2, 2]) == int(CostValues.FREE)
    assert int(fused.grid[3, 3]) == int(CostValues.FREE)
    assert fused.ts == 101.0


def test_go2_costmap_fusion_republishes_when_drone_layer_updates() -> None:
    base = _grid(np.full((5, 5), int(CostValues.FREE), dtype=np.int8), ts=100.0)

    layer_one_values = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    layer_one_values[1, 1] = int(CostValues.OCCUPIED)
    layer_one = _grid(layer_one_values, ts=101.0)

    layer_two_values = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    layer_two_values[3, 3] = int(CostValues.OCCUPIED)
    layer_two = _grid(layer_two_values, ts=102.0)

    fusion = Go2CostmapFusion()
    published: list[OccupancyGrid] = []
    fusion.fused_global_costmap.subscribe(published.append)

    try:
        fusion._on_global_costmap(base)
        fusion._on_drone_obstacle_layer(layer_one)
        fusion._on_drone_obstacle_layer(layer_two)
    finally:
        fusion.stop()

    assert len(published) == 2
    assert int(published[0].grid[1, 1]) == int(CostValues.OCCUPIED)
    assert int(published[1].grid[3, 3]) == int(CostValues.OCCUPIED)
    assert published[0].ts == 101.0
    assert published[1].ts == 102.0


def test_go2_costmap_fusion_projects_different_layer_geometry_into_base() -> None:
    base = _grid(
        np.full((6, 6), int(CostValues.FREE), dtype=np.int8),
        resolution=0.5,
        origin_x=-1.0,
        origin_y=-1.0,
        ts=100.0,
    )
    drone_values = np.full((3, 3), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[0, 0] = int(CostValues.OCCUPIED)
    drone_layer = _grid(drone_values, resolution=0.5, origin_x=0.0, origin_y=0.0, ts=99.0)

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert _grid_value_at(fused, 0.0, 0.0) == int(CostValues.OCCUPIED)
    assert int(fused.grid[0, 0]) == int(CostValues.FREE)
    assert fused.ts == 100.0


def test_go2_costmap_fusion_expands_to_drone_layer_bounds() -> None:
    base = _grid(
        np.full((2, 2), int(CostValues.FREE), dtype=np.int8),
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        ts=100.0,
    )
    drone_values = np.full((2, 2), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[0, 0] = int(CostValues.OCCUPIED)
    drone_layer = _grid(
        drone_values,
        resolution=1.0,
        origin_x=3.0,
        origin_y=0.0,
        ts=101.0,
    )

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert fused.width == 5
    assert fused.height == 2
    assert fused.origin.position.x == 0.0
    assert fused.origin.position.y == 0.0
    assert _grid_value_at(fused, 0.0, 0.0) == int(CostValues.FREE)
    assert _grid_value_at(fused, 2.0, 0.0) == int(CostValues.UNKNOWN)
    assert _grid_value_at(fused, 3.0, 0.0) == int(CostValues.OCCUPIED)
    assert fused.ts == 101.0


def test_go2_costmap_fusion_projects_drone_free_outside_base_bounds() -> None:
    base = _grid(
        np.full((2, 2), int(CostValues.UNKNOWN), dtype=np.int8),
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        ts=100.0,
    )
    drone_values = np.full((2, 2), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[0, 0] = int(CostValues.FREE)
    drone_values[0, 1] = int(CostValues.OCCUPIED)
    drone_layer = _grid(
        drone_values,
        resolution=1.0,
        origin_x=3.0,
        origin_y=0.0,
        ts=101.0,
    )

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert _grid_value_at(fused, 3.0, 0.0) == int(CostValues.FREE)
    assert _grid_value_at(fused, 4.0, 0.0) == int(CostValues.OCCUPIED)
    assert fused.ts == 101.0


def test_go2_costmap_fusion_handles_lcm_decoded_origins() -> None:
    base = OccupancyGrid.lcm_decode(
        _grid(
            np.full((2, 2), int(CostValues.FREE), dtype=np.int8),
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            ts=100.0,
        ).lcm_encode()
    )
    drone_values = np.full((2, 2), int(CostValues.UNKNOWN), dtype=np.int8)
    drone_values[0, 0] = int(CostValues.OCCUPIED)
    drone_layer = OccupancyGrid.lcm_decode(
        _grid(
            drone_values,
            resolution=1.0,
            origin_x=3.0,
            origin_y=0.0,
            ts=101.0,
        ).lcm_encode()
    )

    fusion = Go2CostmapFusion()
    try:
        fused = fusion._fuse_costmaps(base, drone_layer)
    finally:
        fusion.stop()

    assert fused.origin.position.x == 0.0
    assert fused.origin.position.y == 0.0
    assert _grid_value_at(fused, 3.0, 0.0) == int(CostValues.OCCUPIED)
    assert fused.ts == 101.0


def test_drone_go2_lidar_fusion_blueprint_scopes_fusion_modules() -> None:
    fusion_modules = {bp.module for bp in drone_go2_lidar_fusion_sim.blueprints}
    standalone_go2_modules = {bp.module for bp in unitree_go2.blueprints}

    assert GO2Connection in fusion_modules
    assert DroneSimConnectionModule in fusion_modules
    assert Go2FusionVoxelGridMapper in fusion_modules
    assert DroneFusionVoxelGridMapper in fusion_modules
    assert Go2FusionPlanner in fusion_modules
    assert DroneFusionPlanner in fusion_modules
    assert Go2CostmapFusion in fusion_modules
    assert DroneMapToGo2ObstacleLayer in fusion_modules
    assert VoxelGridMapper not in fusion_modules

    assert Go2CostmapFusion not in standalone_go2_modules
    assert DroneMapToGo2ObstacleLayer not in standalone_go2_modules


def test_drone_go2_lidar_fusion_blueprint_uses_go2_mujoco_model() -> None:
    assert drone_go2_lidar_fusion_sim.global_config_overrides["simulation"] is True
    assert drone_go2_lidar_fusion_sim.global_config_overrides["robot_model"] == "unitree_go2"


def test_drone_go2_lidar_fusion_blueprint_routes_go2_planner_to_fused_costmap() -> None:
    remappings = drone_go2_lidar_fusion_sim.remapping_map

    assert remappings[(GO2Connection, "lidar")] == "go2/lidar"
    assert remappings[(GO2Connection, "odom")] == "go2/odom"
    assert remappings[(Go2FusionVoxelGridMapper, "global_map")] == "go2/global_map"
    assert remappings[(Go2FusionCostMapper, "global_costmap")] == "go2/global_costmap"
    assert remappings[(DroneSimConnectionModule, "lidar")] == "drone/lidar"
    assert remappings[(DroneFusionVoxelGridMapper, "global_map")] == "drone/global_map"
    assert remappings[(DroneMapToGo2ObstacleLayer, "global_map")] == "drone/global_map"
    assert (
        remappings[(DroneMapToGo2ObstacleLayer, "drone_obstacle_layer")]
        == "go2/drone_obstacle_layer"
    )
    assert remappings[(Go2CostmapFusion, "global_costmap")] == "go2/global_costmap"
    assert (
        remappings[(Go2CostmapFusion, "drone_obstacle_layer")]
        == "go2/drone_obstacle_layer"
    )
    assert remappings[(Go2CostmapFusion, "fused_global_costmap")] == "go2/fused_global_costmap"
    assert remappings[(Go2FusionPlanner, "global_costmap")] == "go2/fused_global_costmap"
    assert remappings[(Go2FusionPlanner, "goal_request")] == "go2/goal_request"
    assert remappings[(Go2FusionPlanner, "cmd_vel")] == "go2/cmd_vel"
    assert remappings[(DroneFusionPlanner, "global_costmap")] == "drone/global_costmap"
    assert remappings[(DroneFusionPlanner, "goal_request")] == "drone/goal_request"
    assert remappings[(DroneFusionPlanner, "cmd_vel")] == "drone/cmd_vel"
    assert remappings[(DroneFusionPlanner, "path")] == "drone/path"
    assert remappings[(WebsocketVisModule, "goal_request")] == "go2/goal_request"
    assert remappings[(WebsocketVisModule, "go2_goal_request")] == "go2/goal_request"
    assert remappings[(WebsocketVisModule, "drone_goal_request")] == "drone/goal_request"
    assert remappings[(WebsocketVisModule, "drone_global_costmap")] == "drone/global_costmap"
    assert remappings[(WebsocketVisModule, "go2_global_costmap")] == "go2/fused_global_costmap"
    assert remappings[(WebsocketVisModule, "global_costmap")] == "go2/fused_global_costmap"
