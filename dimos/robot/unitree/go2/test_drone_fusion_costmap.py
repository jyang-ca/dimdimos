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

from collections.abc import Callable
import json
import threading
import time

import numpy as np

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.global_config import GlobalConfig
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.geometry_msgs import Pose, PoseStamped
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid, Path
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner
from dimos.robot.drone.sim_connection_module import DroneSimConnectionModule
from dimos.robot.unitree.go2.blueprints.sim.drone_go2_lidar_fusion_sim import (
    DroneFusionPlanner,
    DroneFusionVoxelGridMapper,
    Go2FusionCostMapper,
    Go2FusionPlanner,
    Go2FusionVoxelGridMapper,
    drone_go2_lidar_fusion_sim,
)
from dimos.robot.unitree.go2.blueprints.sim.drone_go2_lidar_fusion_sim_agentic_mcp import (
    drone_go2_lidar_fusion_sim_agentic_mcp,
)
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.drone_fusion_costmap import (
    DroneMapToGo2ObstacleLayer,
    Go2CostmapFusion,
)
from dimos.robot.unitree.go2.drone_go2_fusion_skills import (
    DroneGo2FusionSkillContainer,
    _ScoutResult,
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


def _pose(x: float, y: float, z: float = 0.0, ts: float = 100.0) -> PoseStamped:
    return PoseStamped(
        ts=ts,
        frame_id="world",
        position=(x, y, z),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )


def _publish_drone_path_after(
    skills: DroneGo2FusionSkillContainer,
    delay_s: float,
    poses: list[PoseStamped],
) -> threading.Timer:
    return _run_after(
        delay_s,
        lambda: skills._on_drone_path(Path(frame_id="world", poses=poses)),
    )


def _run_after(delay_s: float, callback: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay_s, callback)
    timer.start()
    return timer


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
    assert remappings[(DroneSimConnectionModule, "lidar")] == "drone/lidar"  # type: ignore[index]
    assert remappings[(DroneFusionVoxelGridMapper, "global_map")] == "drone/global_map"
    assert remappings[(DroneMapToGo2ObstacleLayer, "global_map")] == "drone/global_map"  # type: ignore[index]
    assert (
        remappings[(DroneMapToGo2ObstacleLayer, "drone_obstacle_layer")]  # type: ignore[index]
        == "go2/drone_obstacle_layer"
    )
    assert remappings[(Go2CostmapFusion, "global_costmap")] == "go2/global_costmap"
    assert remappings[(Go2CostmapFusion, "drone_obstacle_layer")] == "go2/drone_obstacle_layer"
    assert remappings[(Go2CostmapFusion, "fused_global_costmap")] == "go2/fused_global_costmap"
    assert remappings[(Go2FusionPlanner, "global_costmap")] == "go2/fused_global_costmap"
    assert remappings[(Go2FusionPlanner, "goal_request")] == "go2/goal_request"
    assert remappings[(Go2FusionPlanner, "recovery_goal_request")] == "go2/recovery_goal_request"
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


def test_drone_go2_lidar_fusion_agentic_mcp_scopes_modules_and_streams() -> None:
    modules = {bp.module for bp in drone_go2_lidar_fusion_sim_agentic_mcp.blueprints}
    remappings = drone_go2_lidar_fusion_sim_agentic_mcp.remapping_map

    assert McpServer in modules
    assert McpClient in modules
    assert DroneGo2FusionSkillContainer in modules
    assert Go2FusionPlanner in modules
    assert DroneFusionPlanner in modules
    assert remappings[(DroneGo2FusionSkillContainer, "drone_odom")] == "drone/odom"
    assert remappings[(DroneGo2FusionSkillContainer, "go2_odom")] == "go2/odom"
    assert remappings[(DroneGo2FusionSkillContainer, "drone_path")] == "drone/path"
    assert (
        remappings[(DroneGo2FusionSkillContainer, "drone_global_costmap")] == "drone/global_costmap"
    )
    assert (
        remappings[(DroneGo2FusionSkillContainer, "go2_fused_global_costmap")]
        == "go2/fused_global_costmap"
    )
    assert remappings[(DroneGo2FusionSkillContainer, "go2_goal_observed")] == "go2/goal_request"
    assert (
        remappings[(DroneGo2FusionSkillContainer, "go2_recovery_goal_request")]
        == "go2/recovery_goal_request"
    )
    assert remappings[(DroneGo2FusionSkillContainer, "drone_goal_request")] == "drone/goal_request"
    assert remappings[(DroneGo2FusionSkillContainer, "go2_goal_request")] == "go2/goal_request"


def test_global_planner_emits_recovery_goal_when_replans_are_exhausted(
    monkeypatch,
) -> None:
    planner = GlobalPlanner(GlobalConfig())
    published: list[PoseStamped] = []
    cancelled: list[bool] = []
    planner.recovery_goal.subscribe(published.append)
    planner._current_goal = _pose(2.0, 3.0, ts=101.0)
    planner._current_odom = _pose(0.0, 0.0, ts=102.0)

    monkeypatch.setattr(planner._replan_limiter, "can_retry", lambda _pos: False)
    monkeypatch.setattr(planner, "cancel_goal", lambda **_kwargs: cancelled.append(True))

    planner._replan_path()

    assert [(goal.position.x, goal.position.y) for goal in published] == [(2.0, 3.0)]
    assert cancelled == [True]


def test_global_planner_emits_recovery_goal_when_no_path_is_found(
    monkeypatch,
) -> None:
    planner = GlobalPlanner(GlobalConfig())
    published: list[PoseStamped] = []
    cancelled: list[dict[str, bool]] = []
    planner.recovery_goal.subscribe(published.append)
    planner._current_goal = _pose(2.0, 3.0, ts=101.0)
    planner._current_odom = _pose(0.0, 0.0, ts=102.0)

    monkeypatch.setattr(planner, "cancel_goal", lambda **kwargs: cancelled.append(kwargs))
    monkeypatch.setattr(planner, "_find_safe_goal", lambda _goal: _goal)
    monkeypatch.setattr(planner, "_find_wide_path", lambda _goal, _pos: None)

    planner._plan_path()

    assert [(goal.position.x, goal.position.y) for goal in published] == [(2.0, 3.0)]
    assert cancelled == [{"but_will_try_again": True}, {}]


def test_global_planner_emits_recovery_goal_after_three_replans(
    monkeypatch,
) -> None:
    planner = GlobalPlanner(GlobalConfig())
    published: list[PoseStamped] = []
    cancelled: list[bool] = []
    planner.recovery_goal.subscribe(published.append)
    planner._current_goal = _pose(2.0, 3.0, ts=101.0)
    planner._current_odom = _pose(0.0, 0.0, ts=102.0)

    monkeypatch.setattr(planner._replan_limiter, "can_retry", lambda _pos: True)
    monkeypatch.setattr(planner._replan_limiter, "get_attempt", lambda: 3)
    monkeypatch.setattr(planner, "cancel_goal", lambda **_kwargs: cancelled.append(True))

    planner._replan_path()

    assert [(goal.position.x, goal.position.y) for goal in published] == [(2.0, 3.0)]
    assert cancelled == [True]


def test_drone_go2_fusion_skills_publish_robot_scoped_goals() -> None:
    skills = DroneGo2FusionSkillContainer()
    drone_goals: list[PoseStamped] = []
    go2_goals: list[PoseStamped] = []
    skills.drone_goal_request.subscribe(drone_goals.append)
    skills.go2_goal_request.subscribe(go2_goals.append)

    try:
        drone_message = skills.drone_go_to(1.0, 2.0)
        go2_message = skills.go2_go_to(3.0, 4.0)
    finally:
        skills.stop()

    assert "Drone goal published" in drone_message
    assert "Go2 goal published" in go2_message
    assert [(goal.position.x, goal.position.y) for goal in drone_goals] == [(1.0, 2.0)]
    assert [(goal.position.x, goal.position.y) for goal in go2_goals] == [(3.0, 4.0)]


def test_drone_go2_fusion_scout_then_go2_publishes_in_order() -> None:
    skills = DroneGo2FusionSkillContainer()
    published: list[tuple[str, PoseStamped]] = []
    skills.drone_goal_request.subscribe(lambda msg: published.append(("drone", msg)))
    skills.go2_goal_request.subscribe(lambda msg: published.append(("go2", msg)))
    skills._on_drone_odom(_pose(0.0, 0.0, 1.2))
    skills._on_drone_costmap(
        _grid(
            np.full((30, 30), int(CostValues.UNKNOWN), dtype=np.int8),
            resolution=0.5,
            origin_x=-5.0,
            origin_y=-5.0,
        )
    )
    timer = _publish_drone_path_after(
        skills,
        0.02,
        [_pose(0.0, 0.0, 1.2), _pose(5.0, 6.0, 1.2)],
    )

    try:
        message = skills.fusion_scout_then_go2(5.0, 6.0, scout_seconds=0.3)
    finally:
        timer.cancel()
        timer.join(timeout=1.0)
        skills.stop()

    assert "Drone scout completed" in message
    assert "Go2 goal published" in message
    assert [(robot, goal.position.x, goal.position.y) for robot, goal in published] == [
        ("drone", 5.0, 6.0),
        ("go2", 5.0, 6.0),
    ]


def test_drone_go2_fusion_zone_skills_resolve_known_zone_names() -> None:
    skills = DroneGo2FusionSkillContainer()
    published: list[tuple[str, PoseStamped]] = []
    skills.drone_goal_request.subscribe(lambda msg: published.append(("drone", msg)))
    skills.go2_goal_request.subscribe(lambda msg: published.append(("go2", msg)))

    try:
        drone_message = skills.drone_go_to_zone("Zone C")
        go2_message = skills.go2_go_to_zone("zone c")
    finally:
        skills.stop()

    assert "Drone goal published" in drone_message
    assert "Go2 goal published" in go2_message
    assert [
        (robot, round(goal.position.x, 2), round(goal.position.y, 2)) for robot, goal in published
    ] == [
        ("drone", 0.08, -5.98),
        ("go2", 0.08, -5.98),
    ]


def test_drone_go2_fusion_scout_zone_publishes_go2_after_drone_path() -> None:
    skills = DroneGo2FusionSkillContainer()
    published: list[tuple[str, PoseStamped]] = []
    skills.drone_goal_request.subscribe(lambda msg: published.append(("drone", msg)))
    skills.go2_goal_request.subscribe(lambda msg: published.append(("go2", msg)))
    skills._on_drone_odom(_pose(0.0, -4.0, 1.2))
    skills._on_drone_costmap(
        _grid(
            np.full((20, 20), int(CostValues.UNKNOWN), dtype=np.int8),
            resolution=0.5,
            origin_x=-2.0,
            origin_y=-8.0,
        )
    )
    timer = _publish_drone_path_after(
        skills,
        0.02,
        [_pose(0.0, -4.0, 1.2), _pose(0.08, -5.98, 1.2)],
    )

    try:
        message = skills.fusion_scout_zone_then_go2("Zone C", scout_seconds=0.3)
    finally:
        timer.cancel()
        timer.join(timeout=1.0)
        skills.stop()

    assert "Drone scout completed for Zone C" in message
    assert "Go2 goal published" in message
    assert [
        (robot, round(goal.position.x, 2), round(goal.position.y, 2)) for robot, goal in published
    ] == [
        ("drone", 0.08, -5.98),
        ("go2", 0.08, -5.98),
    ]


def test_drone_go2_fusion_scout_zone_uses_intermediate_waypoint_outside_bounds() -> None:
    skills = DroneGo2FusionSkillContainer()
    drone_goals: list[PoseStamped] = []
    go2_goals: list[PoseStamped] = []
    skills.drone_goal_request.subscribe(drone_goals.append)
    skills.go2_goal_request.subscribe(go2_goals.append)
    skills._on_drone_odom(_pose(-1.0, 1.0, 1.2))
    skills._on_drone_costmap(
        _grid(
            np.full((78, 75), int(CostValues.UNKNOWN), dtype=np.int8),
            resolution=0.1,
            origin_x=-4.75,
            origin_y=-2.95,
        )
    )

    try:
        message = skills.fusion_scout_zone_then_go2("Zone C", scout_seconds=0.1)
    finally:
        skills.stop()

    assert "Drone scout failed for Zone C" in message
    assert "Go2 goal not published" in message
    assert "drone planner did not produce a path" in message
    assert len(drone_goals) == 1
    assert go2_goals == []
    drone_goal = drone_goals[0]
    assert -4.75 < drone_goal.position.x < 2.75
    assert -2.95 < drone_goal.position.y < 4.85
    assert round(drone_goal.position.x, 2) != 0.08
    assert round(drone_goal.position.y, 2) != -5.98
    assert drone_goal.position.y < 1.0


def test_drone_go2_fusion_scout_zone_continues_past_short_timeout_when_moving() -> None:
    skills = DroneGo2FusionSkillContainer()
    drone_goals: list[PoseStamped] = []
    go2_goals: list[PoseStamped] = []
    timers: list[threading.Timer] = []
    skills._on_drone_odom(_pose(-1.0, 1.0, 1.2))
    skills._on_drone_costmap(
        _grid(
            np.full((78, 75), int(CostValues.UNKNOWN), dtype=np.int8),
            resolution=0.1,
            origin_x=-4.75,
            origin_y=-2.95,
        )
    )

    def on_drone_goal(goal: PoseStamped) -> None:
        drone_goals.append(goal)
        goal_pose = _pose(goal.position.x, goal.position.y, 1.2)
        timers.append(
            _publish_drone_path_after(
                skills,
                0.02,
                [skills._latest_drone_odom or _pose(-1.0, 1.0, 1.2), goal_pose],
            )
        )
        timers.append(_run_after(0.20, lambda: skills._on_drone_odom(goal_pose)))
        if len(drone_goals) == 1:
            timers.append(
                _run_after(
                    0.22,
                    lambda: skills._on_drone_costmap(
                        _grid(
                            np.full((78, 75), int(CostValues.UNKNOWN), dtype=np.int8),
                            resolution=0.1,
                            origin_x=-4.75,
                            origin_y=-4.8,
                        )
                    ),
                )
            )
        else:
            timers.append(
                _run_after(
                    0.22,
                    lambda: skills._on_drone_costmap(
                        _grid(
                            np.full((120, 75), int(CostValues.UNKNOWN), dtype=np.int8),
                            resolution=0.1,
                            origin_x=-4.75,
                            origin_y=-7.0,
                        )
                    ),
                )
            )

    skills.drone_goal_request.subscribe(on_drone_goal)
    skills.go2_goal_request.subscribe(go2_goals.append)

    try:
        message = skills.fusion_scout_zone_then_go2("Zone C", scout_seconds=0.1)
    finally:
        for timer in timers:
            timer.cancel()
        for timer in timers:
            timer.join(timeout=1.0)
        skills.stop()

    assert "Drone scout completed for Zone C" in message
    assert "Go2 goal published" in message
    assert len(drone_goals) >= 2
    assert len(go2_goals) == 1
    assert round(go2_goals[0].position.x, 2) == 0.08
    assert round(go2_goals[0].position.y, 2) == -5.98


def test_drone_go2_fusion_reveal_path_then_go2_publishes_immediately_when_path_exists() -> None:
    skills = DroneGo2FusionSkillContainer()
    drone_goals: list[PoseStamped] = []
    go2_goals: list[PoseStamped] = []
    skills.drone_goal_request.subscribe(drone_goals.append)
    skills.go2_goal_request.subscribe(go2_goals.append)
    skills._on_go2_odom(_pose(0.0, 0.0, 0.0))
    skills._on_go2_fused_costmap(
        _grid(
            np.full((5, 5), int(CostValues.FREE), dtype=np.int8),
            resolution=1.0,
            origin_x=-2.0,
            origin_y=-2.0,
        )
    )

    try:
        message = skills.fusion_reveal_path_then_go2(2.0, 0.0, scout_seconds=0.1)
    finally:
        skills.stop()

    assert "Drone path reveal completed" in message
    assert "Go2 goal published" in message
    assert drone_goals == []
    assert [(goal.position.x, goal.position.y) for goal in go2_goals] == [(2.0, 0.0)]


def test_drone_go2_fusion_reveal_path_zone_then_go2_resolves_zone() -> None:
    skills = DroneGo2FusionSkillContainer()
    drone_goals: list[PoseStamped] = []
    go2_goals: list[PoseStamped] = []
    skills.drone_goal_request.subscribe(drone_goals.append)
    skills.go2_goal_request.subscribe(go2_goals.append)
    skills._on_go2_odom(_pose(0.0, -4.0, 0.0))
    skills._on_go2_fused_costmap(
        _grid(
            np.full((20, 20), int(CostValues.FREE), dtype=np.int8),
            resolution=0.5,
            origin_x=-2.0,
            origin_y=-8.0,
        )
    )

    try:
        message = skills.fusion_reveal_path_zone_then_go2("Zone C", scout_seconds=0.1)
    finally:
        skills.stop()

    assert "Drone path reveal completed for Zone C" in message
    assert "Go2 goal published" in message
    assert drone_goals == []
    assert len(go2_goals) == 1
    assert round(go2_goals[0].position.x, 2) == 0.08
    assert round(go2_goals[0].position.y, 2) == -5.98


def test_drone_go2_fusion_reveal_path_then_go2_uses_current_astar_behavior() -> None:
    skills = DroneGo2FusionSkillContainer()
    published: list[tuple[str, PoseStamped]] = []
    timers: list[threading.Timer] = []
    skills.drone_goal_request.subscribe(lambda msg: published.append(("drone", msg)))
    skills.go2_goal_request.subscribe(lambda msg: published.append(("go2", msg)))
    skills._on_go2_odom(_pose(0.0, 0.0, 0.0))
    skills._on_drone_odom(_pose(0.0, 0.0, 1.2))
    skills._on_drone_costmap(
        _grid(
            np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8),
            resolution=1.0,
            origin_x=-1.0,
            origin_y=-2.0,
        )
    )
    initial_fused = np.full((5, 5), int(CostValues.UNKNOWN), dtype=np.int8)
    initial_fused[2, 1] = int(CostValues.FREE)
    initial_fused[2, 3] = int(CostValues.FREE)
    skills._on_go2_fused_costmap(_grid(initial_fused, resolution=1.0, origin_x=-1.0, origin_y=-2.0))

    def on_drone_goal(goal: PoseStamped) -> None:
        goal_pose = _pose(goal.position.x, goal.position.y, 1.2)
        timers.append(
            _publish_drone_path_after(
                skills,
                0.02,
                [skills._latest_drone_odom or _pose(0.0, 0.0, 1.2), goal_pose],
            )
        )
        timers.append(_run_after(0.04, lambda: skills._on_drone_odom(goal_pose)))
        updated_fused = initial_fused.copy()
        updated_fused[2, 2] = int(CostValues.FREE)
        timers.append(
            _run_after(
                0.05,
                lambda: skills._on_go2_fused_costmap(
                    _grid(updated_fused, resolution=1.0, origin_x=-1.0, origin_y=-2.0)
                ),
            )
        )

    skills.drone_goal_request.subscribe(on_drone_goal)

    try:
        message = skills.fusion_reveal_path_then_go2(2.0, 0.0, scout_seconds=0.1)
    finally:
        for timer in timers:
            timer.cancel()
        for timer in timers:
            timer.join(timeout=1.0)
        skills.stop()

    assert "Drone path reveal completed" in message
    assert "Go2 goal published" in message
    assert [robot for robot, _ in published] == ["go2"]
    assert round(published[0][1].position.x, 2) == 2.0
    assert round(published[0][1].position.y, 2) == 0.0


def test_drone_go2_fusion_known_zones_reports_zone_coordinates() -> None:
    skills = DroneGo2FusionSkillContainer()
    try:
        payload = json.loads(skills.fusion_known_zones())
    finally:
        skills.stop()

    zones = {zone["name"]: zone for zone in payload["zones"]}
    assert zones["Zone C"]["x"] == 0.08
    assert zones["Zone C"]["y"] == -5.98


def test_drone_go2_fusion_map_status_reports_cached_state() -> None:
    skills = DroneGo2FusionSkillContainer()
    try:
        skills._on_drone_odom(_pose(1.0, 2.0, 1.2, ts=101.0))
        skills._on_go2_odom(_pose(3.0, 4.0, 0.0, ts=102.0))
        skills._on_drone_costmap(
            _grid(np.full((2, 3), int(CostValues.FREE), dtype=np.int8), ts=103.0)
        )
        skills._on_go2_fused_costmap(
            _grid(np.full((4, 5), int(CostValues.UNKNOWN), dtype=np.int8), ts=104.0)
        )
        payload = json.loads(skills.fusion_map_status())
    finally:
        skills.stop()

    assert payload["drone_pose"]["x"] == 1.0
    assert payload["go2_pose"]["x"] == 3.0
    assert payload["drone_global_costmap"]["width"] == 3
    assert payload["go2_fused_global_costmap"]["height"] == 4


def test_drone_go2_fusion_auto_recovery_republishes_goal_after_path_reveal(
    monkeypatch,
) -> None:
    skills = DroneGo2FusionSkillContainer()
    published: list[PoseStamped] = []
    goal = _pose(5.0, 6.0, ts=201.0)
    calls: list[tuple[float, float, float, float, float]] = []
    skills.go2_goal_request.subscribe(published.append)
    skills._on_go2_odom(_pose(1.0, 2.0, ts=200.0))
    skills._on_go2_goal_observed(goal)

    def fake_reveal(
        start_pose: PoseStamped,
        x: float,
        y: float,
        *,
        timeout_s: float,
        abort_event=None,
    ) -> _ScoutResult:
        calls.append((start_pose.position.x, start_pose.position.y, x, y, timeout_s))
        return _ScoutResult(True, "go2_astar_path_found", 0.1)

    monkeypatch.setattr(skills, "_reveal_path_to_goal", fake_reveal)

    try:
        skills._on_go2_recovery_goal_request(goal)
        deadline = time.monotonic() + 1.0
        while len(published) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        skills.stop()

    assert calls == [(1.0, 2.0, 5.0, 6.0, 60.0)]
    assert [(msg.position.x, msg.position.y) for msg in published] == [(5.0, 6.0)]


def test_drone_go2_fusion_auto_recovery_ignores_duplicate_triggers_while_active(
    monkeypatch,
) -> None:
    skills = DroneGo2FusionSkillContainer()
    goal = _pose(5.0, 6.0, ts=201.0)
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    skills._on_go2_odom(_pose(1.0, 2.0, ts=200.0))
    skills._on_go2_goal_observed(goal)

    def fake_reveal(
        start_pose: PoseStamped,
        x: float,
        y: float,
        *,
        timeout_s: float,
        abort_event=None,
    ) -> _ScoutResult:
        calls.append(1)
        started.set()
        release.wait(timeout=0.5)
        return _ScoutResult(False, "timed_out", 0.1)

    monkeypatch.setattr(skills, "_reveal_path_to_goal", fake_reveal)

    try:
        skills._on_go2_recovery_goal_request(goal)
        assert started.wait(timeout=0.5)
        skills._on_go2_recovery_goal_request(goal)
        release.set()
        deadline = time.monotonic() + 1.0
        while skills._auto_recovery_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        skills.stop()

    assert calls == [1]


def test_drone_go2_fusion_auto_recovery_cancels_when_new_goal_is_observed(
    monkeypatch,
) -> None:
    skills = DroneGo2FusionSkillContainer()
    original_goal = _pose(5.0, 6.0, ts=201.0)
    newer_goal = _pose(7.0, 8.0, ts=202.0)
    started = threading.Event()
    republished: list[PoseStamped] = []
    skills.go2_goal_request.subscribe(republished.append)
    skills._on_go2_odom(_pose(1.0, 2.0, ts=200.0))
    skills._on_go2_goal_observed(original_goal)

    def fake_reveal(
        start_pose: PoseStamped,
        x: float,
        y: float,
        *,
        timeout_s: float,
        abort_event=None,
    ) -> _ScoutResult:
        started.set()
        deadline = time.monotonic() + 0.5
        while abort_event is not None and not abort_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        return _ScoutResult(False, "path-reveal recovery cancelled", 0.1)

    monkeypatch.setattr(skills, "_reveal_path_to_goal", fake_reveal)

    try:
        skills._on_go2_recovery_goal_request(original_goal)
        assert started.wait(timeout=0.5)
        skills._on_go2_goal_observed(newer_goal)
        deadline = time.monotonic() + 1.0
        while skills._auto_recovery_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        skills.stop()

    assert republished == []
