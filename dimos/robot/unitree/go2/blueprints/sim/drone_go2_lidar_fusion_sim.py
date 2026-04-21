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

"""Go2 + simulated drone LiDAR fusion blueprint."""

from typing import TYPE_CHECKING, Any, cast

from dimos.core.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.drone.flight_costmapper import DroneFlightCostMapper, drone_flight_cost_mapper
from dimos.robot.drone.sim_connection_module import DroneSimConnectionModule, drone_sim_connection
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.drone_fusion_costmap import (
    DroneMapToGo2ObstacleLayer,
    Go2CostmapFusion,
    drone_map_to_go2_obstacle_layer,
    go2_costmap_fusion,
)
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

if TYPE_CHECKING:
    from dimos.core.module import Module

_VOXEL_SIZE = 0.1

_GO2_LIDAR = "go2/lidar"
_GO2_ODOM = "go2/odom"
_GO2_CMD_VEL = "go2/cmd_vel"
_GO2_GOAL_REQUEST = "go2/goal_request"
_GO2_GLOBAL_MAP = "go2/global_map"
_GO2_GLOBAL_COSTMAP = "go2/global_costmap"
_GO2_DRONE_OBSTACLE_LAYER = "go2/drone_obstacle_layer"
_GO2_FUSED_GLOBAL_COSTMAP = "go2/fused_global_costmap"
_GO2_PATH = "go2/path"
_GO2_NAVIGATION_COSTMAP = "go2/navigation_costmap"

_DRONE_LIDAR = "drone/lidar"
_DRONE_ODOM = "drone/odom"
_DRONE_CMD_VEL = "drone/cmd_vel"
_DRONE_GOAL_REQUEST = "drone/goal_request"
_DRONE_MOVECMD_TWIST = "drone/movecmd_twist"
_DRONE_GLOBAL_MAP = "drone/global_map"
_DRONE_GLOBAL_COSTMAP = "drone/global_costmap"
_DRONE_PATH = "drone/path"
_DRONE_NAVIGATION_COSTMAP = "drone/navigation_costmap"

_DRONE_SIM_MODULE = cast("type[Module[Any]]", DroneSimConnectionModule)
_DRONE_FLIGHT_COST_MODULE = cast("type[Module[Any]]", DroneFlightCostMapper)
_DRONE_TO_GO2_LAYER_MODULE = cast("type[Module[Any]]", DroneMapToGo2ObstacleLayer)


class Go2FusionVoxelGridMapper(VoxelGridMapper):
    """Voxel mapper instance for the Go2 side of the fusion blueprint."""


class Go2FusionCostMapper(CostMapper):
    """Cost mapper instance that produces the unfused Go2 costmap."""


class Go2FusionPlanner(ReplanningAStarPlanner):
    """Planner instance that consumes the fused Go2 costmap."""


class DroneFusionPlanner(ReplanningAStarPlanner):
    """Planner instance that consumes the drone flight costmap."""


class DroneFusionVoxelGridMapper(VoxelGridMapper):
    """Voxel mapper instance for the drone side of the fusion blueprint."""


drone_go2_lidar_fusion_sim = (
    autoconnect(
        unitree_go2_basic,
        Go2FusionVoxelGridMapper.blueprint(voxel_size=_VOXEL_SIZE),
        Go2FusionCostMapper.blueprint(),
        drone_sim_connection(),
        DroneFusionVoxelGridMapper.blueprint(
            publish_interval=1.0,
            voxel_size=_VOXEL_SIZE,
            carve_columns=False,
        ),
        drone_flight_cost_mapper(
            resolution=0.1,
            clearance_below=0.25,
            clearance_above=0.25,
            high_obstacle_policy="ignore",
        ),
        drone_map_to_go2_obstacle_layer(
            resolution=0.1,
            min_obstacle_height=0.12,
            max_obstacle_height=1.6,
        ),
        go2_costmap_fusion(),
        Go2FusionPlanner.blueprint(),
        DroneFusionPlanner.blueprint(),
    )
    .remappings(
        [
            (GO2Connection, "lidar", _GO2_LIDAR),
            (GO2Connection, "odom", _GO2_ODOM),
            (GO2Connection, "cmd_vel", _GO2_CMD_VEL),
            (Go2FusionVoxelGridMapper, "lidar", _GO2_LIDAR),
            (Go2FusionVoxelGridMapper, "global_map", _GO2_GLOBAL_MAP),
            (Go2FusionCostMapper, "global_map", _GO2_GLOBAL_MAP),
            (Go2FusionCostMapper, "global_costmap", _GO2_GLOBAL_COSTMAP),
            (Go2FusionPlanner, "odom", _GO2_ODOM),
            (Go2FusionPlanner, "global_costmap", _GO2_FUSED_GLOBAL_COSTMAP),
            (Go2FusionPlanner, "goal_request", _GO2_GOAL_REQUEST),
            (Go2FusionPlanner, "cmd_vel", _GO2_CMD_VEL),
            (Go2FusionPlanner, "path", _GO2_PATH),
            (Go2FusionPlanner, "navigation_costmap", _GO2_NAVIGATION_COSTMAP),
            (WebsocketVisModule, "odom", _GO2_ODOM),
            (WebsocketVisModule, "global_costmap", _GO2_FUSED_GLOBAL_COSTMAP),
            (WebsocketVisModule, "path", _GO2_PATH),
            (WebsocketVisModule, "goal_request", _GO2_GOAL_REQUEST),
            (WebsocketVisModule, "cmd_vel", _GO2_CMD_VEL),
            (WebsocketVisModule, "go2_odom", _GO2_ODOM),
            (WebsocketVisModule, "go2_global_costmap", _GO2_FUSED_GLOBAL_COSTMAP),
            (WebsocketVisModule, "go2_path", _GO2_PATH),
            (WebsocketVisModule, "go2_goal_request", _GO2_GOAL_REQUEST),
            (WebsocketVisModule, "drone_odom", _DRONE_ODOM),
            (WebsocketVisModule, "drone_global_costmap", _DRONE_GLOBAL_COSTMAP),
            (WebsocketVisModule, "drone_path", _DRONE_PATH),
            (WebsocketVisModule, "drone_goal_request", _DRONE_GOAL_REQUEST),
            (_DRONE_SIM_MODULE, "lidar", _DRONE_LIDAR),
            (_DRONE_SIM_MODULE, "odom", _DRONE_ODOM),
            (_DRONE_SIM_MODULE, "cmd_vel", _DRONE_CMD_VEL),
            (_DRONE_SIM_MODULE, "movecmd_twist", _DRONE_MOVECMD_TWIST),
            (DroneFusionVoxelGridMapper, "lidar", _DRONE_LIDAR),
            (DroneFusionVoxelGridMapper, "global_map", _DRONE_GLOBAL_MAP),
            (_DRONE_FLIGHT_COST_MODULE, "global_map", _DRONE_GLOBAL_MAP),
            (_DRONE_FLIGHT_COST_MODULE, "odom", _DRONE_ODOM),
            (_DRONE_FLIGHT_COST_MODULE, "global_costmap", _DRONE_GLOBAL_COSTMAP),
            (DroneFusionPlanner, "odom", _DRONE_ODOM),
            (DroneFusionPlanner, "global_costmap", _DRONE_GLOBAL_COSTMAP),
            (DroneFusionPlanner, "goal_request", _DRONE_GOAL_REQUEST),
            (DroneFusionPlanner, "cmd_vel", _DRONE_CMD_VEL),
            (DroneFusionPlanner, "path", _DRONE_PATH),
            (DroneFusionPlanner, "navigation_costmap", _DRONE_NAVIGATION_COSTMAP),
            (_DRONE_TO_GO2_LAYER_MODULE, "global_map", _DRONE_GLOBAL_MAP),
            (
                _DRONE_TO_GO2_LAYER_MODULE,
                "drone_obstacle_layer",
                _GO2_DRONE_OBSTACLE_LAYER,
            ),
            (Go2CostmapFusion, "global_costmap", _GO2_GLOBAL_COSTMAP),
            (Go2CostmapFusion, "drone_obstacle_layer", _GO2_DRONE_OBSTACLE_LAYER),
            (Go2CostmapFusion, "fused_global_costmap", _GO2_FUSED_GLOBAL_COSTMAP),
        ]
    )
    .global_config(n_workers=10, robot_model="unitree_go2", simulation=True)
)

__all__ = [
    "DroneFusionPlanner",
    "DroneFusionVoxelGridMapper",
    "Go2FusionCostMapper",
    "Go2FusionPlanner",
    "Go2FusionVoxelGridMapper",
    "drone_go2_lidar_fusion_sim",
]
