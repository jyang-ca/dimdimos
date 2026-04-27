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

"""Agentic MCP blueprint for the Go2 + simulated drone LiDAR fusion stack."""

from dimos.agents.mcp.mcp_client import mcp_client
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.sim.drone_go2_lidar_fusion_sim import (
    drone_go2_lidar_fusion_sim,
)
from dimos.robot.unitree.go2.drone_go2_fusion_skills import (
    DroneGo2FusionSkillContainer,
    drone_go2_fusion_skills,
)

DRONE_GO2_FUSION_SYSTEM_PROMPT = """You control a simulated drone and a simulated Unitree Go2 in a shared LiDAR fusion scene.

Use robot-scoped tools precisely:
- Use drone_go_to only for drone movement, scouting, inspection, or mapping.
- Use go2_go_to only for Go2 ground navigation using the fused Go2 costmap.
- Use fusion_scout_then_go2 when the route may be unknown and the drone should map first.
- Use fusion_reveal_path_then_go2 when the goal is to reveal any traversable Go2 route from the Go2's current pose to a target, without trying to optimize the whole route globally.
- Use drone_go_to_zone, go2_go_to_zone, or fusion_scout_zone_then_go2 when the user names Zone A, Zone B, or Zone C. Do not ask the user for coordinates for known zones.
- Use fusion_reveal_path_zone_then_go2 when the user names a zone and wants the drone to uncover any Go2 A* path to that zone before the Go2 moves.
- Use fusion_known_zones when you need the known zone coordinates.
- Use fusion_map_status to inspect map and pose state before or after navigation.

The drone is an aerial mapper. The Go2 is a ground robot. Drone observations update the Go2 fused costmap by adding drone-confirmed free space and obstacles. If the user does not specify a robot and the task involves discovering a better or unknown route, scout with the drone before moving the Go2.
Known zones: Zone A=(-0.35, 7.99), Zone B=(1.30, 0.89), Zone C=(0.08, -5.98).
"""

drone_go2_lidar_fusion_sim_agentic_mcp = autoconnect(
    drone_go2_lidar_fusion_sim,
    McpServer.blueprint(),
    mcp_client(system_prompt=DRONE_GO2_FUSION_SYSTEM_PROMPT),
    drone_go2_fusion_skills(),
).remappings(
    [
        (DroneGo2FusionSkillContainer, "drone_odom", "drone/odom"),
        (DroneGo2FusionSkillContainer, "go2_odom", "go2/odom"),
        (DroneGo2FusionSkillContainer, "drone_path", "drone/path"),
        (DroneGo2FusionSkillContainer, "drone_global_costmap", "drone/global_costmap"),
        (
            DroneGo2FusionSkillContainer,
            "go2_fused_global_costmap",
            "go2/fused_global_costmap",
        ),
        (DroneGo2FusionSkillContainer, "go2_goal_observed", "go2/goal_request"),
        (
            DroneGo2FusionSkillContainer,
            "go2_recovery_goal_request",
            "go2/recovery_goal_request",
        ),
        (DroneGo2FusionSkillContainer, "drone_goal_request", "drone/goal_request"),
        (DroneGo2FusionSkillContainer, "go2_goal_request", "go2/goal_request"),
    ]
)

__all__ = [
    "DRONE_GO2_FUSION_SYSTEM_PROMPT",
    "drone_go2_lidar_fusion_sim_agentic_mcp",
]
