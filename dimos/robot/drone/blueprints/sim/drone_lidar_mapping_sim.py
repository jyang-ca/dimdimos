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

"""Simulated drone LiDAR mapping blueprint."""

from typing import Any

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.health import mapping_health_monitor
from dimos.mapping.pointclouds.occupancy import SimpleOccupancyConfig
from dimos.mapping.voxels import voxel_mapper
from dimos.navigation.replanning_a_star.module import replanning_a_star_planner
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.robot.drone.sim_connection_module import drone_sim_connection
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

_VOXEL_SIZE = 0.1


def _convert_global_map(grid: Any) -> Any:
    return grid.to_rerun(voxel_size=_VOXEL_SIZE, mode="boxes")


def _convert_global_costmap(grid: Any) -> Any:
    return grid.to_rerun(
        colormap="Accent",
        z_offset=0.015,
        opacity=0.35,
        background="#484981",
    )


def _drone_sim_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Drone Sim 3D"),
        ),
        rrb.TimePanel(timeline="dimos_time", play_state="following"),
    )


_rerun_config = {
    "blueprint": _drone_sim_rerun_blueprint,
    "pubsubs": [LCM()],
    "min_interval_by_entity": {
        "world/odom": 0.5,
        "world/lidar": 0.5,
        "world/global_map": 1.0,
        "world/global_costmap": 1.0,
        "world/tf": 0.5,
    },
    "visual_override": {
        "world/global_map": _convert_global_map,
        "world/global_costmap": _convert_global_costmap,
    },
}

if global_config.viewer.startswith("rerun"):
    from dimos.visualization.rerun.bridge import _resolve_viewer_mode, rerun_bridge

    _vis = rerun_bridge(viewer_mode=_resolve_viewer_mode(), **_rerun_config)
else:
    _vis = autoconnect()

drone_lidar_mapping_sim = autoconnect(
    _vis,
    drone_sim_connection(),
    voxel_mapper(publish_interval=1.0, voxel_size=_VOXEL_SIZE, carve_columns=False),
    cost_mapper(
        algo="simple",
        config=SimpleOccupancyConfig(resolution=0.1, min_height=0.15, max_height=2.5),
    ),
    replanning_a_star_planner(),
    mapping_health_monitor(),
    websocket_vis(),
).global_config(n_workers=7, robot_model="drone_lidar_mapping_sim", simulation=True)

__all__ = [
    "drone_lidar_mapping_sim",
]
