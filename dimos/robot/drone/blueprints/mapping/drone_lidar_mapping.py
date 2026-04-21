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

"""Drone LiDAR mapping blueprint with global map/costmap health metrics."""

from typing import TYPE_CHECKING, Any, cast

from dimos.core.blueprints import autoconnect
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.health import mapping_health_monitor
from dimos.mapping.voxels import voxel_mapper
from dimos.robot.drone.blueprints.basic.drone_lidar_basic import drone_lidar_basic

if TYPE_CHECKING:
    from dimos.core.module import Module

_VOXEL_SIZE = 0.05
_FASTLIO2_MODULE = cast("type[Module[Any]]", FastLio2)

drone_lidar_mapping = (
    autoconnect(
        drone_lidar_basic,
        voxel_mapper(publish_interval=1.0, voxel_size=_VOXEL_SIZE, carve_columns=False),
        cost_mapper(),
        mapping_health_monitor(),
    )
    .remappings(
        [
            # Keep the Python voxel mapper as the sole publisher for the public
            # global_map stream. FAST-LIO2 still publishes registered scan frames
            # on lidar; its native map output is disabled in drone_lidar_basic.
            (_FASTLIO2_MODULE, "global_map", "fastlio_global_map"),
        ]
    )
    .global_config(n_workers=7, robot_model="drone_lidar_mapping")
)

__all__ = [
    "drone_lidar_mapping",
]
