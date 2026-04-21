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

"""Drone blueprint with MAVLink camera/control plus a FAST-LIO2 LiDAR stack."""

from dimos.core.blueprints import autoconnect
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.robot.drone.blueprints.basic.drone_basic import drone_basic

_LIDAR_VOXEL_SIZE = 0.05

drone_lidar_basic = autoconnect(
    drone_basic,
    FastLio2.blueprint(
        frame_id="world",
        child_frame_id="drone/base_link",
        voxel_size=_LIDAR_VOXEL_SIZE,
        map_voxel_size=_LIDAR_VOXEL_SIZE,
        map_freq=0.0,
    ),
).global_config(n_workers=5, robot_model="drone_lidar")

__all__ = [
    "drone_lidar_basic",
]
