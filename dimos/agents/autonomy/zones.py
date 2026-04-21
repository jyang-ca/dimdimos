# Copyright 2026 Dimensional Inc.
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

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneDefinition:
    """Known physical or simulated zone that can be inspected by the robot."""

    name: str
    center_x: float
    center_y: float
    center_z: float = 0.0
    frame_id: str = "map"
    default_radius_m: float = 1.0
    default_views: int = 8


DEFAULT_ZONE_DEFINITIONS: tuple[ZoneDefinition, ...] = (
    # Default MuJoCo scene_office1 map/world coordinates.
    # Zone A: meeting-table area; Zone B: central white-table area;
    # Zone C: south workstation area.
    ZoneDefinition(
        name="Zone A",
        center_x=-0.35,
        center_y=7.99,
        default_radius_m=2.4,
    ),
    ZoneDefinition(
        name="Zone B",
        center_x=1.30,
        center_y=0.89,
        default_radius_m=3.4,
    ),
    ZoneDefinition(
        name="Zone C",
        center_x=0.08,
        center_y=-5.98,
        default_radius_m=2.6,
    ),
)


def normalize_zone_name(name: str) -> str:
    return " ".join(name.casefold().split())


def find_zone(
    zone_name: str,
    zones: tuple[ZoneDefinition, ...] = DEFAULT_ZONE_DEFINITIONS,
) -> ZoneDefinition | None:
    target = normalize_zone_name(zone_name)
    for zone in zones:
        if normalize_zone_name(zone.name) == target:
            return zone
    return None


__all__ = [
    "DEFAULT_ZONE_DEFINITIONS",
    "ZoneDefinition",
    "find_zone",
    "normalize_zone_name",
]
