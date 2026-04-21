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

from pathlib import Path

import numpy as np

from dimos.agents.autonomy.zones import ZoneDefinition
from dimos.agents.skills.zone_inspection import (
    DEFAULT_ZONE_INSPECTION_OUTPUT_DIR,
    ZoneInspectionConfig,
    ZoneInspectionSkillContainer,
    inspection_goal_pose,
)
from dimos.msgs.geometry_msgs import PoseStamped, Vector3
from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.navigation.base import NavigationState


class FakeNavigator:
    def __init__(self) -> None:
        self.goals: list[PoseStamped] = []

    def set_goal(self, goal: PoseStamped) -> bool:
        self.goals.append(goal)
        return True

    def get_state(self) -> NavigationState:
        return NavigationState.IDLE

    def is_goal_reached(self) -> bool:
        return True

    def cancel_goal(self) -> bool:
        return True


class _DummyRPC:
    def serve_module_rpc(self, _module) -> None:  # type: ignore[no-untyped-def]
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_inspection_goal_pose_uses_zone_center() -> None:
    zone = ZoneDefinition(name="Zone A", center_x=2.0, center_y=3.0, center_z=0.5)
    pose = inspection_goal_pose(zone)

    assert pose.frame_id == zone.frame_id
    assert pose.position.x == 2.0
    assert pose.position.y == 3.0
    assert pose.position.z == 0.5


def test_default_output_dir_is_project_temp() -> None:
    assert Path(ZoneInspectionConfig().output_dir) == DEFAULT_ZONE_INSPECTION_OUTPUT_DIR
    assert DEFAULT_ZONE_INSPECTION_OUTPUT_DIR.parts[-2:] == (
        "temp",
        "dimos_zone_inspections",
    )


def test_inspect_zone_saves_one_image(tmp_path: Path) -> None:
    zone = ZoneDefinition(name="Zone A", center_x=0.0, center_y=0.0)
    skill = ZoneInspectionSkillContainer(
        zones=(zone,),
        output_dir=str(tmp_path),
        navigation_timeout_s=1.0,
        settle_time_s=0.0,
        rpc_transport=_DummyRPC,
    )
    navigator = FakeNavigator()
    skill._navigator = navigator  # type: ignore[assignment]
    skill._latest_image = Image.from_numpy(
        np.zeros((16, 16, 3), dtype=np.uint8),
        format=ImageFormat.RGB,
    )

    try:
        result = skill.inspect_zone(
            zone_name="Zone A",
            reason="temperature=100 C",
            radius_m=1.0,
            views=3,
        )

        assert "Completed inspection" in result
        assert "Captured 1/1 image" in result
        assert len(navigator.goals) == 1
        assert navigator.goals[0].position.x == zone.center_x
        assert navigator.goals[0].position.y == zone.center_y
        assert len(list(tmp_path.rglob("*.jpg"))) == 1
    finally:
        skill.stop()


def test_inspect_zone_ignores_route_defaults(tmp_path: Path) -> None:
    zone = ZoneDefinition(
        name="Zone C",
        center_x=1.0,
        center_y=-2.0,
        default_radius_m=2.5,
        default_views=2,
    )
    skill = ZoneInspectionSkillContainer(
        zones=(zone,),
        output_dir=str(tmp_path),
        navigation_timeout_s=1.0,
        settle_time_s=0.0,
        rpc_transport=_DummyRPC,
    )
    navigator = FakeNavigator()
    skill._navigator = navigator  # type: ignore[assignment]

    try:
        result = skill.inspect_zone(
            zone_name="Zone C",
            reason="temperature=100 C",
            radius_m=0.0,
            views=0,
        )

        assert "Completed inspection" in result
        assert len(navigator.goals) == 1
        assert navigator.goals[0].position.x == 1.0
        assert navigator.goals[0].position.y == -2.0
    finally:
        skill.stop()


def test_inspect_zone_current_odom_does_not_change_single_goal(tmp_path: Path) -> None:
    zone = ZoneDefinition(
        name="Zone B",
        center_x=1.0,
        center_y=1.0,
        default_radius_m=2.0,
        default_views=4,
    )
    skill = ZoneInspectionSkillContainer(
        zones=(zone,),
        output_dir=str(tmp_path),
        navigation_timeout_s=1.0,
        settle_time_s=0.0,
        rpc_transport=_DummyRPC,
    )
    navigator = FakeNavigator()
    skill._navigator = navigator  # type: ignore[assignment]
    skill._latest_odom = PoseStamped(position=Vector3(-1.0, 1.0, 0.0))

    try:
        result = skill.inspect_zone(
            zone_name="Zone B",
            reason="temperature=100 C",
            radius_m=0.0,
            views=0,
        )

        assert "Completed inspection" in result
        assert len(navigator.goals) == 1
        assert navigator.goals[0].position.x == 1.0
        assert navigator.goals[0].position.y == 1.0
    finally:
        skill.stop()
