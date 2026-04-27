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

"""Agent skills for the Go2 + drone LiDAR fusion simulation."""

from dataclasses import dataclass
import json
import math
from threading import Event, RLock, Thread, current_thread
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.autonomy.zones import DEFAULT_ZONE_DEFINITIONS, ZoneDefinition, find_zone
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped
from dimos.msgs.nav_msgs import CostValues, OccupancyGrid, Path
from dimos.navigation.replanning_a_star.min_cost_astar import min_cost_astar
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DRONE_PATH_TIMEOUT_S = 1.5
_DRONE_PATH_GOAL_TOLERANCE_M = 0.75
_DRONE_SCOUT_MIN_MISSION_TIMEOUT_S = 30.0
_DRONE_STUCK_TIMEOUT_S = 6.0
_DRONE_PROGRESS_EPS_M = 0.05
_SCOUT_POLL_INTERVAL_S = 0.1
_SCOUT_WAYPOINT_REACHED_M = 0.45
_SCOUT_MAP_MARGIN_M = 0.25
_PATH_REVEAL_TARGET_SETTLE_S = 1.5
_AUTO_RECOVERY_TIMEOUT_S = 60.0
_FUSED_ZONE_UNKNOWN_READY = 0.45
_FUSED_ZONE_UNKNOWN_IMPROVEMENT = 0.25


@dataclass(frozen=True)
class _ScoutResult:
    ok: bool
    reason: str
    waited_s: float
    drone_goal_x: float | None = None
    drone_goal_y: float | None = None


class DroneGo2FusionSkillContainer(Module):
    """Robot-scoped skills for the drone-Go2 fusion simulation."""

    drone_odom: In[PoseStamped]
    go2_odom: In[PoseStamped]
    drone_path: In[Path]
    drone_global_costmap: In[OccupancyGrid]
    go2_fused_global_costmap: In[OccupancyGrid]
    go2_goal_observed: In[PoseStamped]
    go2_recovery_goal_request: In[PoseStamped]

    drone_goal_request: Out[PoseStamped]
    go2_goal_request: Out[PoseStamped]

    _latest_drone_odom: PoseStamped | None
    _latest_go2_odom: PoseStamped | None
    _latest_drone_path: Path | None
    _latest_drone_path_update_monotonic: float
    _latest_drone_costmap: OccupancyGrid | None
    _latest_go2_fused_costmap: OccupancyGrid | None
    _latest_go2_goal_request: PoseStamped | None
    _auto_recovery_goal: PoseStamped | None
    _auto_recovery_thread: Thread | None
    _auto_recovery_stop: Event
    _auto_recovery_lock: RLock

    def __init__(self) -> None:
        super().__init__()
        self._latest_drone_odom = None
        self._latest_go2_odom = None
        self._latest_drone_path = None
        self._latest_drone_path_update_monotonic = 0.0
        self._latest_drone_costmap = None
        self._latest_go2_fused_costmap = None
        self._latest_go2_goal_request = None
        self._auto_recovery_goal = None
        self._auto_recovery_thread = None
        self._auto_recovery_stop = Event()
        self._auto_recovery_lock = RLock()

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.drone_odom.subscribe(self._on_drone_odom)))
        self._disposables.add(Disposable(self.go2_odom.subscribe(self._on_go2_odom)))
        self._disposables.add(Disposable(self.drone_path.subscribe(self._on_drone_path)))
        self._disposables.add(
            Disposable(self.drone_global_costmap.subscribe(self._on_drone_costmap))
        )
        self._disposables.add(
            Disposable(self.go2_fused_global_costmap.subscribe(self._on_go2_fused_costmap))
        )
        self._disposables.add(Disposable(self.go2_goal_observed.subscribe(self._on_go2_goal_observed)))
        self._disposables.add(
            Disposable(
                self.go2_recovery_goal_request.subscribe(self._on_go2_recovery_goal_request)
            )
        )

    @rpc
    def stop(self) -> None:
        self._auto_recovery_stop.set()
        with self._auto_recovery_lock:
            thread = self._auto_recovery_thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=2.0)
        super().stop()

    def _on_drone_odom(self, msg: PoseStamped) -> None:
        self._latest_drone_odom = msg

    def _on_go2_odom(self, msg: PoseStamped) -> None:
        self._latest_go2_odom = msg

    def _on_drone_path(self, msg: Path) -> None:
        self._latest_drone_path = msg
        self._latest_drone_path_update_monotonic = time.monotonic()

    def _on_drone_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_drone_costmap = msg

    def _on_go2_fused_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_go2_fused_costmap = msg

    def _on_go2_goal_observed(self, msg: PoseStamped) -> None:
        self._latest_go2_goal_request = msg
        with self._auto_recovery_lock:
            active_goal = self._auto_recovery_goal
            stop_event = self._auto_recovery_stop
        if active_goal is not None and not _same_goal_xy(msg, active_goal):
            logger.info(
                "Cancelling auto recovery because a newer Go2 goal was observed.",
                old_x=round(active_goal.position.x, 2),
                old_y=round(active_goal.position.y, 2),
                new_x=round(msg.position.x, 2),
                new_y=round(msg.position.y, 2),
            )
            stop_event.set()

    def _on_go2_recovery_goal_request(self, msg: PoseStamped) -> None:
        observed_goal = self._latest_go2_goal_request
        if observed_goal is not None and not _same_goal_xy(observed_goal, msg):
            logger.info(
                "Ignoring stale Go2 recovery request because the observed goal changed.",
                requested_x=round(msg.position.x, 2),
                requested_y=round(msg.position.y, 2),
                observed_x=round(observed_goal.position.x, 2),
                observed_y=round(observed_goal.position.y, 2),
            )
            return

        if self._latest_go2_odom is None:
            logger.warning("Ignoring Go2 recovery request because current Go2 pose is missing.")
            return

        with self._auto_recovery_lock:
            thread = self._auto_recovery_thread
            if thread is not None and thread.is_alive():
                logger.info(
                    "Auto recovery already active; ignoring duplicate trigger.",
                    x=round(msg.position.x, 2),
                    y=round(msg.position.y, 2),
                )
                return

            self._auto_recovery_stop = Event()
            goal = _copy_pose_stamped(msg)
            self._auto_recovery_goal = goal
            thread = Thread(
                target=self._run_auto_recovery,
                args=(goal, self._auto_recovery_stop),
                name=f"{self.__class__.__name__}-auto-recovery",
                daemon=True,
            )
            self._auto_recovery_thread = thread

        logger.info(
            "Starting automatic drone path-reveal recovery for Go2.",
            x=round(goal.position.x, 2),
            y=round(goal.position.y, 2),
        )
        thread.start()

    @skill
    def drone_go_to(self, x: float, y: float) -> str:
        """Send only the simulated drone to a world-frame XY goal.

        Use this when the user explicitly asks the drone to move, scout, inspect,
        or map an area. This does not command the Go2.

        Args:
            x: World-frame X coordinate in meters.
            y: World-frame Y coordinate in meters.
        """
        goal = _goal_pose(x, y)
        self.drone_goal_request.publish(goal)
        return f"Drone goal published: x={x:.2f}, y={y:.2f}."

    @skill
    def go2_go_to(self, x: float, y: float) -> str:
        """Send only the Go2 to a world-frame XY goal using the fused costmap.

        Use this when the user explicitly asks the Go2 to move. This does not
        command the drone.

        Args:
            x: World-frame X coordinate in meters.
            y: World-frame Y coordinate in meters.
        """
        goal = _goal_pose(x, y)
        self.go2_goal_request.publish(goal)
        return f"Go2 goal published: x={x:.2f}, y={y:.2f}."

    @skill
    def fusion_scout_then_go2(self, x: float, y: float, scout_seconds: float = 30.0) -> str:
        """Scout a goal with the drone first, then send the Go2 to the same goal.

        Use this for goals where the route may be unknown or where drone mapping
        should improve the Go2 fused costmap before the Go2 moves.

        Args:
            x: World-frame X coordinate in meters.
            y: World-frame Y coordinate in meters.
            scout_seconds: Requested mission timeout in seconds. Short values are
                raised to a practical minimum so the drone can finish moving
                intermediate scout waypoints before the Go2 goal is published.
        """
        scout_seconds = max(_DRONE_SCOUT_MIN_MISSION_TIMEOUT_S, scout_seconds)
        scout_result = self._scout_to_goal(x, y, radius_m=1.0, timeout_s=scout_seconds)

        if not scout_result.ok:
            return (
                "Drone scout failed; Go2 goal not published. "
                f"reason={scout_result.reason}; waited={scout_result.waited_s:.1f}s."
            )

        goal = _goal_pose(x, y)
        self.go2_goal_request.publish(goal)

        return (
            f"Drone scout completed for x={x:.2f}, y={y:.2f}; "
            f"reason={scout_result.reason}; waited={scout_result.waited_s:.1f}s; "
            "Go2 goal published."
        )

    @skill
    def drone_go_to_zone(self, zone_name: str) -> str:
        """Send only the simulated drone to a known named zone.

        Use this when the user names Zone A, Zone B, or Zone C and explicitly
        asks the drone to move, scout, inspect, or map that zone.

        Args:
            zone_name: Known zone name, such as "Zone A", "Zone B", or "Zone C".
        """
        zone = _require_zone(zone_name)
        return self.drone_go_to(zone.center_x, zone.center_y)

    @skill
    def go2_go_to_zone(self, zone_name: str) -> str:
        """Send only the Go2 to a known named zone using the fused costmap.

        Use this when the user names Zone A, Zone B, or Zone C and explicitly
        asks the Go2 to move there.

        Args:
            zone_name: Known zone name, such as "Zone A", "Zone B", or "Zone C".
        """
        zone = _require_zone(zone_name)
        return self.go2_go_to(zone.center_x, zone.center_y)

    @skill
    def fusion_scout_zone_then_go2(self, zone_name: str, scout_seconds: float = 30.0) -> str:
        """Scout a known zone with the drone first, then send the Go2 there.

        Use this when the user asks for a named zone and wants drone scouting,
        route discovery, or map fusion before moving the Go2.

        Args:
            zone_name: Known zone name, such as "Zone A", "Zone B", or "Zone C".
            scout_seconds: Requested mission timeout in seconds. Short values are
                raised to a practical minimum so the drone can finish moving
                intermediate scout waypoints before the Go2 goal is published.
        """
        zone = _require_zone(zone_name)
        scout_result = self._scout_to_goal(
            zone.center_x,
            zone.center_y,
            radius_m=zone.default_radius_m,
            timeout_s=max(_DRONE_SCOUT_MIN_MISSION_TIMEOUT_S, scout_seconds),
        )

        if not scout_result.ok:
            return (
                f"Drone scout failed for {zone.name}; Go2 goal not published. "
                f"reason={scout_result.reason}; waited={scout_result.waited_s:.1f}s."
            )

        self.go2_goal_request.publish(_goal_pose(zone.center_x, zone.center_y))
        return (
            f"Drone scout completed for {zone.name}; reason={scout_result.reason}; "
            f"waited={scout_result.waited_s:.1f}s; Go2 goal published."
        )

    @skill
    def fusion_reveal_path_then_go2(
        self,
        x: float,
        y: float,
        scout_seconds: float = 30.0,
    ) -> str:
        """Use the drone to reveal any Go2 A* path to a goal, then move the Go2.

        This is intentionally simpler than optimal or epsilon-optimal scouting.
        The skill snapshots the Go2's current pose and repeatedly asks only
        whether the current fused costmap already admits a path under the same
        min_cost_astar behavior the Go2 planner uses at runtime. Drone scouting
        stops as soon as that path exists, then the Go2 goal is published.

        Args:
            x: World-frame X coordinate in meters.
            y: World-frame Y coordinate in meters.
            scout_seconds: Requested mission timeout in seconds. Short values are
                raised to a practical minimum so the drone can finish moving
                intermediate scout waypoints before the Go2 goal is published.
        """
        go2_start_pose = self._latest_go2_odom
        if go2_start_pose is None:
            return "Drone path reveal failed; Go2 goal not published. reason=missing current Go2 pose."

        reveal_result = self._reveal_path_to_goal(
            go2_start_pose,
            x,
            y,
            timeout_s=max(_DRONE_SCOUT_MIN_MISSION_TIMEOUT_S, scout_seconds),
        )
        if not reveal_result.ok:
            return (
                "Drone path reveal failed; Go2 goal not published. "
                f"reason={reveal_result.reason}; waited={reveal_result.waited_s:.1f}s."
            )

        self.go2_goal_request.publish(_goal_pose(x, y))
        return (
            f"Drone path reveal completed for x={x:.2f}, y={y:.2f}; "
            f"reason={reveal_result.reason}; waited={reveal_result.waited_s:.1f}s; "
            "Go2 goal published."
        )

    @skill
    def fusion_reveal_path_zone_then_go2(
        self,
        zone_name: str,
        scout_seconds: float = 30.0,
    ) -> str:
        """Reveal any Go2 A* path to a known zone, then move the Go2 there.

        Use this when the user names a zone and the goal is simply to uncover a
        traversable Go2 route from the Go2's current pose, without trying to prove
        global optimality. The drone scouts until the current fused costmap admits
        a path under the same min_cost_astar behavior the Go2 planner uses, then
        the Go2 goal is published for the same zone center.

        Args:
            zone_name: Known zone name, such as "Zone A", "Zone B", or "Zone C".
            scout_seconds: Requested mission timeout in seconds. Short values are
                raised to a practical minimum so the drone can finish moving
                intermediate scout waypoints before the Go2 goal is published.
        """
        zone = _require_zone(zone_name)
        go2_start_pose = self._latest_go2_odom
        if go2_start_pose is None:
            return (
                f"Drone path reveal failed for {zone.name}; "
                "Go2 goal not published. reason=missing current Go2 pose."
            )

        reveal_result = self._reveal_path_to_goal(
            go2_start_pose,
            zone.center_x,
            zone.center_y,
            timeout_s=max(_DRONE_SCOUT_MIN_MISSION_TIMEOUT_S, scout_seconds),
        )
        if not reveal_result.ok:
            return (
                f"Drone path reveal failed for {zone.name}; Go2 goal not published. "
                f"reason={reveal_result.reason}; waited={reveal_result.waited_s:.1f}s."
            )

        self.go2_goal_request.publish(_goal_pose(zone.center_x, zone.center_y))
        return (
            f"Drone path reveal completed for {zone.name}; reason={reveal_result.reason}; "
            f"waited={reveal_result.waited_s:.1f}s; Go2 goal published."
        )

    @skill
    def fusion_known_zones(self) -> str:
        """List known named zones and their world-frame coordinates.

        Use this when the user asks what zones are available or when a named zone
        must be resolved to coordinates before navigation.
        """
        return json.dumps(
            {
                "zones": [
                    {
                        "name": zone.name,
                        "x": zone.center_x,
                        "y": zone.center_y,
                        "z": zone.center_z,
                        "radius_m": zone.default_radius_m,
                        "frame_id": zone.frame_id,
                    }
                    for zone in DEFAULT_ZONE_DEFINITIONS
                ]
            },
            sort_keys=True,
        )

    @skill
    def fusion_map_status(self) -> str:
        """Report current drone and Go2 fused map health and robot poses.

        Use this before or after coordinated navigation to check whether the drone
        and Go2 maps are being updated and whether fused map data is available.
        """
        payload: dict[str, Any] = {
            "drone_pose": _pose_payload(self._latest_drone_odom),
            "go2_pose": _pose_payload(self._latest_go2_odom),
            "drone_global_costmap": _costmap_payload(self._latest_drone_costmap),
            "go2_fused_global_costmap": _costmap_payload(self._latest_go2_fused_costmap),
        }
        return json.dumps(payload, sort_keys=True)

    def _run_auto_recovery(self, goal: PoseStamped, stop_event: Event) -> None:
        start_pose = self._latest_go2_odom
        if start_pose is None:
            logger.warning("Auto recovery aborted because the Go2 pose disappeared.")
            self._finish_auto_recovery(stop_event)
            return

        result = self._reveal_path_to_goal(
            _copy_pose_stamped(start_pose),
            goal.position.x,
            goal.position.y,
            timeout_s=_AUTO_RECOVERY_TIMEOUT_S,
            abort_event=stop_event,
        )

        if stop_event.is_set():
            logger.info(
                "Auto recovery cancelled before completion.",
                x=round(goal.position.x, 2),
                y=round(goal.position.y, 2),
            )
            self._finish_auto_recovery(stop_event)
            return

        if result.ok and self._can_republish_recovery_goal(goal):
            self.go2_goal_request.publish(_copy_pose_stamped(goal))
            logger.info(
                "Auto recovery revealed a Go2 path; republished original goal.",
                x=round(goal.position.x, 2),
                y=round(goal.position.y, 2),
                waited_s=round(result.waited_s, 1),
                reason=result.reason,
            )
        else:
            logger.info(
                "Auto recovery did not republish the Go2 goal.",
                x=round(goal.position.x, 2),
                y=round(goal.position.y, 2),
                waited_s=round(result.waited_s, 1),
                reason=result.reason,
                ok=result.ok,
            )

        self._finish_auto_recovery(stop_event)

    def _can_republish_recovery_goal(self, goal: PoseStamped) -> bool:
        observed_goal = self._latest_go2_goal_request
        if observed_goal is None:
            return True
        return _same_goal_xy(observed_goal, goal)

    def _finish_auto_recovery(self, stop_event: Event) -> None:
        with self._auto_recovery_lock:
            if self._auto_recovery_stop is stop_event:
                self._auto_recovery_goal = None
                self._auto_recovery_thread = None

    def _scout_to_goal(
        self,
        x: float,
        y: float,
        *,
        radius_m: float,
        timeout_s: float,
    ) -> _ScoutResult:
        start = time.monotonic()
        deadline = start + timeout_s
        before_unknown_ratio = _unknown_ratio_near(
            self._latest_go2_fused_costmap,
            x,
            y,
            radius_m,
        )
        scout_goals_sent = 0

        while time.monotonic() <= deadline:
            completion_reason = self._scout_completion_reason(
                x,
                y,
                radius_m,
                before_unknown_ratio,
            )
            if completion_reason is not None and scout_goals_sent > 0:
                return _ScoutResult(True, completion_reason, time.monotonic() - start)

            scout_goal = _select_scout_goal(
                self._latest_drone_odom,
                self._latest_drone_costmap,
                x,
                y,
            )
            if scout_goal is None:
                return _ScoutResult(
                    False,
                    "missing drone pose or drone costmap for scout waypoint selection",
                    time.monotonic() - start,
                )

            issued_at = time.monotonic()
            self.drone_goal_request.publish(_goal_pose(*scout_goal))
            scout_goals_sent += 1

            path_timeout = min(_DRONE_PATH_TIMEOUT_S, max(0.1, deadline - issued_at))
            if not self._wait_for_drone_path_after(issued_at, path_timeout, scout_goal):
                return _ScoutResult(
                    False,
                    (
                        "drone planner did not produce a path to scout waypoint "
                        f"x={scout_goal[0]:.2f}, y={scout_goal[1]:.2f}"
                    ),
                    time.monotonic() - start,
                    drone_goal_x=scout_goal[0],
                    drone_goal_y=scout_goal[1],
                )

            waypoint_result = self._wait_for_scout_waypoint(
                scout_goal,
                x,
                y,
                radius_m,
                before_unknown_ratio,
                start,
                deadline,
            )
            if waypoint_result is not None:
                if waypoint_result.ok:
                    return waypoint_result
                return _ScoutResult(
                    False,
                    waypoint_result.reason,
                    time.monotonic() - start,
                    drone_goal_x=scout_goal[0],
                    drone_goal_y=scout_goal[1],
                )

        return _ScoutResult(
            False,
            "scout timed out before drone pose, drone map, or Go2 fused map completion criteria were met",
            time.monotonic() - start,
        )

    def _wait_for_drone_path_after(
        self,
        issued_at: float,
        timeout_s: float,
        scout_goal: tuple[float, float],
        *,
        abort_event: Event | None = None,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() <= deadline:
            if _abort_requested(abort_event):
                return False
            path = self._latest_drone_path
            last_pose = path.last() if path is not None else None
            if (
                path is not None
                and path.poses
                and last_pose is not None
                and self._latest_drone_path_update_monotonic >= issued_at
                and _pose_near_xy(
                    last_pose,
                    scout_goal[0],
                    scout_goal[1],
                    _DRONE_PATH_GOAL_TOLERANCE_M,
                )
            ):
                return True
            time.sleep(_SCOUT_POLL_INTERVAL_S)
        return False

    def _wait_for_scout_waypoint(
        self,
        scout_goal: tuple[float, float],
        target_x: float,
        target_y: float,
        radius_m: float,
        before_unknown_ratio: float | None,
        mission_start: float,
        mission_deadline: float,
    ) -> _ScoutResult | None:
        last_progress_pose = self._latest_drone_odom
        last_progress_time = time.monotonic()

        while time.monotonic() <= mission_deadline:
            completion_reason = self._scout_completion_reason(
                target_x,
                target_y,
                radius_m,
                before_unknown_ratio,
            )
            if completion_reason is not None:
                return _ScoutResult(True, completion_reason, time.monotonic() - mission_start)

            current_pose = self._latest_drone_odom
            if _pose_near_xy(
                current_pose,
                scout_goal[0],
                scout_goal[1],
                _SCOUT_WAYPOINT_REACHED_M,
            ):
                return None

            if current_pose is not None:
                if last_progress_pose is None:
                    last_progress_pose = current_pose
                    last_progress_time = time.monotonic()
                elif _pose_distance_xy(current_pose, last_progress_pose) >= _DRONE_PROGRESS_EPS_M:
                    last_progress_pose = current_pose
                    last_progress_time = time.monotonic()

            if time.monotonic() - last_progress_time >= _DRONE_STUCK_TIMEOUT_S:
                return _ScoutResult(
                    False,
                    (
                        "drone appears stuck before reaching scout waypoint "
                        f"x={scout_goal[0]:.2f}, y={scout_goal[1]:.2f}"
                    ),
                    time.monotonic() - mission_start,
                )

            time.sleep(_SCOUT_POLL_INTERVAL_S)

        return _ScoutResult(
            False,
            "scout mission timed out before completion criteria were met",
            time.monotonic() - mission_start,
        )

    def _scout_completion_reason(
        self,
        x: float,
        y: float,
        radius_m: float,
        before_unknown_ratio: float | None,
    ) -> str | None:
        if _pose_near_xy(self._latest_drone_odom, x, y, radius_m):
            return "drone_pose_near_goal"

        if _costmap_contains_xy(self._latest_drone_costmap, x, y):
            return "drone_costmap_contains_goal"

        current_unknown_ratio = _unknown_ratio_near(
            self._latest_go2_fused_costmap,
            x,
            y,
            radius_m,
        )
        if current_unknown_ratio is None:
            return None

        if current_unknown_ratio <= _FUSED_ZONE_UNKNOWN_READY:
            return "go2_fused_zone_unknown_ratio_ready"

        if (
            before_unknown_ratio is not None
            and before_unknown_ratio - current_unknown_ratio >= _FUSED_ZONE_UNKNOWN_IMPROVEMENT
        ):
            return "go2_fused_zone_unknown_ratio_reduced"

        return None

    def _reveal_path_to_goal(
        self,
        go2_start_pose: PoseStamped,
        x: float,
        y: float,
        *,
        timeout_s: float,
        abort_event: Event | None = None,
    ) -> _ScoutResult:
        start = time.monotonic()
        deadline = start + timeout_s

        while time.monotonic() <= deadline:
            if _abort_requested(abort_event):
                return _ScoutResult(
                    False,
                    "path-reveal recovery cancelled because a newer Go2 goal was observed",
                    time.monotonic() - start,
                )

            completion_reason = _known_path_completion_reason(
                self._latest_go2_fused_costmap,
                go2_start_pose,
                x,
                y,
            )
            if completion_reason is not None:
                return _ScoutResult(True, completion_reason, time.monotonic() - start)

            scout_goal = _select_reveal_path_goal(
                self._latest_drone_odom,
                self._latest_drone_costmap,
                self._latest_go2_fused_costmap,
                go2_start_pose.position.x,
                go2_start_pose.position.y,
                x,
                y,
            )
            if scout_goal is None:
                return _ScoutResult(
                    False,
                    "missing drone pose or costmap for path-reveal waypoint selection",
                    time.monotonic() - start,
                )

            issued_at = time.monotonic()
            self.drone_goal_request.publish(_goal_pose(*scout_goal))

            path_timeout = min(_DRONE_PATH_TIMEOUT_S, max(0.1, deadline - issued_at))
            if not self._wait_for_drone_path_after(
                issued_at,
                path_timeout,
                scout_goal,
                abort_event=abort_event,
            ):
                if _abort_requested(abort_event):
                    return _ScoutResult(
                        False,
                        "path-reveal recovery cancelled because a newer Go2 goal was observed",
                        time.monotonic() - start,
                    )
                return _ScoutResult(
                    False,
                    (
                        "drone planner did not produce a path to path-reveal waypoint "
                        f"x={scout_goal[0]:.2f}, y={scout_goal[1]:.2f}"
                    ),
                    time.monotonic() - start,
                    drone_goal_x=scout_goal[0],
                    drone_goal_y=scout_goal[1],
                )

            waypoint_result = self._wait_for_reveal_path_waypoint(
                scout_goal,
                go2_start_pose,
                x,
                y,
                start,
                deadline,
                abort_event=abort_event,
            )
            if waypoint_result is not None:
                if waypoint_result.ok:
                    return waypoint_result
                return _ScoutResult(
                    False,
                    waypoint_result.reason,
                    time.monotonic() - start,
                    drone_goal_x=scout_goal[0],
                    drone_goal_y=scout_goal[1],
                )

        return _ScoutResult(
            False,
            "path-reveal mission timed out before a Go2 A* path was found",
            time.monotonic() - start,
        )

    def _wait_for_reveal_path_waypoint(
        self,
        scout_goal: tuple[float, float],
        go2_start_pose: PoseStamped,
        target_x: float,
        target_y: float,
        mission_start: float,
        mission_deadline: float,
        *,
        abort_event: Event | None = None,
    ) -> _ScoutResult | None:
        last_progress_pose = self._latest_drone_odom
        last_progress_time = time.monotonic()

        while time.monotonic() <= mission_deadline:
            if _abort_requested(abort_event):
                return _ScoutResult(
                    False,
                    "path-reveal recovery cancelled because a newer Go2 goal was observed",
                    time.monotonic() - mission_start,
                )

            completion_reason = _known_path_completion_reason(
                self._latest_go2_fused_costmap,
                go2_start_pose,
                target_x,
                target_y,
            )
            if completion_reason is not None:
                return _ScoutResult(True, completion_reason, time.monotonic() - mission_start)

            current_pose = self._latest_drone_odom
            if _pose_near_xy(
                current_pose,
                scout_goal[0],
                scout_goal[1],
                _SCOUT_WAYPOINT_REACHED_M,
            ):
                if math.hypot(scout_goal[0] - target_x, scout_goal[1] - target_y) <= (
                    _SCOUT_WAYPOINT_REACHED_M
                ):
                    settle_deadline = min(
                        mission_deadline,
                        time.monotonic() + _PATH_REVEAL_TARGET_SETTLE_S,
                    )
                    while time.monotonic() <= settle_deadline:
                        if _abort_requested(abort_event):
                            return _ScoutResult(
                                False,
                                "path-reveal recovery cancelled because a newer Go2 goal was observed",
                                time.monotonic() - mission_start,
                            )
                        completion_reason = _known_path_completion_reason(
                            self._latest_go2_fused_costmap,
                            go2_start_pose,
                            target_x,
                            target_y,
                        )
                        if completion_reason is not None:
                            return _ScoutResult(
                                True,
                                completion_reason,
                                time.monotonic() - mission_start,
                            )
                        time.sleep(_SCOUT_POLL_INTERVAL_S)
                    return _ScoutResult(
                        False,
                        (
                            "drone reached the target area but no Go2 A* path "
                            "was found after map updates settled"
                        ),
                        time.monotonic() - mission_start,
                    )
                return None

            if current_pose is not None:
                if last_progress_pose is None:
                    last_progress_pose = current_pose
                    last_progress_time = time.monotonic()
                elif _pose_distance_xy(current_pose, last_progress_pose) >= _DRONE_PROGRESS_EPS_M:
                    last_progress_pose = current_pose
                    last_progress_time = time.monotonic()

            if time.monotonic() - last_progress_time >= _DRONE_STUCK_TIMEOUT_S:
                return _ScoutResult(
                    False,
                    (
                        "drone appears stuck before reaching path-reveal waypoint "
                        f"x={scout_goal[0]:.2f}, y={scout_goal[1]:.2f}"
                    ),
                    time.monotonic() - mission_start,
                )

            time.sleep(_SCOUT_POLL_INTERVAL_S)

        return _ScoutResult(
            False,
            "path-reveal mission timed out before a Go2 A* path was found",
            time.monotonic() - mission_start,
        )


def _goal_pose(x: float, y: float) -> PoseStamped:
    return PoseStamped(
        position=(x, y, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )


def _copy_pose_stamped(pose: PoseStamped) -> PoseStamped:
    return PoseStamped(
        ts=pose.ts,
        frame_id=pose.frame_id,
        position=(pose.position.x, pose.position.y, pose.position.z),
        orientation=(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
    )


def _require_zone(zone_name: str) -> ZoneDefinition:
    zone = find_zone(zone_name)
    if zone is None:
        known = ", ".join(zone.name for zone in DEFAULT_ZONE_DEFINITIONS)
        raise ValueError(f"Unknown zone {zone_name!r}. Known zones: {known}.")
    return zone


def _pose_near_xy(pose: PoseStamped | None, x: float, y: float, radius_m: float) -> bool:
    if pose is None:
        return False
    return math.hypot(pose.position.x - x, pose.position.y - y) <= radius_m


def _pose_distance_xy(first: PoseStamped, second: PoseStamped) -> float:
    return math.hypot(
        first.position.x - second.position.x,
        first.position.y - second.position.y,
    )


def _same_goal_xy(first: PoseStamped, second: PoseStamped, tolerance_m: float = 0.05) -> bool:
    return _pose_distance_xy(first, second) <= tolerance_m


def _abort_requested(abort_event: Event | None) -> bool:
    return abort_event is not None and abort_event.is_set()


def _costmap_bounds(costmap: OccupancyGrid) -> tuple[float, float, float, float]:
    origin_x = costmap.origin.position.x
    origin_y = costmap.origin.position.y
    return (
        origin_x,
        origin_y,
        origin_x + costmap.width * costmap.resolution,
        origin_y + costmap.height * costmap.resolution,
    )


def _costmap_contains_xy(
    costmap: OccupancyGrid | None,
    x: float,
    y: float,
    *,
    margin_m: float = 0.0,
) -> bool:
    if costmap is None or costmap.width <= 0 or costmap.height <= 0:
        return False

    min_x, min_y, max_x, max_y = _costmap_bounds(costmap)
    return min_x + margin_m <= x <= max_x - margin_m and min_y + margin_m <= y <= max_y - margin_m


def _select_scout_goal(
    drone_pose: PoseStamped | None,
    drone_costmap: OccupancyGrid | None,
    target_x: float,
    target_y: float,
) -> tuple[float, float] | None:
    if drone_pose is None or drone_costmap is None:
        return None

    if _costmap_contains_xy(drone_costmap, target_x, target_y):
        return target_x, target_y

    min_x, min_y, max_x, max_y = _costmap_bounds(drone_costmap)
    min_x += _SCOUT_MAP_MARGIN_M
    min_y += _SCOUT_MAP_MARGIN_M
    max_x -= _SCOUT_MAP_MARGIN_M
    max_y -= _SCOUT_MAP_MARGIN_M
    if min_x >= max_x or min_y >= max_y:
        return None

    pose_x = drone_pose.position.x
    pose_y = drone_pose.position.y
    dx = target_x - pose_x
    dy = target_y - pose_y
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return target_x, target_y

    current_inside = min_x <= pose_x <= max_x and min_y <= pose_y <= max_y
    if not current_inside:
        return (
            min(max(target_x, min_x), max_x),
            min(max(target_y, min_y), max_y),
        )

    boundary_fractions: list[float] = []
    if dx > 0.0:
        boundary_fractions.append((max_x - pose_x) / dx)
    elif dx < 0.0:
        boundary_fractions.append((min_x - pose_x) / dx)
    if dy > 0.0:
        boundary_fractions.append((max_y - pose_y) / dy)
    elif dy < 0.0:
        boundary_fractions.append((min_y - pose_y) / dy)

    positive_fractions = [fraction for fraction in boundary_fractions if fraction > 0.0]
    if not positive_fractions:
        return None

    fraction = min(1.0, min(positive_fractions) * 0.85)
    return (
        min(max(pose_x + dx * fraction, min_x), max_x),
        min(max(pose_y + dy * fraction, min_y), max_y),
    )


def _select_reveal_path_goal(
    drone_pose: PoseStamped | None,
    drone_costmap: OccupancyGrid | None,
    fused_costmap: OccupancyGrid | None,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
) -> tuple[float, float] | None:
    # Prefer the farthest still-unknown point that already lies inside the current
    # drone map bounds. This pushes the drone toward the frontier of the missing
    # corridor instead of repeatedly selecting the first unknown cell near the
    # Go2 start pose, which can cause goal churn without extending coverage.
    scout_target = _farthest_unknown_point_on_line_in_costmap(
        fused_costmap,
        drone_costmap,
        start_x,
        start_y,
        target_x,
        target_y,
    )
    if scout_target is None:
        scout_target = (target_x, target_y)
    return _select_scout_goal(
        drone_pose,
        drone_costmap,
        scout_target[0],
        scout_target[1],
    )


def _farthest_unknown_point_on_line_in_costmap(
    costmap: OccupancyGrid | None,
    drone_costmap: OccupancyGrid | None,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
) -> tuple[float, float] | None:
    if (
        costmap is None
        or costmap.width <= 0
        or costmap.height <= 0
        or drone_costmap is None
        or drone_costmap.width <= 0
        or drone_costmap.height <= 0
    ):
        return None

    distance = math.hypot(target_x - start_x, target_y - start_y)
    if distance <= 1e-6:
        return None

    step_m = max(costmap.resolution * 0.5, 0.05)
    n_steps = max(1, math.ceil(distance / step_m))
    best_point: tuple[float, float] | None = None
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        x = start_x + (target_x - start_x) * alpha
        y = start_y + (target_y - start_y) * alpha
        if not _costmap_contains_xy(costmap, x, y):
            break
        if not _costmap_contains_xy(drone_costmap, x, y, margin_m=_SCOUT_MAP_MARGIN_M):
            break

        grid_x = int((x - costmap.origin.position.x) / costmap.resolution + 0.5)
        grid_y = int((y - costmap.origin.position.y) / costmap.resolution + 0.5)
        if costmap.grid[grid_y, grid_x] == int(CostValues.UNKNOWN):
            best_point = (x, y)
    return best_point


def _unknown_ratio_near(
    costmap: OccupancyGrid | None,
    x: float,
    y: float,
    radius_m: float,
) -> float | None:
    if costmap is None or costmap.width <= 0 or costmap.height <= 0:
        return None

    grid_x = int((x - costmap.origin.position.x) / costmap.resolution + 0.5)
    grid_y = int((y - costmap.origin.position.y) / costmap.resolution + 0.5)
    if grid_x < 0 or grid_y < 0 or grid_x >= costmap.width or grid_y >= costmap.height:
        return None

    radius_cells = max(1, math.ceil(radius_m / costmap.resolution))
    min_x = max(0, grid_x - radius_cells)
    max_x = min(costmap.width, grid_x + radius_cells + 1)
    min_y = max(0, grid_y - radius_cells)
    max_y = min(costmap.height, grid_y + radius_cells + 1)
    window = costmap.grid[min_y:max_y, min_x:max_x]
    if window.size == 0:
        return None
    return float(np.mean(window == int(CostValues.UNKNOWN)))


def _known_path_completion_reason(
    fused_costmap: OccupancyGrid | None,
    go2_start_pose: PoseStamped | None,
    target_x: float,
    target_y: float,
) -> str | None:
    if go2_start_pose is None:
        return None
    if _known_path_exists(
        fused_costmap,
        go2_start_pose.position.x,
        go2_start_pose.position.y,
        target_x,
        target_y,
    ):
        return "go2_astar_path_exists"
    return None


def _known_path_exists(
    fused_costmap: OccupancyGrid | None,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
) -> bool:
    if fused_costmap is None or fused_costmap.width <= 0 or fused_costmap.height <= 0:
        return False
    if not _costmap_contains_xy(fused_costmap, start_x, start_y):
        return False
    if not _costmap_contains_xy(fused_costmap, target_x, target_y):
        return False

    path = min_cost_astar(
        fused_costmap,
        (target_x, target_y),
        start=(start_x, start_y),
    )
    return path is not None


def _pose_payload(pose: PoseStamped | None) -> dict[str, float] | None:
    if pose is None:
        return None
    return {
        "x": pose.position.x,
        "y": pose.position.y,
        "z": pose.position.z,
        "ts": pose.ts,
    }


def _costmap_payload(costmap: OccupancyGrid | None) -> dict[str, float | int] | None:
    if costmap is None:
        return None
    return {
        "width": costmap.width,
        "height": costmap.height,
        "resolution": costmap.resolution,
        "origin_x": costmap.origin.position.x,
        "origin_y": costmap.origin.position.y,
        "unknown_percent": costmap.unknown_percent,
        "occupied_percent": costmap.occupied_percent,
        "free_percent": costmap.free_percent,
        "ts": costmap.ts,
    }


drone_go2_fusion_skills = DroneGo2FusionSkillContainer.blueprint

__all__ = [
    "DroneGo2FusionSkillContainer",
    "drone_go2_fusion_skills",
]
