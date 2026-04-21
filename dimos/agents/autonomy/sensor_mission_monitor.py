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

import csv
from dataclasses import dataclass, field
from io import StringIO
import re
from threading import Event, Thread
import time
from typing import Literal

from langchain_core.messages import HumanMessage
from reactivex.disposable import Disposable
import requests

from dimos.agents.agent import AgentSpec
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_SENSOR_FEED_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1dMfmVTwttaFSXk-F98cUN2oC9qImOlpeWNv97Debay0/export?format=csv&gid=0"
)


Comparison = Literal["gt", "gte", "lt", "lte", "eq", "neq"]
Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class SensorTriggerRule:
    """Rule for turning one sensor reading into an autonomous mission event."""

    name: str
    threshold: float
    comparison: Comparison = "gte"
    severity: Severity = "critical"
    variable: str = "temperature"
    unit: str = ""
    column_contains: str | None = None
    cooldown_s: float = 60.0
    recommended_skill: str = "inspect_zone"
    message_template: str | None = None


@dataclass(frozen=True)
class SensorMissionEvent:
    """High-level event produced from an external sensor feed reading."""

    event_key: str
    feed_url: str
    time_label: str
    zone: str
    variable: str
    value: float
    threshold: float
    comparison: Comparison
    severity: Severity
    unit: str
    source_field: str
    rule_name: str
    recommended_skill: str
    ts: float


@dataclass
class SensorMissionMonitorConfig(ModuleConfig):
    feed_url: str = DEFAULT_SENSOR_FEED_CSV_URL
    poll_interval_s: float = 10.0
    request_timeout_s: float = 5.0
    enabled: bool = True
    time_field: str = "Time"
    ignore_fields: tuple[str, ...] = ("Time", "Timestamp", "timestamp")
    require_idle_for_noncritical: bool = True
    rules: tuple[SensorTriggerRule, ...] = field(
        default_factory=lambda: (
            SensorTriggerRule(
                name="critical_temperature",
                threshold=75.0,
                comparison="gte",
                severity="critical",
                variable="temperature",
                unit="C",
                column_contains="\u00b0C",
                cooldown_s=60.0,
                recommended_skill="inspect_zone",
            ),
        )
    )


class SensorMissionMonitor(Module[SensorMissionMonitorConfig]):
    """Poll an external sensor feed and inject high-level missions into the agent queue."""

    default_config = SensorMissionMonitorConfig
    config: SensorMissionMonitorConfig

    agent_idle: In[bool]
    sensor_mission_event: Out[SensorMissionEvent]

    _agent_spec: AgentSpec

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._seen_event_keys: set[str] = set()
        self._last_sent_by_rule_zone: dict[tuple[str, str, str], float] = {}
        self._agent_is_idle = True

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.agent_idle.subscribe(self._on_agent_idle)))
        if not self.config.enabled:
            logger.info("SensorMissionMonitor disabled")
            return
        self._thread = Thread(
            target=self._poll_loop,
            name=f"{self.__class__.__name__}-thread",
            daemon=True,
        )
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().stop()

    @rpc
    def poll_once(self) -> int:
        """Poll the configured sensor feed once and return the number of events sent."""
        response = requests.get(
            self.config.feed_url,
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        return len(self._process_csv(response.text))

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("Failed to poll autonomous sensor feed")
            self._stop_event.wait(self.config.poll_interval_s)

    def _on_agent_idle(self, idle: bool) -> None:
        self._agent_is_idle = idle

    def _process_csv(self, csv_text: str) -> list[SensorMissionEvent]:
        row = _latest_nonempty_row(csv_text)
        if row is None:
            return []

        events: list[SensorMissionEvent] = []
        time_label = row.get(self.config.time_field, "") or "unknown"

        for field_name, raw_value in row.items():
            if field_name in self.config.ignore_fields or not raw_value:
                continue

            value = _parse_float(raw_value)
            if value is None:
                continue

            for rule in self.config.rules:
                if rule.column_contains and rule.column_contains not in field_name:
                    continue
                if not _matches(value, rule.threshold, rule.comparison):
                    continue

                event = self._build_event(
                    field_name=field_name,
                    raw_time_label=time_label,
                    value=value,
                    rule=rule,
                )
                if self._send_event(event, rule):
                    events.append(event)

        return events

    def _build_event(
        self,
        *,
        field_name: str,
        raw_time_label: str,
        value: float,
        rule: SensorTriggerRule,
    ) -> SensorMissionEvent:
        zone = _zone_from_field(field_name)
        time_label = raw_time_label.strip()
        event_key = (
            f"{self.config.feed_url}|{time_label}|{zone}|{rule.name}|"
            f"{rule.variable}|{value:g}"
        )
        return SensorMissionEvent(
            event_key=event_key,
            feed_url=self.config.feed_url,
            time_label=time_label,
            zone=zone,
            variable=rule.variable,
            value=value,
            threshold=rule.threshold,
            comparison=rule.comparison,
            severity=rule.severity,
            unit=rule.unit,
            source_field=field_name,
            rule_name=rule.name,
            recommended_skill=rule.recommended_skill,
            ts=time.time(),
        )

    def _send_event(self, event: SensorMissionEvent, rule: SensorTriggerRule) -> bool:
        if event.event_key in self._seen_event_keys:
            return False
        if (
            self.config.require_idle_for_noncritical
            and event.severity != "critical"
            and not self._agent_is_idle
        ):
            return False

        cooldown_key = (event.zone, event.variable, event.rule_name)
        now = time.monotonic()
        last_sent = self._last_sent_by_rule_zone.get(cooldown_key)
        if last_sent is not None and now - last_sent < rule.cooldown_s:
            return False

        self.sensor_mission_event.publish(event)
        self._agent_spec.add_message(HumanMessage(content=_mission_message(event, rule)))
        self._seen_event_keys.add(event.event_key)
        self._last_sent_by_rule_zone[cooldown_key] = now
        logger.info(
            "Sent autonomous sensor mission event",
            zone=event.zone,
            variable=event.variable,
            value=event.value,
            threshold=event.threshold,
            severity=event.severity,
            recommended_skill=event.recommended_skill,
        )
        return True


def _latest_nonempty_row(csv_text: str) -> dict[str, str] | None:
    reader = csv.DictReader(StringIO(csv_text))
    latest: dict[str, str] | None = None
    for row in reader:
        normalized = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}
        if any(value for value in normalized.values()):
            latest = normalized
    return latest


def _parse_float(raw_value: str) -> float | None:
    value = raw_value.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _matches(value: float, threshold: float, comparison: Comparison) -> bool:
    if comparison == "gt":
        return value > threshold
    if comparison == "gte":
        return value >= threshold
    if comparison == "lt":
        return value < threshold
    if comparison == "lte":
        return value <= threshold
    if comparison == "eq":
        return value == threshold
    if comparison == "neq":
        return value != threshold
    raise ValueError(f"Unsupported comparison: {comparison}")


def _zone_from_field(field_name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", field_name).strip()


def _mission_message(event: SensorMissionEvent, rule: SensorTriggerRule) -> str:
    if rule.message_template:
        return rule.message_template.format(
            comparison=event.comparison,
            recommended_skill=event.recommended_skill,
            severity=event.severity,
            source_field=event.source_field,
            threshold=event.threshold,
            time_label=event.time_label,
            unit=event.unit,
            value=event.value,
            variable=event.variable,
            zone=event.zone,
        )

    unit = f" {event.unit}" if event.unit else ""
    if event.recommended_skill == "inspect_zone":
        skill_instruction = (
            f"Call inspect_zone with zone_name='{event.zone}', "
            f"reason='{event.variable}={event.value:g}{unit}'. "
            "The skill should navigate to that zone and capture one camera image."
        )
    else:
        skill_instruction = (
            f"Use the recommended skill '{event.recommended_skill}' if it is available."
        )

    return (
        "Autonomous sensor event detected. "
        f"At time '{event.time_label}', zone '{event.zone}' reported "
        f"{event.variable}={event.value:g}{unit}, which crossed the "
        f"{event.comparison} threshold {event.threshold:g}{unit}. "
        f"Severity is '{event.severity}'. "
        f"{skill_instruction} "
        "If you act, briefly speak what you are doing."
    )


sensor_mission_monitor = SensorMissionMonitor.blueprint

__all__ = [
    "DEFAULT_SENSOR_FEED_CSV_URL",
    "SensorMissionEvent",
    "SensorMissionMonitor",
    "SensorMissionMonitorConfig",
    "SensorTriggerRule",
    "sensor_mission_monitor",
]
