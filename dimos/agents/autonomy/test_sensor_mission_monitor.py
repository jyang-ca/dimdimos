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

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.autonomy.sensor_mission_monitor import (
    SensorMissionMonitor,
    SensorTriggerRule,
)


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)


class _DummyRPC:
    def serve_module_rpc(self, _module) -> None:  # type: ignore[no-untyped-def]
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_process_csv_sends_latest_temperature_mission() -> None:
    monitor = SensorMissionMonitor(
        feed_url="https://example.invalid/sensor.csv",
        rpc_transport=_DummyRPC,
    )
    fake_agent = FakeAgent()
    monitor._agent_spec = fake_agent  # type: ignore[assignment]

    try:
        events = monitor._process_csv(
            "Time,Zone A (\u00b0C),Zone B (\u00b0C),Zone C (\u00b0C)\n"
            "10:00,22.1,23,21.8\n"
            "11:00,22.5,23.4,22\n"
            "12:00,90,23.4,22\n"
        )

        assert len(events) == 1
        event = events[0]
        assert event.time_label == "12:00"
        assert event.zone == "Zone A"
        assert event.variable == "temperature"
        assert event.value == 90.0
        assert event.recommended_skill == "inspect_zone"

        assert len(fake_agent.messages) == 1
        message = fake_agent.messages[0]
        assert isinstance(message, HumanMessage)
        assert "Autonomous sensor event" in str(message.content)
        assert "inspect_zone" in str(message.content)
        assert "Zone A" in str(message.content)
        assert "90" in str(message.content)
        assert "capture one camera image" in str(message.content)
    finally:
        monitor.stop()


def test_process_csv_is_generic_for_non_temperature_rules() -> None:
    monitor = SensorMissionMonitor(
        feed_url="https://example.invalid/sensor.csv",
        rpc_transport=_DummyRPC,
        rules=(
            SensorTriggerRule(
                name="high_humidity",
                threshold=70.0,
                comparison="gte",
                severity="warning",
                variable="humidity",
                unit="%",
                column_contains="%RH",
            ),
        ),
    )
    fake_agent = FakeAgent()
    monitor._agent_spec = fake_agent  # type: ignore[assignment]

    try:
        events = monitor._process_csv(
            "Time,Zone A (%RH),Zone B (%RH)\n"
            "10:00,30,40\n"
            "11:00,82,41\n"
        )

        assert len(events) == 1
        assert events[0].zone == "Zone A"
        assert events[0].variable == "humidity"
        assert events[0].severity == "warning"
        assert len(fake_agent.messages) == 1
    finally:
        monitor.stop()


def test_process_csv_dedupes_repeated_events() -> None:
    monitor = SensorMissionMonitor(
        feed_url="https://example.invalid/sensor.csv",
        rpc_transport=_DummyRPC,
    )
    fake_agent = FakeAgent()
    monitor._agent_spec = fake_agent  # type: ignore[assignment]
    csv_text = "Time,Zone A (\u00b0C)\n12:00,90\n"

    try:
        assert len(monitor._process_csv(csv_text)) == 1
        assert len(monitor._process_csv(csv_text)) == 0
        assert len(fake_agent.messages) == 1
    finally:
        monitor.stop()


def test_process_csv_respects_noncritical_idle_gate() -> None:
    monitor = SensorMissionMonitor(
        feed_url="https://example.invalid/sensor.csv",
        rpc_transport=_DummyRPC,
        rules=(
            SensorTriggerRule(
                name="warning_temperature",
                threshold=55.0,
                comparison="gte",
                severity="warning",
                variable="temperature",
                unit="C",
                column_contains="\u00b0C",
            ),
        ),
    )
    fake_agent = FakeAgent()
    monitor._agent_spec = fake_agent  # type: ignore[assignment]

    try:
        monitor._agent_is_idle = False
        csv_text = "Time,Zone A (\u00b0C)\n12:00,60\n"
        assert len(monitor._process_csv(csv_text)) == 0
        assert len(fake_agent.messages) == 0

        monitor._agent_is_idle = True
        assert len(monitor._process_csv(csv_text)) == 1
        assert len(fake_agent.messages) == 1
    finally:
        monitor.stop()
