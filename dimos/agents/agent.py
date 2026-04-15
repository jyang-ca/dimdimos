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

from dataclasses import dataclass
import json
from queue import Empty, Queue
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from reactivex.disposable import Disposable

from dimos.agents.system_prompt import SYSTEM_PROMPT
from dimos.agents.utils import pretty_print_langchain_message
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig, SkillInfo
from dimos.core.rpc_client import RpcCall, RPCClient
from dimos.core.stream import In, Out
from dimos.protocol.rpc import RPCSpec
from dimos.spec.utils import Spec

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass
class AgentConfig(ModuleConfig):
    system_prompt: str | None = SYSTEM_PROMPT
    model: str = "gpt-4o"
    model_fixture: str | None = None

"""
준비 단계 (__init__): "너는 나중에 이 일을 해야 해"라고 일꾼(Thread)에게 **업무 매뉴얼 (_thread_loop)**만 쥐여준 상태입니다. 아직 출근은 하지 않은 상태죠.
도구 파악 (on_system_modules 시작): "내 몸(로봇)에 어떤 스킬들이 있나?"를 먼저 훑어봅니다.
두뇌 생성 (create_agent): 수집한 정보를 바탕으로 에이전트의 **지능(엔진)**을 구축합니다.
근무 시작 (_thread.start()): 이제 모든 준비(메뉴얼 + 도구 + 지능)가 끝났으니, 비로소 일꾼을 출근시켜 본격적인 명령 대기 루프를 돌리는 것입니다.
"""

class Agent(Module[AgentConfig]):
    default_config = AgentConfig
    agent: Out[BaseMessage]
    human_input: In[str]
    agent_idle: Out[bool]

    _lock: RLock
    _state_graph: CompiledStateGraph[Any, Any, Any, Any] | None
    _message_queue: Queue[BaseMessage]
    _history: list[BaseMessage]
    _thread: Thread
    _stop_event: Event

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = RLock()
        self._state_graph = None
        self._message_queue = Queue()
        self._history = []
        self._thread = Thread(
            target=self._thread_loop, # 실행할 대상 함수 지정
            name=f"{self.__class__.__name__}-thread",
            daemon=True, # 프로세스 종료 시 함께 종료되도록 설정
        )
        self._stop_event = Event()

    @rpc
    def start(self) -> None:
        super().start()

        # human_input 채널을 구독하여 메시지를 받으면 HumanMessage 큐에 넣습니다. (mcp_server.py -> agnet.py)
        def _on_human_input(string: str) -> None:
            self._message_queue.put(HumanMessage(content=string))

        # 사용자의 입력을 실시간으로 모니터링하기 시작하되, 나중에 로봇 시스템(모듈)이 꺼질 때 잊지 말고 관련 리소스를 깔끔하게 정리해라
        self._disposables.add(Disposable(self.human_input.subscribe(_on_human_input)))

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().stop()

    @rpc
    def on_system_modules(self, modules: list[RPCClient]) -> None:
        # module_coordinator.py에 있는 def start_all_modules에서 위 함수 호출
        assert self.rpc is not None

        if self.config.model.startswith("ollama:"):
            from dimos.agents.ollama_agent import ensure_ollama_model

            ensure_ollama_model(self.config.model.removeprefix("ollama:"))

        model: str | BaseChatModel = self.config.model
        if self.config.model_fixture is not None:
            from dimos.agents.testing import MockModel

            model = MockModel(json_path=self.config.model_fixture)

        with self._lock:
            # Here to prevent unwanted imports in the file.
            from langchain.agents import create_agent

            # agent엔진 구축
            self._state_graph = create_agent(
                model=model,
                tools=_get_tools_from_modules(self, modules, self.rpc), # 로봇의 스킬 명세 전달. 여기에 wrapped_func가 들어있음
                system_prompt=self.config.system_prompt,
            )
            # _thread_loop라는 별도의 스레드가 돌아가기 시작합니다.
            # 이제부터 사용자가 어떤 명령을 내리면 큐(_message_queue)에서 쏙 꺼내서, 방금 만든 뇌(_state_graph)를 돌려 판단을 내릴 준비가 완료된 것입니다.
            self._thread.start()

    @rpc
    def add_message(self, message: BaseMessage) -> None:
        self._message_queue.put(message)

    def _thread_loop(self) -> None:
        # stop_event가 설정되지 않은 동안 계속 반복합니다.
        while not self._stop_event.is_set():
            try:
                # 큐에서 메시지를 하나 꺼냅니다. (없으면 기다림)
                message = self._message_queue.get(timeout=0.5)
            except Empty:
                continue
            
            # 메시지가 있다면 '처리' 로직으로 넘깁니다.
            with self._lock:
                if not self._state_graph:
                    raise ValueError("No state graph initialized")
                self._process_message(self._state_graph, message)

    def _process_message(
        self, state_graph: CompiledStateGraph[Any, Any, Any, Any], message: BaseMessage
    ) -> None:
        # agent_idle 채널에 False 값을 발송.
        # 시각화 도구(GUI)에서 에이전트의 상태가 "작동 중"으로 전환
        # 다른 제어 모듈이 이 신호를 감시하고 있다가, 에이전트가 생각 중일 때 발생할 수 있는 충돌을 방지하기 위해 대기 모드로 진입 가능.
        self.agent_idle.publish(False)
        # 대화 내역에 저장
        self._history.append(message)
        pretty_print_langchain_message(message)
        self.agent.publish(message)

        # LangGraph를 통해 LLM에게 메시지와 대화 내역을 전달하고 응답을 스트리밍합니다. 에이전트는 내가 어떤 스킬을 가지고 있는지에 대한 설명도 같이 전달 받음.
        for update in state_graph.stream({"messages": self._history}, stream_mode="updates"):
            """
            모델은 전달받은 상황을 보고 두 가지 중 하나를 결정합니다.
            유형 A (단순 답변): "알겠어, 앞으로 갈게." 같은 일반 텍스트 응답.
            유형 B (도구 호출): "내가 직접 하는 것보다 move 스킬을 쓰는 게 좋겠어. x=2.0으로 호출해줘."
            
            1. 모델이 **스킬 호출(Tool Call)**을 결정하면, _skill_to_tool에 정의된 함수가 실행됩니다.
            2. 이 함수는 RPC를 통해 실제 로봇의 하드웨어를 움직입니다.
            3. 스킬의 실행 결과(성공/실패/데이터)가 다시 모델에게 전달됩니다.
            """
            for node_output in update.values():
                for msg in node_output.get("messages", []):
                    self._history.append(msg)
                    pretty_print_langchain_message(msg)
                    # 도구를 실행하는 코드가 아니고, UI에 표시하기 위한 코드
                    self.agent.publish(msg)

        if self._message_queue.empty():
            self.agent_idle.publish(True)


class AgentSpec(Spec, Protocol):
    def add_message(self, message: BaseMessage) -> None: ...

# 여기서 로봇의 모든 스킬 명세서(이름, 설명, 파라미터)가 JSON 형태로 변환되어 모델에게 전달됨.
def _get_tools_from_modules(
    agent: Agent, modules: list[RPCClient], rpc: RPCSpec
) -> list[StructuredTool]:
    skills = [skill for module in modules for skill in (module.get_skills() or [])]
    return [_skill_to_tool(agent, skill, rpc) for skill in skills]


def _skill_to_tool(agent: Agent, skill: SkillInfo, rpc: RPCSpec) -> StructuredTool:
    rpc_call = RpcCall(None, rpc, skill.func_name, skill.class_name, [])

    # create_agent 생성자 tools인자에 등록된 스킬을 호출하는 함수
    def wrapped_func(*args: Any, **kwargs: Any) -> str | list[dict[str, Any]]:
        result = None

        try:
            result = rpc_call(*args, **kwargs)
        except Exception as e:
            return f"Exception: Error: {e}"

        if result is None:
            return "It has started. You will be updated later."

        if hasattr(result, "agent_encode"):
            uuid_ = str(uuid.uuid4())
            _append_image_to_history(agent, skill, uuid_, result)
            return f"Tool call started with UUID: {uuid_}"

        return str(result)

    return StructuredTool(
        name=skill.func_name,
        func=wrapped_func,
        args_schema=json.loads(skill.args_schema),
    )


def _append_image_to_history(agent: Agent, skill: SkillInfo, uuid_: str, result: Any) -> None:
    agent.add_message(
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"This is the artefact for the '{skill.func_name}' tool with UUID:={uuid_}.",
                },
                *result.agent_encode(),
            ]
        )
    )


agent = Agent.blueprint

__all__ = ["Agent", "AgentSpec", "agent"]
