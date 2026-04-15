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

from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property, reduce
import inspect
import operator
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from dimos.protocol.service.system_configurator.base import SystemConfigurator

from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, is_module_type
from dimos.core.module_coordinator import ModuleCoordinator
from dimos.core.stream import In, Out
from dimos.core.transport import LCMTransport, PubSubTransport, pLCMTransport
from dimos.spec.utils import Spec, is_spec, spec_annotation_compliance, spec_structural_compliance
from dimos.utils.generic import short_id
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass(frozen=True)
class StreamRef:
    name: str
    type: type
    direction: Literal["in", "out"]


@dataclass(frozen=True)
class ModuleRef:
    name: str
    spec: type[Spec] | type[Module]


@dataclass(frozen=True)
class _BlueprintAtom:
    module: type[Module]
    streams: tuple[StreamRef, ...]
    module_refs: tuple[ModuleRef, ...]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @classmethod
    def create(
        cls, module: type[Module], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> "_BlueprintAtom":
        streams: list[StreamRef] = []
        module_refs: list[ModuleRef] = []

        # Resolve annotations using namespaces from the full MRO chain so that
        # In/Out behind TYPE_CHECKING + `from __future__ import annotations` work.
        # Iterate reversed MRO so the most specific class's namespace wins when
        # parent modules shadow names (e.g. spec.perception.Image vs sensor_msgs.Image).
        globalns: dict[str, Any] = {}
        for c in reversed(module.__mro__):
            if c.__module__ in sys.modules:
                globalns.update(sys.modules[c.__module__].__dict__)
        try:
            # 모듈에 정의되어 있는 In/Out, Spec, ModuleRef를 가져옴.
            all_annotations = get_type_hints(module, globalns=globalns)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails.
            all_annotations = {}
            for base_class in reversed(module.__mro__):
                if hasattr(base_class, "__annotations__"):
                    all_annotations.update(base_class.__annotations__)

        # In/Out, Spec, ModuleRef를 순회하며 스트림 객체를 생성하여 streams에 추가.
        for name, annotation in all_annotations.items():
            origin = get_origin(annotation)
            # Streams
            if origin in (In, Out):
                direction = "in" if origin == In else "out"
                type_ = get_args(annotation)[0]
                streams.append(
                    StreamRef(name=name, type=type_, direction=direction)  # type: ignore[arg-type]
                )
            # linking to unknown module via Spec
            elif is_spec(annotation):
                module_refs.append(ModuleRef(name=name, spec=annotation))
            # linking to specific/known module directly
            elif is_module_type(annotation):
                module_refs.append(ModuleRef(name=name, spec=annotation))

        return cls(
            module=module,
            streams=tuple(streams),
            module_refs=tuple(module_refs),
            args=args,
            kwargs=kwargs,
        )


@dataclass(frozen=True)
class Blueprint:
    blueprints: tuple[_BlueprintAtom, ...]
    disabled_modules_tuple: tuple[type[Module], ...] = field(default_factory=tuple)
    transport_map: Mapping[tuple[str, type], PubSubTransport[Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    global_config_overrides: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    remapping_map: Mapping[tuple[type[Module], str], str | type[Module] | type[Spec]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    requirement_checks: tuple[Callable[[], str | None], ...] = field(default_factory=tuple)
    configurator_checks: "tuple[SystemConfigurator, ...]" = field(default_factory=tuple)

    @classmethod
    def create(cls, module: type[Module], *args: Any, **kwargs: Any) -> "Blueprint":
        blueprint = _BlueprintAtom.create(module, args, kwargs)
        return cls(blueprints=(blueprint,))

    def disabled_modules(self, *modules: type[Module]) -> "Blueprint":
        return replace(self, disabled_modules_tuple=self.disabled_modules_tuple + modules)

    def transports(self, transports: dict[tuple[str, type], Any]) -> "Blueprint":
        return replace(self, transport_map=MappingProxyType({**self.transport_map, **transports}))

    def global_config(self, **kwargs: Any) -> "Blueprint":
        return replace(
            self,
            global_config_overrides=MappingProxyType({**self.global_config_overrides, **kwargs}),
        )

    def remappings(
        self, remappings: list[tuple[type[Module], str, str | type[Module] | type[Spec]]]
    ) -> "Blueprint":
        remappings_dict = dict(self.remapping_map)
        for module, old, new in remappings:
            remappings_dict[(module, old)] = new
        return replace(self, remapping_map=MappingProxyType(remappings_dict))

    def requirements(self, *checks: Callable[[], str | None]) -> "Blueprint":
        return replace(self, requirement_checks=self.requirement_checks + tuple(checks))

    def configurators(self, *checks: "SystemConfigurator") -> "Blueprint":
        return replace(self, configurator_checks=self.configurator_checks + tuple(checks))

    @cached_property
    def _active_blueprints(self) -> tuple[_BlueprintAtom, ...]:
        if not self.disabled_modules_tuple:
            return self.blueprints
        disabled = set(self.disabled_modules_tuple)
        # 전체 모듈(self.blueprints) 중 disabled 리스트에 없는 것만 골라냄.
        return tuple(bp for bp in self.blueprints if bp.module not in disabled)

    def _check_ambiguity(
        self,
        requested_method_name: str,
        interface_methods: Mapping[str, list[tuple[type[Module], Callable[..., Any]]]],
        requesting_module: type[Module],
    ) -> None:
        if (
            requested_method_name in interface_methods
            and len(interface_methods[requested_method_name]) > 1
        ):
            modules_str = ", ".join(
                impl[0].__name__ for impl in interface_methods[requested_method_name]
            )
            raise ValueError(
                f"Ambiguous RPC method '{requested_method_name}' requested by "
                f"{requesting_module.__name__}. Multiple implementations found: "
                f"{modules_str}. Please use a concrete class name instead."
            )

    def _get_transport_for(self, name: str, stream_type: type) -> PubSubTransport[Any]:
        transport = self.transport_map.get((name, stream_type), None)
        if transport:
            return transport

        use_pickled = getattr(stream_type, "lcm_encode", None) is None
        topic = f"/{name}" if self._is_name_unique(name) else f"/{short_id()}"
        transport = pLCMTransport(topic) if use_pickled else LCMTransport(topic, stream_type)

        return transport

    @cached_property
    def _all_name_types(self) -> set[tuple[str, type]]:
        # Apply remappings to get the actual names that will be used
        result = set()
        for blueprint in self._active_blueprints:
            for conn in blueprint.streams:
                # Check if this stream should be remapped
                remapped_name = self.remapping_map.get((blueprint.module, conn.name), conn.name)
                if isinstance(remapped_name, str):
                    result.add((remapped_name, conn.type))
        return result

    def _is_name_unique(self, name: str) -> bool:
        return sum(1 for n, _ in self._all_name_types if n == name) == 1

    def _run_configurators(self) -> None:
        from dimos.protocol.service.system_configurator import configure_system, lcm_configurators

        configurators = [*lcm_configurators(), *self.configurator_checks]

        try:
            configure_system(configurators)
        except SystemExit:
            labels = [type(c).__name__ for c in configurators]
            print(
                f"Required system configuration was declined: {', '.join(labels)}",
                file=sys.stderr,
            )
            sys.exit(1)

    def _check_requirements(self) -> None:
        errors = []
        red = "\033[31m"
        reset = "\033[0m"

        for check in self.requirement_checks:
            error = check()
            if error:
                errors.append(error)

        if errors:
            for error in errors:
                print(f"{red}Error: {error}{reset}", file=sys.stderr)
            sys.exit(1)

    def _verify_no_name_conflicts(self) -> None:
        name_to_types = defaultdict(set)
        name_to_modules = defaultdict(list)

        for blueprint in self._active_blueprints:
            for conn in blueprint.streams:
                stream_name = self.remapping_map.get((blueprint.module, conn.name), conn.name)
                name_to_types[stream_name].add(conn.type)
                name_to_modules[stream_name].append((blueprint.module, conn.type))

        conflicts = {}
        for conn_name, types in name_to_types.items():
            if len(types) > 1:
                modules_by_type = defaultdict(list)
                for module, conn_type in name_to_modules[conn_name]:
                    modules_by_type[conn_type].append(module)
                conflicts[conn_name] = modules_by_type

        if not conflicts:
            return

        error_lines = ["Blueprint cannot start because there are conflicting streams."]
        for name, modules_by_type in conflicts.items():
            type_entries = []
            for conn_type, modules in modules_by_type.items():
                for module in modules:
                    type_str = f"{conn_type.__module__}.{conn_type.__name__}"
                    module_str = module.__name__
                    type_entries.append((type_str, module_str))
            if len(type_entries) >= 2:
                locations = ", ".join(f"{type_} in {module}" for type_, module in type_entries)
                error_lines.append(f"    - '{name}' has conflicting types. {locations}")

        raise ValueError("\n".join(error_lines))

    def _deploy_all_modules(
        self, module_coordinator: ModuleCoordinator, global_config: GlobalConfig
    ) -> None:
        module_specs: list[tuple[type[Module], tuple[Any, ...], dict[str, Any]]] = []
        for blueprint in self._active_blueprints:
            kwargs = {**blueprint.kwargs}
            sig = inspect.signature(blueprint.module.__init__)
            if "cfg" in sig.parameters:
                kwargs["cfg"] = global_config
            module_specs.append((blueprint.module, blueprint.args, kwargs))

        module_coordinator.deploy_parallel(module_specs)

    def _connect_streams(self, module_coordinator: ModuleCoordinator) -> None:
        # dict when given (final/remapped) stream name+type, provides a list of modules + original (non-remapped) stream names
        streams = defaultdict(list)

        for blueprint in self._active_blueprints:
            for conn in blueprint.streams:
                # Check if this stream should be remapped
                remapped_name = self.remapping_map.get((blueprint.module, conn.name), conn.name)
                if isinstance(remapped_name, str):
                    # Group by remapped name and type
                    streams[remapped_name, conn.type].append((blueprint.module, conn.name))

        # Connect all In/Out streams by remapped name and type.
        for remapped_name, stream_type in streams.keys():
            transport = self._get_transport_for(remapped_name, stream_type)
            for module, original_name in streams[(remapped_name, stream_type)]:
                instance = module_coordinator.get_instance(module)  # type: ignore[assignment]
                instance.set_transport(original_name, transport)  # type: ignore[union-attr]
                logger.info(
                    "Transport",
                    name=remapped_name,
                    original_name=original_name,
                    topic=str(getattr(transport, "topic", None)),
                    type=f"{stream_type.__module__}.{stream_type.__qualname__}",
                    module=module.__name__,
                    transport=transport.__class__.__name__,
                )

    def _connect_module_refs(self, module_coordinator: ModuleCoordinator) -> None:
        # partly fill out the mod_and_mod_ref_to_proxy 
        """
        remapping_map에 정보가 들어가는 과정은 사용자가 설계도(Blueprint)에서 .remappings() 메서드를 호출할 때 발생합니다.
        이때 전달하는 튜플(tuple)의 세 번째 인자가 무엇이냐에 따라 A와 B가 결정됩니다.
        # A: 스트림 이름 변경 (세 번째 인자가 문자열 'str')
        (CameraModule, "image", "/world/camera_image"),
        
        # B: 모듈 참조 변경 (세 번째 인자가 클래스 타입 'type')
        (AgentModule, "_navigator", AStarPlanner)
        """
        # 이 코드는 모듈 간의 전역적인 "자동 연결" 규칙을 사용자가 수동으로 덮어쓸(Override) 준비를 하는 과정입니다.
        mod_and_mod_ref_to_proxy = {
            (module, name): replacement
            for (module, name), replacement in self.remapping_map.items()
            if is_spec(replacement) or is_module_type(replacement)
        }

        # after this loop we should have an exact module for every module_ref on every blueprint
        for blueprint in self._active_blueprints:
            for each_module_ref in blueprint.module_refs:
                # we've got to find a another module that implements this spec
                # "찾아야 할 기준"을 설정. 수동 예약 명부에 있으면 그 클래스를 쓰고, 없으면 모듈 클래스에 정의된 **기본 인터페이스(Spec)**를 기준으로 삼음.
                spec = mod_and_mod_ref_to_proxy.get(
                    (blueprint.module, each_module_ref.name), each_module_ref.spec
                )

                # if the spec is actually module, use that (basically a user override)
                if is_module_type(spec):
                    mod_and_mod_ref_to_proxy[blueprint.module, each_module_ref.name] = spec
                    continue

                # find all available candidates
                possible_module_candidates = [
                    each_other_blueprint.module
                    for each_other_blueprint in self._active_blueprints
                    if (
                        each_other_blueprint != blueprint
                        # 이 모듈이 인터페이스(spec)에 적힌 함수들(이름)을 다 가지고 있고, 인자와 반환 타입이 일치하는지 확인
                        and spec_structural_compliance(each_other_blueprint.module, spec)
                    )
                ]
                # we keep valid separate from invalid to provide a better error message for "almost" valid cases
                valid_module_candidates = [
                    each_candidate
                    for each_candidate in possible_module_candidates
                    if spec_annotation_compliance(each_candidate, spec)
                ]
                # none (에러 발생! "이 인터페이스가 꼭 필요한데, 수행할 수 있는 모듈이 아무도 없어!" 라며 중단합니다.)
                if len(possible_module_candidates) == 0:
                    raise Exception(
                        f"""The {blueprint.module.__name__} has a module reference ({each_module_ref}) which requested a module that fills out the {each_module_ref.spec.__name__} spec. But I couldn't find a module that met that spec.\n"""
                    )
                # exactly one structurally valid candidate (성공! 해당 모듈을 파트너로 확정하고 장부에 기록합니다.)
                elif len(possible_module_candidates) == 1:
                    if len(valid_module_candidates) == 0:
                        logger.warning(
                            f"""The {blueprint.module.__name__} has a module reference ({each_module_ref}) which requested a module that fills out the {each_module_ref.spec.__name__} spec. I found a module ({possible_module_candidates[0].__name__}) that met that spec structurally, but it had a mismatch in type annotations.\nPlease either change the {each_module_ref.spec.__name__} spec or the {possible_module_candidates[0].__name__} module.\n"""
                        )
                    mod_and_mod_ref_to_proxy[blueprint.module, each_module_ref.name] = (
                        possible_module_candidates[0]
                    )
                    continue
                # more than one (모호성 에러 발생! "후보가 너무 많아서 내가 고를 수 없어! 설계도(remappings)에서 직접 하나를 골라줘!"라고 요청합니다.)
                elif len(valid_module_candidates) > 1:
                    raise Exception(
                        f"""The {blueprint.module.__name__} has a module reference ({each_module_ref}) which requested a module that fills out the {each_module_ref.spec.__name__} spec. But I found multiple modules that met that spec: {possible_module_candidates}.\nTo fix this use .remappings, for example:\n    autoconnect(...).remappings([ ({blueprint.module.__name__}, {each_module_ref.name!r}, <ModuleThatHasTheRpcCalls>) ])\n"""
                    )
                # structural candidates, but no valid candidates (경고! "구조는 맞는데 타입이 안 맞아!" 경고를 띄우고 넘어갑니다.)
                elif len(valid_module_candidates) == 0:
                    possible_module_candidates_str = ", ".join(
                        [each_candidate.__name__ for each_candidate in possible_module_candidates]
                    )
                    raise Exception(
                        f"""The {blueprint.module.__name__} has a module reference ({each_module_ref}) which requested a module that fills out the {each_module_ref.spec.__name__} spec. Some modules ({possible_module_candidates_str}) met the spec structurally but had a mismatch in type annotations\n"""
                    )
                # one valid candidate (and more than one structurally valid candidate)
                else:
                    mod_and_mod_ref_to_proxy[blueprint.module, each_module_ref.name] = (
                        valid_module_candidates[0]
                    )

        # now that we know the streams, we mutate the RPCClient objects
        for (base_module, module_ref_name), target_module in mod_and_mod_ref_to_proxy.items():
            base_module_proxy = module_coordinator.get_instance(base_module)
            target_module_proxy = module_coordinator.get_instance(target_module)  # type: ignore[type-var,arg-type]
            setattr(
                base_module_proxy,
                module_ref_name,
                target_module_proxy,
            )
            # Ensure the remote module instance can use the module ref inside its own RPC handlers.
            base_module_proxy.set_module_ref(module_ref_name, target_module_proxy)

    # 특정 모듈이 @rpc로 선언한 함수들을 전체 시스템의 "공용 함수 목록"에 등록합니다. 그리고 다른 모듈이 그 함수를 필요로 할 때(예: set_ 패턴 등), 해당 함수의 **실행 권한(Proxy)**을 넘겨줍니다.
    def _connect_rpc_methods(self, module_coordinator: ModuleCoordinator) -> None:
        # Gather all RPC methods.
        rpc_methods = {}
        rpc_methods_dot = {}

        # 구형 방식: Navigator_set_goal
        # 신형 방식: Navigator.set_goal
        # 현재 위 두 방식을 모두 쓰느건 과도기적인 패턴
        # 어떤 클래스에서 어떤 메서드를 쓰는지 구분하기 위해 네이밍을 설정해주는 로직
        # Track interface methods to detect ambiguity.
        interface_methods: defaultdict[str, list[tuple[type[Module], Callable[..., Any]]]] = (
            defaultdict(list)
        )  # interface_name_method -> [(module_class, method)]
        interface_methods_dot: defaultdict[str, list[tuple[type[Module], Callable[..., Any]]]] = (
            defaultdict(list)
        )  # interface_name.method -> [(module_class, method)]

        for blueprint in self._active_blueprints:
            for method_name in blueprint.module.rpcs.keys():  # type: ignore[attr-defined]
                module_proxy = module_coordinator.get_instance(blueprint.module)  # type: ignore[assignment]
                method_for_rpc_client = getattr(module_proxy, method_name)
                # Register under concrete class name (backward compatibility)
                rpc_methods[f"{blueprint.module.__name__}_{method_name}"] = method_for_rpc_client
                rpc_methods_dot[f"{blueprint.module.__name__}.{method_name}"] = (
                    method_for_rpc_client
                )

                # Also register under any interface names
                for base in blueprint.module.mro():
                    # Check if this base is an abstract interface with the method
                    if (
                        base is not Module
                        and issubclass(base, ABC)
                        and hasattr(base, method_name)
                        and getattr(base, method_name, None) is not None
                    ):
                        interface_key = f"{base.__name__}.{method_name}"
                        interface_methods_dot[interface_key].append(
                            (blueprint.module, method_for_rpc_client)
                        )
                        interface_key_underscore = f"{base.__name__}_{method_name}"
                        interface_methods[interface_key_underscore].append(
                            (blueprint.module, method_for_rpc_client)
                        )

        # Check for ambiguity in interface methods and add non-ambiguous ones
        for interface_key, implementations in interface_methods_dot.items():
            if len(implementations) == 1:
                rpc_methods_dot[interface_key] = implementations[0][1]
        for interface_key, implementations in interface_methods.items():
            if len(implementations) == 1:
                rpc_methods[interface_key] = implementations[0][1]

        # Fulfil method requests (so modules can call each other).
        for blueprint in self._active_blueprints:
            instance = module_coordinator.get_instance(blueprint.module)  # type: ignore[assignment]

            for method_name in blueprint.module.rpcs.keys():  # type: ignore[attr-defined]
                if not method_name.startswith("set_"):
                    continue

                linked_name = method_name.removeprefix("set_")

                # autoconnect로 연결한 모듈 중에서 동일한 인터페이스를 가진 모듈이 두개 이상 있으면 에러 발생
                self._check_ambiguity(linked_name, interface_methods, blueprint.module)

                if linked_name not in rpc_methods:
                    continue

                getattr(instance, method_name)(rpc_methods[linked_name])

            # 신형 방식 "_dot"에 맞게 다시 연결
            for requested_method_name in instance.get_rpc_method_names():  # type: ignore[union-attr]
                self._check_ambiguity(
                    requested_method_name, interface_methods_dot, blueprint.module
                )

                if requested_method_name not in rpc_methods_dot:
                    continue

                instance.set_rpc_method(  # type: ignore[union-attr]
                    requested_method_name, rpc_methods_dot[requested_method_name]
                )

    def build(
        self,
        cli_config_overrides: Mapping[str, Any] | None = None,
    ) -> ModuleCoordinator:
        logger.info("Building the blueprint")
        # cli 인자로 넘긴 값들이 override 됨.
        global_config.update(**dict(self.global_config_overrides))
        if cli_config_overrides:
            global_config.update(**dict(cli_config_overrides))

        self._run_configurators()
        self._check_requirements()
        self._verify_no_name_conflicts()

        logger.info("Starting the modules")
        module_coordinator = ModuleCoordinator(cfg=global_config)
        module_coordinator.start()

        # all module constructors are called here (each of them setup their own)
        # 모듈의 프로세스를 띄우는 객체를 생성 (모든 모듈의 생성자가 여기서 호출됨.)
        self._deploy_all_modules(module_coordinator, global_config)
        
        # 모듈 간의 데이터 통로(In, Out)를 연결. 
        # 스트림 연결 (In, Out)에 비약이 존재하는 경우, 오류가 발생하지는 않지만, 아무 동작도 하지 않는 상태가 됨.
        self._connect_streams(module_coordinator)
        
        # [Legacy] 개별 RPC 메서드 연결
        # dimos run으로 시스템이 시작되면, 각 모듈(예: 라이다 모듈, 제어 모듈)은 서로 **완전히 다른 프로세스(Workers)**에서 실행됩니다. 파이썬에서 다른 프로세스에 있는 객체의 함수를 직접 실행하는 것은 불가능하기 때문에, 프로세스 간에 **"메시지"**를 주고받아 함수를 실행시켜야 합니다. 이 기술이 바로 RPC입니다.
        # 참조/RPC 연결 (Spec, ModuleRef)의 경우에는 빌드 시점에 에러가 발생함. (line352)
        # 기존 모듈과의 하위 호환성을 위해 유지됩니다. (실행 시점에 연결 실패 확인됨)
        self._connect_rpc_methods(module_coordinator)

        # [Modern] 모듈 참조/스펙 연결 (권장 방식)
        # RPC 방식은 "A라는 함수를 연결해줘", "B라는 함수를 연결해줘"라고 일일이 요청해야 했습니다. 하지만 module_refs 방식은 **"이 역할을 수행하는 모듈 통째로 나한테 줘!"**라고 요청합니다
        # 모듈에 들어가는 메서드를 모두 사용 할 수 있도록 프록시 객체를 만드는 작업
        # AGENTS.md에 따라 새로운 코드에서는 이 방식을 우선적으로 사용해야 합니다.
        # 타입 안정성을 보장하며, 빌드 단계에서 연결 오류를 즉시 감지할 수 있습니다.
        self._connect_module_refs(module_coordinator)

        module_coordinator.start_all_modules()

        return module_coordinator


def autoconnect(*blueprints: Blueprint) -> Blueprint:
    all_blueprints = tuple(_eliminate_duplicates([bp for bs in blueprints for bp in bs.blueprints]))
    all_transports = dict(  # type: ignore[var-annotated]
        reduce(operator.iadd, [list(x.transport_map.items()) for x in blueprints], [])
    )
    all_config_overrides = dict(  # type: ignore[var-annotated]
        reduce(operator.iadd, [list(x.global_config_overrides.items()) for x in blueprints], [])
    )
    all_remappings = dict(  # type: ignore[var-annotated]
        reduce(operator.iadd, [list(x.remapping_map.items()) for x in blueprints], [])
    )
    all_requirement_checks = tuple(check for bs in blueprints for check in bs.requirement_checks)
    all_configurator_checks = tuple(check for bs in blueprints for check in bs.configurator_checks)

    return Blueprint(
        blueprints=all_blueprints,
        disabled_modules_tuple=tuple(
            module for bp in blueprints for module in bp.disabled_modules_tuple
        ),
        transport_map=MappingProxyType(all_transports),
        global_config_overrides=MappingProxyType(all_config_overrides),
        remapping_map=MappingProxyType(all_remappings),
        requirement_checks=all_requirement_checks,
        configurator_checks=all_configurator_checks,
    )


def _eliminate_duplicates(blueprints: list[_BlueprintAtom]) -> list[_BlueprintAtom]:
    # The duplicates are eliminated in reverse so that newer blueprints override older ones.
    seen = set()
    unique_blueprints = []
    for bp in reversed(blueprints):
        if bp.module not in seen:
            seen.add(bp.module)
            unique_blueprints.append(bp)
    return list(reversed(unique_blueprints))
