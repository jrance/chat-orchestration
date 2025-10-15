from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import StateGraph

from ...config.schema import AgentNode, GraphConfig, ToolNode, ToolsConfig
from ...providers.base import LLMProvider
from ...providers.types import ChatMessage, ChatRequest
from ...tools.execution import execute_tool_call
from ...tools.registry import ToolEntry, ToolRegistry
from ..tool_bindings import resolve_attached_tools
from ..types import ToolAttachmentBinding
from ..errors import GraphCompileError, ToolExecutionError
from ..state import EngineState
from ...runtime.history import assemble_prompt
from ...runtime.filters import pre_provider_filter, post_provider_filter
from ...runtime.policies import get_policy
from ...runtime.safety import redact_text
from ...runtime.structured_output import ensure_structured_output, build_system_hint

# Optional telemetry helper may not exist; guarded usage.


def _build_initial_system_message(agent: AgentNode) -> ChatMessage | None:
    sys = agent.data.system_instructions or ""
    guide = agent.data.style_guide or ""
    text_parts: list[str] = []
    if sys:
        text_parts.append(sys)
    if guide:
        text_parts.append(guide)
    if not text_parts:
        return None
    return ChatMessage(role="system", content="\n\n".join(text_parts))


@dataclass(slots=True)
class AgentArtifacts:
    node: AgentNode
    provider: LLMProvider
    tool_cfg: ToolsConfig
    tool_bindings: dict[str, ToolEntry]
    provider_tool_specs: list[Any]
    overrides_by_name: dict[str, dict[str, Any]]
    bindings_by_fn: dict[str, ToolAttachmentBinding]

    def agent_fn(self, state: EngineState) -> EngineState:
        state["agent_cursor"] = self.node.id
        mp = self.node.data.model
        meta = state.get("metadata", {}) or {}
        telemetry = meta.get("_telemetry") if isinstance(meta, dict) else None
        try:
            if telemetry is not None and hasattr(telemetry, "set_provider_model"):
                telemetry.set_provider_model(getattr(mp, "provider", None), getattr(mp, "model_id", None))
        except Exception:
            pass
        preamble = meta.get("org_preamble") if self.node.data.context.inject_org_preamble else None
        vars_req = meta.get("vars") or {}
        req_messages = assemble_prompt(self.node, state.get("messages", []), preamble, vars_req)
        safety_cfg = self.node.data.safety
        policy = get_policy(getattr(safety_cfg, "policy_ref", None))
        pre = pre_provider_filter(req_messages, safety_cfg=safety_cfg, policy=policy)
        messages = state.setdefault("messages", [])
        if pre.blocked:
            assistant = post_provider_filter(pre.refusal_message or ChatMessage(role="assistant", content=""), safety_cfg=safety_cfg, policy=policy)
            messages.append(assistant)
            state["tool_calls"] = None
            state["steps"] = int(state.get("steps", 0)) + 1
            meta2 = dict(meta)
            meta2["_last_finish_reason"] = "content_filter"
            if pre.warnings:
                meta2["warnings"] = pre.warnings
            state["metadata"] = meta2
            return state

        req = ChatRequest(
            model=mp.model_id,
            messages=pre.redacted_messages,
            tools=self.provider_tool_specs or None,
            tool_choice=(
                "auto"
                if self.tool_cfg.policy == "Auto" or self.tool_cfg.policy == "Required"
                else "none"
            ),
            temperature=mp.temperature,
            top_p=mp.top_p,
            max_tokens=mp.max_tokens,
            stop=mp.stop or None,
            seed=mp.seed,
            json_mode_enabled=mp.json_mode_enabled,
            response_format_schema=None,
            metadata={
                **(state.get("metadata", {}) or {}),
                "agent_id": self.node.id,
                "agent_label": self.node.label,
            },
            timeout=None,
        )
        so = self.node.data.structured_output
        t0 = None
        telemetry = meta.get("_telemetry") if isinstance(meta, dict) else None
        try:
            if telemetry is not None and hasattr(telemetry, "start_timer"):
                t0 = telemetry.start_timer("turn")
        except Exception:
            t0 = None
        assistant: ChatMessage | None = None
        try:
            if getattr(so, "enabled", False):
                hint = build_system_hint(getattr(so, "schema_", None) or None)
                assistant, usage, finish_reason, attempts = ensure_structured_output(
                    self.provider,
                    req,
                    policy_cfg=so,
                    post_cfg=getattr(so, "post_process", None),
                    system_hint=hint,
                )
                assistant = post_provider_filter(assistant, safety_cfg=safety_cfg, policy=policy)
                messages.append(assistant)
                state["tool_calls"] = assistant.tool_calls or None
                state["steps"] = int(state.get("steps", 0)) + int(attempts)
                meta2 = dict(state.get("metadata", {}) or {})
                meta2["_last_finish_reason"] = finish_reason
                meta2["_last_usage"] = usage
                state["metadata"] = meta2
            else:
                resp = self.provider.chat(req)
                assistant = post_provider_filter(resp.message, safety_cfg=safety_cfg, policy=policy)
                messages.append(assistant)
                state["tool_calls"] = assistant.tool_calls or None
                state["steps"] = int(state.get("steps", 0)) + 1
                meta2 = dict(state.get("metadata", {}) or {})
                meta2["_last_finish_reason"] = resp.finish_reason
                meta2["_last_usage"] = resp.usage
                state["metadata"] = meta2
        finally:
            try:
                if telemetry is not None and t0 is not None and hasattr(telemetry, "stop_timer"):
                    duration = telemetry.stop_timer(t0)
                else:
                    duration = 0.0
                had_tools = bool((state.get("tool_calls") or []))
                if telemetry is not None and hasattr(telemetry, "record_turn"):
                    telemetry.record_turn(
                        duration_ms=duration,
                        finish_reason=(state.get("metadata", {}) or {}).get("_last_finish_reason"),
                        had_tool_calls=had_tools,
                    )
            except Exception:
                pass
        if assistant is not None and not (assistant.tool_calls or []):
            bucket = state.get("group")
            if isinstance(bucket, dict):
                bucket["turn"] = int(bucket.get("turn", 0) or 0) + 1
                state["group"] = bucket
        return state

    def tools_fn(self, state: EngineState) -> EngineState:
        tool_calls = list(state.get("tool_calls") or [])
        if not tool_calls:
            return state

        max_calls = self.tool_cfg.max_calls_per_turn or 0
        limit = len(tool_calls) if max_calls <= 0 else min(max_calls, len(tool_calls))
        timeout_s = (self.tool_cfg.timeout_ms or 0) / 1000.0 if self.tool_cfg.timeout_ms else None
        telemetry = (state.get("metadata", {}) or {}).get("_telemetry")
        messages = state.setdefault("messages", [])

        for tc in tool_calls[:limit]:
            tool_name = tc.name
            if tool_name not in self.tool_bindings:
                raise ToolExecutionError(tool_name=tool_name, reason="Tool not attached")
            entry = self.tool_bindings[tool_name]
            overrides = self.overrides_by_name.get(tool_name) or {}
            result_obj = execute_tool_call(
                entry.impl,
                tc.arguments_json or "",
                parameter_overrides=overrides,
                timeout_s=timeout_s,
                redact_for_logging=bool(getattr(self.node.data.safety, "pii_redaction", True)),
                telemetry=telemetry,
            )
            content_json = json.dumps(result_obj)
            if getattr(self.node.data.safety, "pii_redaction", True):
                content_json = redact_text(content_json)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=content_json,
                    tool_call_id=tc.id,
                )
            )

        state["tool_calls"] = None
        return state


class CompileContext:
    """Shared utilities for region builders during compilation."""

    def __init__(
        self,
        *,
        cfg: GraphConfig,
        provider_resolver: Callable[[Any], LLMProvider],
        registry: ToolRegistry,
    ) -> None:
        self.cfg = cfg
        self.provider_resolver = provider_resolver
        self.registry = registry
        self._agents: dict[str, AgentNode] = {
            n.id: n for n in cfg.nodes if isinstance(n, AgentNode)
        }
        self._agent_artifacts: dict[str, AgentArtifacts] = {}
        self._passthrough_nodes: set[str] = set()

    # -------- Agent helpers --------
    def get_agent(self, agent_id: str) -> AgentNode:
        if agent_id not in self._agents:
            raise GraphCompileError(f"Agent not found: {agent_id}")
        return self._agents[agent_id]

    def agent_artifacts(self, agent_id: str) -> AgentArtifacts:
        if agent_id in self._agent_artifacts:
            return self._agent_artifacts[agent_id]
        node = self.get_agent(agent_id)
        provider = self.provider_resolver(node.data.model)
        tool_cfg = node.data.tools
        # Resolve attachments with unique function names and provider specs
        bindings = resolve_attached_tools(node, self.cfg, self.registry)
        provider_tool_specs = [b.tool_spec for b in bindings]
        tool_bindings: dict[str, ToolEntry] = {b.function_name: b.entry for b in bindings}
        overrides: dict[str, dict[str, Any]] = {b.function_name: dict(b.overrides or {}) for b in bindings}
        bindings_by_fn: dict[str, ToolAttachmentBinding] = {b.function_name: b for b in bindings}
        artifacts = AgentArtifacts(
            node=node,
            provider=provider,
            tool_cfg=tool_cfg,
            tool_bindings=tool_bindings,
            provider_tool_specs=provider_tool_specs,
            overrides_by_name=overrides,
            bindings_by_fn=bindings_by_fn,
        )
        self._agent_artifacts[agent_id] = artifacts
        return artifacts

    # -------- Naming helpers --------
    @staticmethod
    def region_entry_key(region_id: str) -> str:
        return f"region__{region_id}__entry"

    @staticmethod
    def region_exit_key(region_id: str) -> str:
        return f"region__{region_id}__exit"

    @staticmethod
    def agent_node_key(region_id: str, agent_id: str) -> str:
        return f"region__{region_id}__agent__{agent_id}"

    @staticmethod
    def tools_node_key(region_id: str, agent_id: str) -> str:
        return f"region__{region_id}__tools__{agent_id}"

    def ensure_passthrough_node(self, graph: StateGraph, key: str) -> None:
        if key in self._passthrough_nodes:
            return
        graph.add_node(key, lambda state: state)
        self._passthrough_nodes.add(key)


__all__ = ["CompileContext", "AgentArtifacts"]
