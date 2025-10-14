from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..config.schema import GraphConfig
from ..providers.types import ChatChunk, ChatMessage, ChatRequest
from ..tools.registry import ToolRegistry, default_registry
from .compiler import compile_graph
from .state import EngineState
from .types import CompiledSingleAgent, EngineRunInput, EngineRunResult, ProviderResolver
from ..runtime.templates import render
from ..runtime.structured_output import ensure_structured_output, build_system_hint
from ..runtime.safety import redact_text
from ..runtime.filters import pre_provider_filter
from ..runtime.policies import get_policy


class OrchestrationEngine:
    def compile(
        self,
        cfg: GraphConfig,
        *,
        provider_resolver: ProviderResolver | None = None,
        registry: ToolRegistry | None = None,
    ) -> CompiledSingleAgent:
        return compile_graph(
            cfg, provider_resolver=provider_resolver, registry=registry or default_registry
        )

    # ------------------- Non-stream execution -------------------
    def execute(
        self, compiled: CompiledSingleAgent, run_input: EngineRunInput | dict[str, Any]
    ) -> EngineRunResult:
        # Coerce inputs
        if not isinstance(run_input, EngineRunInput):
            run_input = EngineRunInput.model_validate(run_input)
        # Normalize run input messages
        if isinstance(run_input.messages, str):
            # Optionally template user input with request-level vars (not past messages)
            text = render(run_input.messages, run_input.vars or {})
            initial_messages = [ChatMessage(role="user", content=text)]
        else:
            initial_messages = list(run_input.messages)

        state: EngineState = {
            "messages": initial_messages,
            "metadata": dict(run_input.metadata or {}),
            "tool_calls": None,
            "steps": 0,
            "agent_cursor": None,
            "concurrent": None,
            "group": None,
        }
        # Plumb request vars for runtime prompt assembly
        if run_input.vars:
            meta = state.get("metadata", {})
            meta["vars"] = dict(run_input.vars)
            state["metadata"] = meta

        # Invoke graph
        app = compiled.graph
        final_state: EngineState = app.invoke(state)

        # Compute finish info
        meta = final_state.get("metadata", {}) or {}
        finish_reason = meta.get("_last_finish_reason", None) or "stop"
        usage = meta.get("_last_usage", None)
        return EngineRunResult(
            messages=final_state["messages"],
            finish_reason=finish_reason,
            usage=usage,
            steps=final_state.get("steps", 0),
        )

    # ------------------- Basic streaming -------------------
    def stream_execute(
        self, compiled: CompiledSingleAgent, run_input: EngineRunInput | dict[str, Any]
    ) -> Iterator[ChatChunk]:
        if not isinstance(run_input, EngineRunInput):
            run_input = EngineRunInput.model_validate(run_input)
        # Basic implementation: stream a single assistant turn when tools are disabled
        tools_policy = getattr(compiled.tool_config, "policy", "Auto")
        has_tools = bool(compiled.tools_attached)

        if tools_policy != "None" and has_tools:
            # For now, only stream turns without tool calls; full interleaving in later PR
            # Fall back to non-stream path by performing a single provider stream for the next turn
            yield from self._stream_single_turn(compiled, run_input)
            return

        # Streaming when no tools are available or policy is None
        yield from self._stream_single_turn(compiled, run_input)

    # Internal helper
    def _stream_single_turn(
        self, compiled: CompiledSingleAgent, run_input: EngineRunInput
    ) -> Iterator[ChatChunk]:
        provider = compiled.provider
        mp = compiled.model_params
        # Telemetry (if attached via metadata)
        telemetry = None
        try:
            telemetry = (run_input.metadata or {}).get("_telemetry")
            if telemetry is not None and hasattr(telemetry, "set_provider_model"):
                telemetry.set_provider_model(getattr(mp, "provider", None), getattr(mp, "model_id", None))
        except Exception:
            telemetry = None

        if isinstance(run_input.messages, str):
            text = render(run_input.messages, run_input.vars or {})
            messages = [ChatMessage(role="user", content=text)]
        else:
            messages = list(run_input.messages)

        # Safety pre-filter for streaming path (block or redact)
        safety_cfg = getattr(compiled, "safety", None)
        if safety_cfg is not None:
            policy = get_policy(getattr(safety_cfg, "policy_ref", None))
            pre = pre_provider_filter(messages, safety_cfg=safety_cfg, policy=policy)
            if pre.blocked:
                refusal = pre.refusal_message or ChatMessage(role="assistant", content="")
                yield ChatChunk(type="role", role="assistant")
                txt = refusal.content if isinstance(refusal.content, str) else ""
                if getattr(safety_cfg, "pii_redaction", True):
                    txt = redact_text(txt)
                yield ChatChunk(type="content.delta", text_delta=txt)
                yield ChatChunk(type="message.end", finish_reason="content_filter", usage=None)
                return
            messages = pre.redacted_messages

        # Build request (no tools for basic streaming path, or included if none attached)
        req = ChatRequest(
            model=mp.model_id,
            messages=messages,
            tools=None,
            tool_choice="none",
            temperature=mp.temperature,
            top_p=mp.top_p,
            max_tokens=mp.max_tokens,
            stop=mp.stop or None,
            seed=mp.seed,
            json_mode_enabled=mp.json_mode_enabled,
            response_format_schema=None,
            metadata=run_input.metadata or None,
            timeout=None,
        )
        # If structured output is enabled on root agent, do a non-stream call and then emit SSE
        so = getattr(compiled, "structured_output", None)
        if so is not None and getattr(so, "enabled", False):
            hint = build_system_hint(getattr(so, "schema_", None) or None)
            t0 = None
            try:
                if telemetry is not None and hasattr(telemetry, "start_timer"):
                    t0 = telemetry.start_timer("turn")
            except Exception:
                t0 = None
            assistant, usage, finish_reason, _attempts = ensure_structured_output(
                provider, req, policy_cfg=so, post_cfg=getattr(so, "post_process", None), system_hint=hint
            )
            # Emit a simple SSE sequence
            yield ChatChunk(type="role", role="assistant")
            text = assistant.content if isinstance(assistant.content, str) else ""
            if text:
                yield ChatChunk(type="content.delta", text_delta=text)
            yield ChatChunk(type="message.end", finish_reason=finish_reason, usage=usage)
            # Telemetry turn
            try:
                if telemetry is not None:
                    if usage is not None and hasattr(telemetry, "usage"):
                        telemetry.usage(
                            getattr(usage, "prompt_tokens", None),
                            getattr(usage, "completion_tokens", None),
                            getattr(usage, "total_tokens", None),
                        )
                    dur_ms = telemetry.stop_timer(t0) if t0 is not None else 0.0
                    had_tools = bool(getattr(assistant, "tool_calls", None))
                    if hasattr(telemetry, "record_turn"):
                        telemetry.record_turn(
                            duration_ms=dur_ms, finish_reason=finish_reason, had_tool_calls=had_tools
                        )
            except Exception:
                pass
            return

        # Otherwise, stream normally
        assembled_text: list[str] = []
        role_seen: str | None = None
        pii = False
        try:
            safety = getattr(compiled, "safety", None)
            if safety is not None and getattr(safety, "pii_redaction", True):
                pii = True
        except Exception:
            pii = False

        t0 = None
        try:
            if telemetry is not None and hasattr(telemetry, "start_timer"):
                t0 = telemetry.start_timer("turn")
        except Exception:
            t0 = None
        finish_reason_seen = None
        for chunk in provider.stream(req):
            if chunk.type == "role":
                role_seen = chunk.role or role_seen
                yield chunk
            elif chunk.type == "content.delta" and (chunk.text_delta or ""):
                text = chunk.text_delta or ""
                if pii:
                    text = redact_text(text)
                assembled_text.append(text)
                yield ChatChunk(type="content.delta", text_delta=text)
            else:
                yield chunk
            if chunk.type == "message.end":
                finish_reason_seen = chunk.finish_reason
        # If we saw end, record telemetry
        try:
            if telemetry is not None:
                dur_ms = telemetry.stop_timer(t0) if t0 is not None else 0.0
                if hasattr(telemetry, "record_turn"):
                    telemetry.record_turn(
                        duration_ms=dur_ms, finish_reason=finish_reason_seen, had_tool_calls=False
                    )
        except Exception:
            pass


__all__ = ["OrchestrationEngine"]
