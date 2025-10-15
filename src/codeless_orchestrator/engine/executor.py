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

    # ------------------- Streaming with auto tool calls -------------------
    def stream_execute(
        self, compiled: CompiledSingleAgent, run_input: EngineRunInput | dict[str, Any]
    ) -> Iterator[ChatChunk]:
        """Stream assistant output, auto-invoking tools when requested by the model.

        - Emits provider token deltas as-is.
        - When tool calls are requested (finish_reason == 'tool_calls' or any tool_call.* seen),
          executes attached tools, appends tool results to the transcript, then continues
          streaming the next assistant turn.
        """
        if not isinstance(run_input, EngineRunInput):
            run_input = EngineRunInput.model_validate(run_input)

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

        # Normalize run input messages
        if isinstance(run_input.messages, str):
            text = render(run_input.messages, run_input.vars or {})
            messages = [ChatMessage(role="user", content=text)]
        else:
            messages = list(run_input.messages)

        # Safety config
        safety_cfg = getattr(compiled, "safety", None)
        pii_redact = True
        try:
            pii_redact = bool(getattr(safety_cfg, "pii_redaction", True))
        except Exception:
            pii_redact = True

        # Prepare provider tool specs if tools are attached
        provider_tools = None
        try:
            if compiled.tool_bindings_by_fn:
                provider_tools = [b.tool_spec for b in compiled.tool_bindings_by_fn.values()]
            elif compiled.tools_attached:
                provider_tools = [entry.to_tool_spec() for entry in compiled.tools_attached.values()]
        except Exception:
            provider_tools = None

        # If structured output is enabled on root agent, do a non-stream call and then emit SSE
        so = getattr(compiled, "structured_output", None)
        if so is not None and getattr(so, "enabled", False):
            hint = build_system_hint(getattr(so, "schema_", None) or None)
            t0_so = None
            try:
                if telemetry is not None and hasattr(telemetry, "start_timer"):
                    t0_so = telemetry.start_timer("turn")
            except Exception:
                t0_so = None
            assistant, usage, finish_reason, _attempts = ensure_structured_output(
                provider,
                ChatRequest(
                    model=mp.model_id,
                    messages=messages,
                    tools=provider_tools or None,
                    tool_choice=(
                        "auto"
                        if (provider_tools and getattr(compiled.tool_config, "policy", "Auto") in ("Auto", "Required"))
                        else "none"
                    ),
                    temperature=mp.temperature,
                    top_p=mp.top_p,
                    max_tokens=mp.max_tokens,
                    stop=mp.stop or None,
                    seed=mp.seed,
                    json_mode_enabled=mp.json_mode_enabled,
                    response_format_schema=None,
                    metadata=run_input.metadata or None,
                    timeout=None,
                ),
                policy_cfg=so,
                post_cfg=getattr(so, "post_process", None),
                system_hint=hint,
            )
            yield ChatChunk(type="role", role="assistant")
            text_so = assistant.content if isinstance(assistant.content, str) else ""
            if text_so:
                yield ChatChunk(type="content.delta", text_delta=(redact_text(text_so) if pii_redact else text_so))
            yield ChatChunk(type="message.end", finish_reason=finish_reason, usage=usage)
            try:
                if telemetry is not None:
                    if usage is not None and hasattr(telemetry, "usage"):
                        telemetry.usage(
                            getattr(usage, "prompt_tokens", None),
                            getattr(usage, "completion_tokens", None),
                            getattr(usage, "total_tokens", None),
                        )
                    dur_ms = telemetry.stop_timer(t0_so) if t0_so is not None else 0.0
                    had_tools = bool(getattr(assistant, "tool_calls", None))
                    if hasattr(telemetry, "record_turn"):
                        telemetry.record_turn(
                            duration_ms=dur_ms, finish_reason=finish_reason, had_tool_calls=had_tools
                        )
            except Exception:
                pass
            return

        # Execution loop: assistant -> (optional tools) -> assistant ...
        # Avoid infinite loops by bounding the number of tool-assisted turns
        max_tool_rounds = 5
        rounds = 0
        while True:
            rounds += 1
            if rounds > max_tool_rounds:
                # Fail-safe: stop streaming if too many tool rounds requested
                break

            # Safety pre-filter per turn (block or redact)
            pre_messages = messages
            if safety_cfg is not None:
                policy = get_policy(getattr(safety_cfg, "policy_ref", None))
                pre = pre_provider_filter(pre_messages, safety_cfg=safety_cfg, policy=policy)
                if pre.blocked:
                    refusal = pre.refusal_message or ChatMessage(role="assistant", content="")
                    yield ChatChunk(type="role", role="assistant")
                    txt = refusal.content if isinstance(refusal.content, str) else ""
                    if pii_redact:
                        txt = redact_text(txt)
                    yield ChatChunk(type="content.delta", text_delta=txt)
                    yield ChatChunk(type="message.end", finish_reason="content_filter", usage=None)
                    return
                pre_messages = pre.redacted_messages

            # Build request
            req = ChatRequest(
                model=mp.model_id,
                messages=pre_messages,
                tools=provider_tools or None,
                tool_choice=(
                    "auto"
                    if (provider_tools and getattr(compiled.tool_config, "policy", "Auto") in ("Auto", "Required"))
                    else "none"
                ),
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

            # Stream this assistant turn, capturing tool call arguments
            assembled_text: list[str] = []
            tool_args_by_index: dict[int, dict[str, Any]] = {}
            saw_tool_signal = False
            finish_reason_seen = None

            t0 = None
            try:
                if telemetry is not None and hasattr(telemetry, "start_timer"):
                    t0 = telemetry.start_timer("turn")
            except Exception:
                t0 = None

            for chunk in provider.stream(req):
                # Forward role and normal content tokens
                if chunk.type == "role":
                    yield chunk
                    continue
                if chunk.type == "content.delta" and (chunk.text_delta or ""):
                    text = chunk.text_delta or ""
                    if pii_redact:
                        text = redact_text(text)
                    assembled_text.append(text)
                    yield ChatChunk(type="content.delta", text_delta=text)
                    continue

                # Track tool call deltas for later execution
                if chunk.type == "tool_call.start":
                    saw_tool_signal = True
                    idx = chunk.tool_call_index or 0
                    tool_args_by_index.setdefault(idx, {"id": chunk.tool_call_id, "name": chunk.tool_name, "args": []})
                    # Forward start event
                    yield chunk
                    continue
                if chunk.type == "tool_call.delta":
                    saw_tool_signal = True
                    idx = chunk.tool_call_index or 0
                    rec = tool_args_by_index.setdefault(idx, {"id": chunk.tool_call_id, "name": chunk.tool_name, "args": []})
                    if chunk.arguments_delta:
                        rec["args"].append(chunk.arguments_delta)
                    # Forward delta event
                    yield chunk
                    continue

                # End of message
                if chunk.type == "message.end":
                    finish_reason_seen = chunk.finish_reason
                    yield chunk
                    continue

                # Default passthrough for any other chunk types
                yield chunk

            # Finalize assistant message for this turn
            assistant_tool_calls = []
            if saw_tool_signal and tool_args_by_index:
                from ..providers.types import ToolCall as _ToolCall

                # Build ToolCall list in index order
                for idx in sorted(tool_args_by_index.keys()):
                    rec = tool_args_by_index[idx]
                    arguments_json = "".join(rec.get("args", []))
                    assistant_tool_calls.append(
                        _ToolCall(
                            id=rec.get("id"),
                            name=rec.get("name") or "",
                            arguments_json=arguments_json,
                        )
                    )

            assistant_msg = ChatMessage(
                role="assistant",
                content="".join(assembled_text),
                tool_calls=(assistant_tool_calls or None),
            )
            messages.append(assistant_msg)

            # Telemetry for this turn
            try:
                if telemetry is not None:
                    dur_ms = telemetry.stop_timer(t0) if t0 is not None else 0.0
                    had_tools = bool(assistant_tool_calls)
                    if hasattr(telemetry, "record_turn"):
                        telemetry.record_turn(
                            duration_ms=dur_ms, finish_reason=finish_reason_seen, had_tool_calls=had_tools
                        )
            except Exception:
                pass

            # If no tool calls were requested, we're done
            if not assistant_tool_calls:
                break

            # Execute tool calls and append tool messages
            try:
                from ..tools.execution import execute_many

                name_to_entry = {fn: b.entry for fn, b in compiled.tool_bindings_by_fn.items()}
                overrides_by_name = {fn: (b.overrides or {}) for fn, b in compiled.tool_bindings_by_fn.items()}
                timeout_s = (compiled.tool_config.timeout_ms or 0) / 1000.0 if getattr(compiled.tool_config, "timeout_ms", None) else None
                max_workers = max(1, int(getattr(compiled.tool_config, "parallelism", 1)))

                # Respect max_calls_per_turn if set (>0)
                max_calls = int(getattr(compiled.tool_config, "max_calls_per_turn", 0) or 0)
                calls_to_run = assistant_tool_calls if max_calls <= 0 else assistant_tool_calls[: max(0, min(max_calls, len(assistant_tool_calls)))]

                tool_messages = execute_many(
                    calls_to_run,
                    name_to_entry=name_to_entry,
                    overrides_by_name=overrides_by_name,
                    timeout_s=timeout_s,
                    max_workers=max_workers,
                    redact_for_logging=pii_redact,
                    telemetry=telemetry,
                )
                for i, tm in enumerate(tool_messages):
                    messages.append(tm)
                    # Emit a tool_call.end event for UI synchronization (id and index when available)
                    tc = calls_to_run[i]
                    yield ChatChunk(
                        type="tool_call.end",
                        tool_call_index=i,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                    )
            except Exception:
                # If tool execution fails, stop attempting further rounds
                break

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

        # Build request, including tools if available
        provider_tools = None
        try:
            if compiled.tool_bindings_by_fn:
                provider_tools = [b.tool_spec for b in compiled.tool_bindings_by_fn.values()]
            elif compiled.tools_attached:
                # Fallback: use ToolEntry -> ToolSpec (names may not be unique across duplicates)
                provider_tools = [entry.to_tool_spec() for entry in compiled.tools_attached.values()]
        except Exception:
            provider_tools = None
        req = ChatRequest(
            model=mp.model_id,
            messages=messages,
            tools=provider_tools or None,
            tool_choice=(
                "auto"
                if (provider_tools and getattr(compiled.tool_config, "policy", "Auto") in ("Auto", "Required"))
                else "none"
            ),
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
