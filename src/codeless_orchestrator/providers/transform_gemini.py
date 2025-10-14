from __future__ import annotations

from typing import Any

from .types import ChatMessage, ToolCall, ToolSpec, Usage


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = content or []
    out: list[str] = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            t = p.get("text")
            if isinstance(t, str):
                out.append(t)
        else:
            t = _attr_or_key(p, "text")
            if isinstance(t, str):
                out.append(t)
    return "".join(out)


def to_gemini_system_instruction(messages: list[ChatMessage]) -> str | None:
    parts: list[str] = []
    for m in messages:
        if m.role == "system":
            text = _content_to_text(m.content)
            if text:
                parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def to_gemini_history(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Build a mapping of tool_call_id -> name from prior assistant message
    id_to_name: dict[str, str] = {}
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.id:
                    id_to_name[tc.id] = tc.name

    for m in messages:
        if m.role == "system":
            # Excluded from history (represented via system_instruction)
            continue

        # Gemini roles are strings and include "model" where our interface uses "assistant"
        g_role: str = m.role
        if g_role == "assistant":
            g_role = "model"

        parts: list[dict[str, Any]] = []
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                # Arguments arrive as raw JSON string; try to parse later at tool side.
                # Gemini expects a dict for args in function_call.
                args_json = tc.arguments_json or ""
                try:
                    # Safe best-effort parse; fall back to string if invalid
                    import json

                    args = json.loads(args_json) if args_json else {}
                except Exception:
                    args = {"__raw": args_json}
                parts.append({"function_call": {"name": tc.name, "args": args}})
        elif m.role == "tool":
            # Tool result must be a function_response part with name resolved from the id
            tool_name = None
            if m.tool_call_id and m.tool_call_id in id_to_name:
                tool_name = id_to_name[m.tool_call_id]
            # Attempt to parse JSON response; fallback to text
            payload = _content_to_text(m.content)
            response_obj: Any
            try:
                import json

                response_obj = json.loads(payload)
            except Exception:
                response_obj = payload
            parts.append(
                {"function_response": {"name": tool_name or "", "response": response_obj}}
            )
        else:
            # user/model normal text message
            text = _content_to_text(m.content)
            if text:
                parts.append({"text": text})

        out.append({"role": g_role, "parts": parts or [{"text": ""}]})
    return out


def to_gemini_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    if not tools:
        return []
    fns: list[dict[str, Any]] = []
    for t in tools:
        fns.append(
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        )
    return [{"function_declarations": fns}]


def to_gemini_tool_config(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"function_calling_config": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    # Specific selection e.g., {"type":"function","function":{"name":"x"}}
    if isinstance(tool_choice, dict):
        fn = _attr_or_key(_attr_or_key(tool_choice, "function", {}), "name")
        if isinstance(fn, str) and fn:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [fn],
                }
            }
    return None


def to_gemini_generation_config(req: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if req.top_p is not None:
        config["top_p"] = req.top_p
    if req.max_tokens is not None:
        config["max_output_tokens"] = req.max_tokens
    if req.stop is not None:
        config["stop_sequences"] = req.stop
    if req.seed is not None:
        # best-effort: newer SDKs may support seed
        config["seed"] = req.seed
    if req.json_mode_enabled:
        config["response_mime_type"] = "application/json"
        if req.response_format_schema:
            # Best-effort: some SDK versions accept response_schema
            try:
                config["response_schema"] = req.response_format_schema  # type: ignore[assignment]
            except Exception:
                pass
    return config


def _gemini_parts_to_message(parts: list[Any]) -> ChatMessage:
    text_acc: list[str] = []
    tool_calls: list[ToolCall] = []
    for p in parts:
        # text part
        text = _attr_or_key(p, "text")
        if isinstance(text, str) and text:
            text_acc.append(text)
            continue

        # function call part
        fc = _attr_or_key(p, "function_call") or _attr_or_key(p, "functionCall")
        if fc:
            name = _attr_or_key(fc, "name")
            args = _attr_or_key(fc, "args") or {}
            import json

            try:
                args_json = json.dumps(args)
            except Exception:
                args_json = "{}"
            tool_calls.append(ToolCall(id=None, name=name, arguments_json=args_json))

    if tool_calls:
        return ChatMessage(role="assistant", content="", tool_calls=tool_calls)
    return ChatMessage(role="assistant", content="".join(text_acc))


def from_gemini_candidate(candidate: Any) -> ChatMessage:
    # Candidate.content.parts is the primary source
    content = _attr_or_key(candidate, "content")
    if content is None:
        return ChatMessage(role="assistant", content="")
    parts = _attr_or_key(content, "parts") or []
    return _gemini_parts_to_message(parts)


def usage_from_gemini(resp: Any) -> Usage | None:
    meta = _attr_or_key(resp, "usage_metadata") or _attr_or_key(resp, "usageMetadata")
    if not meta:
        return None
    # Support both snake_case and camelCase
    prompt = (
        _attr_or_key(meta, "prompt_token_count")
        or _attr_or_key(meta, "promptTokenCount")
        or None
    )
    completion = (
        _attr_or_key(meta, "candidates_token_count")
        or _attr_or_key(meta, "candidatesTokenCount")
        or None
    )
    total = (
        _attr_or_key(meta, "total_token_count")
        or _attr_or_key(meta, "totalTokenCount")
        or None
    )
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def map_finish_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    # Gemini reasons are typically UPPERCASE identifiers
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "BLOCKED": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(reason, reason.lower())
