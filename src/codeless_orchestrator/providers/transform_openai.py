from __future__ import annotations

from typing import Any

from .types import ChatMessage, ContentPartText, ToolCall, ToolSpec, Usage


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    # Content list: join text parts (multimodal extension later)
    parts = content or []
    texts: list[str] = []
    for p in parts:
        if isinstance(p, ContentPartText):
            texts.append(p.text)
        elif isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
    return "".join(texts)


def to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        data: dict[str, Any] = {"role": m.role}
        text = _content_to_text(m.content)
        data["content"] = text
        if m.role == "tool" and m.tool_call_id:
            data["tool_call_id"] = m.tool_call_id
        if m.role == "assistant" and m.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_json,
                    },
                }
                for tc in m.tool_calls
            ]
        out.append(data)
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
        )
    return out


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def from_openai_message(msg: Any) -> ChatMessage:
    role = _attr_or_key(msg, "role")
    content = _attr_or_key(msg, "content")
    # Tool calls mapping when present on assistant messages
    tool_calls_raw = _attr_or_key(msg, "tool_calls")
    tool_calls: list[ToolCall] | None = None
    if tool_calls_raw:
        tool_calls = []
        for tc in tool_calls_raw:
            tc_id = _attr_or_key(tc, "id")
            function_obj = _attr_or_key(tc, "function")
            name = _attr_or_key(function_obj, "name")
            arguments = _attr_or_key(function_obj, "arguments")
            tool_calls.append(ToolCall(id=tc_id, name=name, arguments_json=arguments or ""))

    return ChatMessage(role=role, content=content or "", tool_calls=tool_calls)


def usage_from_openai(resp: Any) -> Usage | None:
    usage = _attr_or_key(resp, "usage")
    if not usage:
        return None
    return Usage(
        prompt_tokens=_attr_or_key(usage, "prompt_tokens"),
        completion_tokens=_attr_or_key(usage, "completion_tokens"),
        total_tokens=_attr_or_key(usage, "total_tokens"),
    )
