from __future__ import annotations

from typing import Any

from ..providers.types import ChatChunk


def chunk_to_event(chunk: ChatChunk) -> dict[str, Any]:
    """Convert a ChatChunk into a plain dict suitable for SSE or JSON.

    This preserves a stable set of keys based on the chunk type.
    """
    base: dict[str, Any] = {"type": chunk.type}
    if chunk.role is not None:
        base["role"] = chunk.role
    if chunk.text_delta is not None:
        base["text_delta"] = chunk.text_delta
    if chunk.tool_call_index is not None:
        base["tool_call_index"] = chunk.tool_call_index
    if chunk.tool_call_id is not None:
        base["tool_call_id"] = chunk.tool_call_id
    if chunk.tool_name is not None:
        base["tool_name"] = chunk.tool_name
    if chunk.arguments_delta is not None:
        base["arguments_delta"] = chunk.arguments_delta
    if chunk.finish_reason is not None:
        base["finish_reason"] = chunk.finish_reason
    if chunk.usage is not None:
        base["usage"] = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }
    return base


__all__ = ["chunk_to_event"]
