from __future__ import annotations

from typing import Any, Literal

from ..config.schema import AgentNode
from ..providers.types import ChatMessage
from .templates import merge_vars, render


RoleCounted = {"user", "assistant", "tool"}


def window_messages(
    messages: list[ChatMessage],
    mode: Literal["LastN"],
    n: int,
) -> list[ChatMessage]:
    """Return a history window over `messages`.

    - Excludes `system` role from the count.
    - Preserves original order in the returned list.
    - Returns the last `n` messages where role in {user, assistant, tool}.
    """
    if mode != "LastN":  # pragma: no cover - only LastN supported for now
        return list(messages)
    if n <= 0:
        return []
    # Filter to counted roles and take last n, preserving order
    counted: list[ChatMessage] = [m for m in messages if getattr(m, "role", None) in RoleCounted]
    if len(counted) <= n:
        return counted
    return counted[-n:]


def make_system_message(
    system_instructions: str,
    style_guide: str,
    org_preamble: str | None,
    vars: dict[str, Any] | None,
) -> ChatMessage | None:
    """Construct a single system ChatMessage from components.

    - Concatenate optional organization preamble, system instructions, style guide.
    - Apply template rendering to system and style guide (not to preamble).
    - Strip empty parts; return None if all are empty after rendering.
    """
    parts: list[str] = []
    preamble = (org_preamble or "").strip()
    if preamble:
        parts.append(preamble)
    sys_txt = render(system_instructions or "", vars or {})
    if sys_txt.strip():
        parts.append(sys_txt.strip())
    guide_txt = render(style_guide or "", vars or {})
    if guide_txt.strip():
        parts.append(guide_txt.strip())
    if not parts:
        return None
    return ChatMessage(role="system", content="\n\n".join(parts))


def assemble_prompt(
    agent: AgentNode,
    all_messages: list[ChatMessage],
    org_preamble: str | None,
    vars: dict[str, Any] | None,
) -> list[ChatMessage]:
    """Build the prompt messages for an agent's turn.

    Order: [system?] + windowed history
    """
    ctx = agent.data.context
    # Merge precedence: engine-level (empty for now) < config < request
    merged = merge_vars({}, ctx.vars or {}, vars or {})
    preamble = org_preamble if ctx.inject_org_preamble else None
    sys_msg = make_system_message(
        agent.data.system_instructions or "",
        agent.data.style_guide or "",
        preamble,
        merged,
    )
    hw = ctx.history_window
    history = window_messages(all_messages, hw.mode, int(hw.n or 0))
    return ([sys_msg] if sys_msg else []) + history


__all__ = ["window_messages", "make_system_message", "assemble_prompt"]

