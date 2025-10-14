from __future__ import annotations

import json
import re
from typing import Any, Tuple

from jsonschema import ValidationError as JSONSchemaValidationError  # type: ignore[import-untyped]
from jsonschema import validate as jsonschema_validate  # type: ignore[import-untyped]

from ..providers.base import LLMProvider
from ..providers.types import ChatMessage, ChatRequest, ChatResponse, Usage


def build_system_hint(schema: dict[str, Any] | None) -> str:
    base = (
        "Output ONLY valid JSON. No code fences. Do not include explanations."
    )
    if schema:
        return (
            base
            + " Follow the provided JSON Schema exactly, including required fields and types."
        )
    return base


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    # Remove common triple backtick fences possibly with json tag
    stripped = _FENCE_RE.sub("", text.strip())
    # Also remove standalone triple backticks within
    return stripped.strip().strip("`")


def parse_json(text: str) -> tuple[Any | None, str | None]:
    try:
        obj = json.loads(text)
        return obj, None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc.msg} at pos {exc.pos}"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"JSON parse error: {exc}"


def validate_against_schema(obj: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        jsonschema_validate(instance=obj, schema=schema)
        return True, None
    except JSONSchemaValidationError as exc:
        return False, exc.message or "Schema validation error"
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def format_compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def make_repair_instruction(
    schema: dict[str, Any] | None,
    parse_err: str | None,
    val_err: str | None,
    previous_output: str,
) -> str:
    parts = [
        "The previous output was invalid. Provide corrected JSON only, no extra text.",
    ]
    if parse_err:
        parts.append(f"Parsing error: {parse_err}.")
    if val_err:
        parts.append(f"Validation error: {val_err}.")
    if schema:
        parts.append("Follow this JSON Schema exactly (no additional properties unless allowed):")
        parts.append(json.dumps(schema, ensure_ascii=False))
    parts.append("Previous output for reference (do not copy explanations):")
    parts.append(previous_output)
    return "\n".join(parts)


def _normalize_ws(text: str) -> str:
    # Collapse runs of whitespace to single spaces, preserving basic newlines
    if not text:
        return ""
    # Replace multiple whitespace (including newlines) with single space
    return re.sub(r"\s+", " ", text).strip()


def post_process_text(text: str, ensure_markdown: bool, normalize_ws: bool) -> str:
    out = text or ""
    if normalize_ws:
        out = _normalize_ws(out)
    if ensure_markdown:
        # If not already fenced or markdown-looking, wrap in a code block for safety
        if not out.startswith("```"):
            out = f"```\n{out}\n```"
    return out


def ensure_structured_output(
    provider: LLMProvider,
    req: ChatRequest,
    *,
    policy_cfg: Any,
    post_cfg: Any | None = None,
    system_hint: str | None = None,
) -> tuple[ChatMessage, Usage | None, str | None, int]:
    """Perform one or more non-stream completion attempts to ensure structured JSON.

    Returns (final_message, usage, finish_reason, attempts).
    The final_message.content will be a compact JSON string when successful.
    On Ignore policy or unrepairable errors, returns the last raw assistant message
    (possibly post-processed if non-JSON).
    """
    # Always force json mode for structured path
    base_messages = list(req.messages)
    if system_hint:
        base_messages = [*base_messages, ChatMessage(role="system", content=system_hint)]

    attempts = 0
    max_attempts = int(getattr(policy_cfg, "max_repair_attempts", 2) or 0) + 1
    on_violation = getattr(policy_cfg, "on_violation", "RetryAndRepair")
    schema = getattr(policy_cfg, "schema_", None) or {}

    last_usage: Usage | None = None
    last_finish: str | None = None
    last_text: str = ""

    while attempts < max_attempts:
        attempts += 1
        req_eff = ChatRequest(
            model=req.model,
            messages=list(base_messages),
            tools=req.tools,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            seed=req.seed,
            json_mode_enabled=True,
            response_format_schema=getattr(policy_cfg, "schema_", None) or getattr(policy_cfg, "schema", None) or None,
            metadata=req.metadata,
            timeout=req.timeout,
        )
        resp: ChatResponse = provider.chat(req_eff)
        last_usage = resp.usage
        last_finish = resp.finish_reason
        message = resp.message
        text = message.content if isinstance(message.content, str) else ""
        last_text = text

        # JSON cleaning
        raw = strip_code_fences(text)
        obj, parse_err = parse_json(raw)
        val_err: str | None = None
        if obj is not None and schema:
            ok, verr = validate_against_schema(obj, schema)
            if not ok:
                val_err = verr or "Invalid per schema"

        if obj is not None and not val_err:
            # Success; compact and return
            compact = format_compact_json(obj)
            final_msg = ChatMessage(role="assistant", content=compact)
            return final_msg, last_usage, last_finish, attempts

        # Violation
        if on_violation == "Ignore":
            # Return original assistant text, possibly post-processed
            processed = post_process_text(
                text,
                ensure_markdown=(getattr(post_cfg, "ensure_markdown", None)
                                 if post_cfg is not None else getattr(getattr(policy_cfg, "post_process", None), "ensure_markdown", True)),
                normalize_ws=(getattr(post_cfg, "normalize_whitespace", None)
                              if post_cfg is not None else getattr(getattr(policy_cfg, "post_process", None), "normalize_whitespace", True)),
            )
            return ChatMessage(role="assistant", content=processed), last_usage, last_finish, attempts
        if attempts >= max_attempts:
            if on_violation == "Refuse":
                raise ValueError(
                    f"Structured output validation failed after {attempts} attempt(s): "
                    f"parse_err={parse_err or ''} val_err={val_err or ''}"
                )
            # Return best-effort processed text
            processed = post_process_text(
                text,
                ensure_markdown=(getattr(post_cfg, "ensure_markdown", None)
                                 if post_cfg is not None else getattr(getattr(policy_cfg, "post_process", None), "ensure_markdown", True)),
                normalize_ws=(getattr(post_cfg, "normalize_whitespace", None)
                              if post_cfg is not None else getattr(getattr(policy_cfg, "post_process", None), "normalize_whitespace", True)),
            )
            return ChatMessage(role="assistant", content=processed), last_usage, last_finish, attempts

        # Retry with repair instruction appended
        instruction = make_repair_instruction(schema or {}, parse_err, val_err, last_text)
        base_messages = [*base_messages, ChatMessage(role="system", content=instruction)]

    # Fallback (should not reach)
    return ChatMessage(role="assistant", content=last_text), last_usage, last_finish, attempts


__all__ = [
    "build_system_hint",
    "strip_code_fences",
    "parse_json",
    "validate_against_schema",
    "format_compact_json",
    "make_repair_instruction",
    "post_process_text",
    "ensure_structured_output",
]
