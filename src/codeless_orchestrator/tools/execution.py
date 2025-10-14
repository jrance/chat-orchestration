from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from jsonschema import (  # type: ignore[import-untyped]
    ValidationError as JSONSchemaValidationError,
)
from jsonschema import (
    validate as jsonschema_validate,
)

from ..providers.types import ChatMessage, ToolCall
from ..runtime.safety import redact_text
from .base import BaseTool, ToolTimeoutError, ToolValidationError
from .registry import ToolEntry


def merge_overrides(args: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge overrides with precedence over provided args."""
    merged = dict(args or {})
    merged.update(overrides or {})
    return merged


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Validate arguments against a JSON Schema, raising ToolValidationError on failure."""
    try:
        jsonschema_validate(instance=args, schema=schema)
    except JSONSchemaValidationError as exc:  # pragma: no cover - message verified via exception
        raise ToolValidationError(f"Arguments failed schema validation: {exc.message}") from exc


def run_with_timeout(func: Callable[[], Any], timeout_s: float | None) -> Any:
    """Run a function with an optional timeout in seconds."""
    if timeout_s is None or timeout_s <= 0:
        return func()
    with ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(func)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            fut.cancel()
            raise ToolTimeoutError("Tool execution timed out") from exc


def _parse_arguments_json(arguments_json: str) -> dict[str, Any]:
    if not arguments_json:
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ToolValidationError(f"Invalid arguments JSON: {exc.msg}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ToolValidationError("Tool arguments must be a JSON object")
    return parsed


def execute_tool_call(
    tool: BaseTool,
    arguments_json: str,
    parameter_overrides: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    *,
    redact_for_logging: bool = True,
    telemetry: Any | None = None,
) -> dict[str, Any]:
    """Execute a tool deterministically with overrides, validation, and timeout.

    Returns a dictionary suitable for downstream transcript inclusion.
    """
    args = _parse_arguments_json(arguments_json)
    final_args = merge_overrides(args, parameter_overrides or {})
    validate_args(tool.parameters, final_args)

    def _call() -> Any:
        return tool.invoke(final_args)

    import time as _t

    t0 = None
    try:
        if telemetry is not None and hasattr(telemetry, "start_timer"):
            t0 = telemetry.start_timer(f"tool:{tool.name}")
    except Exception:
        t0 = None
    try:
        result = run_with_timeout(_call, timeout_s)
        normalized = BaseTool.normalize_result(result)
        return {"result": normalized, "tool_name": tool.name}
    except Exception as exc:
        try:
            if telemetry is not None and hasattr(telemetry, "record_error"):
                telemetry.record_error("tool", str(exc))
            if telemetry is not None and hasattr(telemetry, "record_tool_call"):
                # Record as an errored call with no duration (or partial)
                duration = None
                if t0 is not None and hasattr(telemetry, "stop_timer"):
                    try:
                        duration = telemetry.stop_timer(t0)
                    except Exception:
                        duration = None
                telemetry.record_tool_call(tool.name, duration_ms=duration, error=True)
        except Exception:
            pass
        raise
    finally:
        try:
            if t0 is not None and hasattr(telemetry, "stop_timer"):
                dur_ms = telemetry.stop_timer(t0)
                if telemetry is not None and hasattr(telemetry, "record_tool_call"):
                    telemetry.record_tool_call(tool.name, duration_ms=dur_ms, error=False)
        except Exception:
            pass


def execute_many(
    tool_calls: list[ToolCall],
    *,
    name_to_entry: dict[str, ToolEntry],
    overrides_by_name: dict[str, dict[str, Any]],
    timeout_s: float | None,
    max_workers: int,
    redact_for_logging: bool = True,
    telemetry: Any | None = None,
) -> list[ChatMessage]:
    """Execute multiple tool calls concurrently with bounded parallelism.

    Preserves the original order of tool calls in the returned ChatMessages.
    """
    if not tool_calls:
        return []

    # Local worker for a single tool call
    def _run_one(idx: int, tc: ToolCall) -> tuple[int, ChatMessage]:
        tool_name = tc.name
        if tool_name not in name_to_entry:
            # Use a validation-style error to avoid cross-module import here
            raise ToolValidationError(f"Tool '{tool_name}' is not attached to the agent")
        entry = name_to_entry[tool_name]
        overrides = overrides_by_name.get(tool_name) or {}
        result_obj = execute_tool_call(
            entry.impl,
            tc.arguments_json or "",
            parameter_overrides=overrides,
            timeout_s=timeout_s,
            redact_for_logging=redact_for_logging,
            telemetry=telemetry,
        )
        content_json = json.dumps(result_obj)
        if redact_for_logging:
            content_json = redact_text(content_json)
        return idx, ChatMessage(role="tool", content=content_json, tool_call_id=tc.id)

    # Serial path if only one call or parallelism is 1
    if len(tool_calls) == 1 or max_workers <= 1:
        out: list[ChatMessage] = []
        for i, tc in enumerate(tool_calls):
            _, cm = _run_one(i, tc)
            out.append(cm)
        return out

    # Concurrent path
    maxw = max(1, min(max_workers, len(tool_calls)))
    results: list[tuple[int, ChatMessage]] = []
    with ThreadPoolExecutor(max_workers=maxw) as pool:
        futs = {pool.submit(_run_one, i, tc): i for i, tc in enumerate(tool_calls)}
        for fut in as_completed(futs):
            results.append(fut.result())
    # Restore original order by index
    results.sort(key=lambda t: t[0])
    return [cm for _, cm in results]
