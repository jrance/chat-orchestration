from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from codeless_orchestrator.tools.base import (
    BaseTool,
    ToolError,
    ToolTimeoutError,
    ToolValidationError,
)
from codeless_orchestrator.tools.execution import (
    execute_tool_call,
    merge_overrides,
)


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input arguments back as result."
    parameters = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args}


def test_merge_overrides_precedence() -> None:
    llm_args = {"q": "foo", "max_results": 10}
    overrides = {"max_results": 3}
    merged = merge_overrides(llm_args, overrides)
    assert merged == {"q": "foo", "max_results": 3}


def test_validate_args_schema_violation() -> None:
    tool = EchoTool()
    # Missing required q
    with pytest.raises(ToolValidationError):
        execute_tool_call(
            tool,
            arguments_json=json.dumps({}),
            parameter_overrides=None,
            timeout_s=None,
        )


class SleeperTool(BaseTool):
    name = "sleeper"
    description = "Sleeps for the requested duration."
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {"type": "number"},
        },
        "required": ["seconds"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> str:
        seconds = float(args["seconds"])  # type: ignore[assignment]
        time.sleep(seconds)
        return "done"


def test_timeout() -> None:
    tool = SleeperTool()
    with pytest.raises(ToolTimeoutError):
        execute_tool_call(
            tool,
            arguments_json=json.dumps({"seconds": 0.5}),
            parameter_overrides=None,
            timeout_s=0.1,
        )


@dataclass
class Data:
    a: int
    b: str


class ReturnsVariousTool(BaseTool):
    name = "various"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, result: Any) -> None:
        self._result = result

    def invoke(self, args: dict[str, Any]) -> Any:  # pragma: no cover - simple passthrough
        return self._result


def test_result_json_serializable_dataclass_normalized() -> None:
    tool = ReturnsVariousTool(Data(a=1, b="x"))
    out = execute_tool_call(tool, arguments_json="{}", parameter_overrides=None, timeout_s=None)
    assert out["result"] == {"a": 1, "b": "x"}


def test_result_json_serializable_invalid_type_raises() -> None:
    tool = ReturnsVariousTool(set([1, 2]))
    with pytest.raises(ToolError):
        execute_tool_call(tool, arguments_json="{}", parameter_overrides=None, timeout_s=None)
