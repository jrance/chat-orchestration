from __future__ import annotations

from typing import Any

from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input arguments back as result."
    parameters = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args}


def test_register_and_get_tool() -> None:
    reg = ToolRegistry()
    entry = ToolEntry(tool_id="tool:echo", version="1", impl=EchoTool())
    reg.register(entry)

    got = reg.get("tool:echo")
    assert got.tool_id == "tool:echo"
    assert isinstance(got.impl, EchoTool)

    specs = reg.to_provider_tool_specs(["tool:echo"])
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "echo"
    assert spec.description == "Echo input arguments back as result."
    assert spec.parameters["type"] == "object"
    assert "q" in spec.parameters["properties"]


def test_unknown_tool_lookup() -> None:
    reg = ToolRegistry()
    try:
        reg.get("tool:missing")
        raise AssertionError("Expected KeyError")
    except KeyError as e:
        assert "Unknown tool_id" in str(e)
