# Tools

This document describes the tools subsystem: a modular framework to define, register, validate, and execute tools, and how they are exposed to providers.

## Overview

- A tool is a small, typed unit of work with:
  - `name`: stable identifier for provider-facing function name
  - `description`: optional text shown to models
  - `parameters`: JSON Schema describing accepted arguments
  - `invoke(args) -> dict | list | str`: pure, side-effect-light implementation
- Tools are discoverable via a registry and can be presented to model providers via `ToolSpec`.

## Implementing a Tool

1. Subclass `BaseTool` and set `name`, `description`, and a JSON Schema in `parameters`.
2. Implement `invoke(self, args: dict[str, Any]) -> dict | list | str`.
3. Ensure the result is JSON-serializable (dict/list/str). Dataclasses are normalized automatically.

Example skeleton:

```
from typing import Any
from codeless_orchestrator.tools.base import BaseTool

class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input back"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args}
```

## Registry

Use `ToolRegistry` to register and discover tools by `tool_id` (e.g., `tool:web-search`).

```
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry

reg = ToolRegistry()
reg.register(ToolEntry(tool_id="tool:echo", version="1", impl=EchoTool()))
entry = reg.get("tool:echo")
specs = reg.to_provider_tool_specs(["tool:echo"])  # -> list[ToolSpec]
```

`to_provider_tool_specs` converts registry entries into provider-facing `ToolSpec` objects, which can be passed to providers (e.g., OpenAI) using existing transforms.

## Overrides

Per-call `parameterOverrides` take precedence over LLM-proposed arguments. Overrides are a shallow merge and replace any overlapping values.

```
from codeless_orchestrator.tools.execution import merge_overrides

final_args = merge_overrides({"q": "foo", "max_results": 10}, {"max_results": 3})
# final_args == {"q": "foo", "max_results": 3}
```

## Validation

Arguments are validated against the tool's JSON Schema before invocation. Invalid inputs raise `ToolValidationError` with a clear message.

## Timeouts

Use `execute_tool_call` to enforce per-call timeouts and deterministic behavior.

```
from codeless_orchestrator.tools.execution import execute_tool_call

result = execute_tool_call(tool, arguments_json="{\"q\": \"openai\"}", parameter_overrides=None, timeout_s=5.0)
```

- If execution exceeds `timeout_s`, `ToolTimeoutError` is raised.
- Results are normalized to `dict | list | str` for transcript inclusion.

## Builtin: Web Search

`tool:web-search` is provided via DuckDuckGo.

**Note**: This tool requires the optional `websearch` dependency. Install with: `pip install -e .[websearch]`  
If the dependency is not installed, the tool will be gracefully skipped during registration.

- Parameters:
  - `q` (string, required)
  - `max_results` (int, 1–50, default 5)
  - `region` (string)
  - `safesearch` (enum: off, moderate, strict)
  - `time` (enum: d, w, m, y)
  - `backend` (enum: auto, api, html)
- Output shape: list of objects with `title`, `href`, `body`, and `source` set to `"duckduckgo"`.

You can inject a stub client for testing (no network).

