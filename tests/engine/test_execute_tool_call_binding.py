from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    GraphConfig,
    Meta,
    MetaMetadata,
    ModelParams,
    ToolNode,
    ToolNodeData,
    ToolsConfig,
)
from codeless_orchestrator.engine.compiler import compile_single_agent
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class StubProvider(LLMProvider):
    def __init__(self) -> None:
        self.fn_name: Optional[str] = None
        self.turn = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            # First turn: emit tool call
            assert self.fn_name is not None, "function name must be set by test"
            tc = ToolCall(id="call_1", name=self.fn_name, arguments_json=json.dumps({"q": "x", "max_results": 10}))
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        # Second turn: normal assistant completes
        return ChatResponse(message=ChatMessage(role="assistant", content="done"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        if False:
            yield


class EchoTool(BaseTool):
    name = "web_search"
    description = "Search"
    parameters = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> Any:
        # Echo args back to validate merging
        return {"args": dict(args)}


def _cfg() -> GraphConfig:
    tool_node = ToolNode(
        id="node-93d1bbce",
        label="Search",
        data=ToolNodeData(name="web_search", tool_id="tool:web-search", version="1.0", parameter_overrides={"max_results": 3}),
    )
    agent = AgentNode(
        id="agent-1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 5}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["node-93d1bbce"], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()), nodes=[tool_node, agent], edges=[])


def test_execute_tool_call_binding_and_overrides_merge() -> None:
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=EchoTool(), version=None))
    provider = StubProvider()

    compiled = compile_single_agent(_cfg(), provider_resolver=lambda _mp: provider, registry=reg)

    # Capture resolved function name from compiled bindings
    assert compiled.tool_bindings_by_fn is not None
    assert len(compiled.tool_bindings_by_fn) == 1
    fn_name = next(iter(compiled.tool_bindings_by_fn.keys()))
    provider.fn_name = fn_name

    app = compiled.graph
    out = app.invoke({"messages": [ChatMessage(role="user", content="hi")], "metadata": {}, "tool_calls": None, "steps": 0})  # type: ignore[attr-defined]
    assert isinstance(out, dict)
    msgs = out.get("messages") or []
    # Expect a tool output message with the same call id
    tm = next(m for m in msgs if getattr(m, "role", None) == "tool")
    assert tm.tool_call_id == "call_1"
    parsed = json.loads(tm.content)
    assert parsed["result"]["args"]["max_results"] == 3
