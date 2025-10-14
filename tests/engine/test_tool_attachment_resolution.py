from __future__ import annotations

from typing import Any, Optional

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
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class StubProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_req: Optional[ChatRequest] = None

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.last_req = req
        return ChatResponse(message=ChatMessage(role="assistant", content="ok"))

    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        self.last_req = req
        yield  # type: ignore[misc]


class StubSearch(BaseTool):
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

    def invoke(self, args: dict[str, Any]) -> Any:  # pragma: no cover - not used here
        return {"args": args}


def _cfg_with_tool_node(node_id: str) -> GraphConfig:
    tool = ToolNode(
        id=node_id,
        label="Web Search",
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
            tools=ToolsConfig(attached=[node_id], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()), nodes=[tool, agent], edges=[])


def test_attached_tool_node_resolves_to_binding_and_function_name() -> None:
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=StubSearch(), version=None))

    stub = StubProvider()
    cfg = _cfg_with_tool_node("node-93d1bbce")

    compiled = compile_single_agent(cfg, provider_resolver=lambda _mp: stub, registry=reg)

    # Trigger a provider call so last_req is captured
    app = compiled.graph
    state = {"messages": [ChatMessage(role="user", content="hi")], "metadata": {}, "tool_calls": None, "steps": 0}
    _ = app.invoke(state)  # type: ignore[attr-defined]

    assert stub.last_req is not None
    assert stub.last_req.tools is not None
    assert len(stub.last_req.tools) == 1
    fn = stub.last_req.tools[0].name
    assert fn.startswith("web_search__")

    # Compiled artifact should expose binding map for single-agent
    assert compiled.tool_bindings_by_fn is not None
    assert fn in compiled.tool_bindings_by_fn
    b = compiled.tool_bindings_by_fn[fn]
    assert b.tool_id == "tool:web-search"
    assert b.overrides.get("max_results") == 3


def test_backward_compat_direct_tool_id_attachment() -> None:
    # No tool node; attached contains the registry toolId string
    agent = AgentNode(
        id="agent-1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 5}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["tool:web-search"], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()), nodes=[agent], edges=[])
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=StubSearch(), version=None))

    stub = StubProvider()
    compiled = compile_single_agent(cfg, provider_resolver=lambda _mp: stub, registry=reg)

    app = compiled.graph
    state = {"messages": [ChatMessage(role="user", content="hi")], "metadata": {}, "tool_calls": None, "steps": 0}
    _ = app.invoke(state)  # type: ignore[attr-defined]

    assert stub.last_req is not None
    assert stub.last_req.tools is not None
    assert len(stub.last_req.tools) == 1
    # Back-compat: function name equals base tool name when attached by toolId
    assert stub.last_req.tools[0].name == "web_search"
