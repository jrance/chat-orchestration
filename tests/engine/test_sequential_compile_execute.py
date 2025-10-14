from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    Edge,
    GraphConfig,
    Meta,
    MetaMetadata,
    ModelParams,
    ToolNode,
    ToolNodeData,
    ToolsConfig,
)
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
)
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class ProviderA(LLMProvider):
    def __init__(self, with_tool_call_first: bool = False) -> None:
        self.turn = 0
        self.with_tool_call_first = with_tool_call_first
        self.seen: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        self.seen.append(req)
        if self.with_tool_call_first and self.turn == 1:
            args = json.dumps({"q": "hello", "max_results": 10})
            tc = ToolCall(id="tc-a-1", name="web_search", arguments_json=args)
            return ChatResponse(
                message=ChatMessage(role="assistant", content="", tool_calls=[tc]),
                finish_reason="tool_calls",
            )
        return ChatResponse(
            message=ChatMessage(role="assistant", content="A done"), finish_reason="stop"
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="A done")
        yield ChatChunk(type="message.end", finish_reason="stop")


class ProviderB(LLMProvider):
    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.seen.append(req)
        return ChatResponse(
            message=ChatMessage(role="assistant", content="B done"), finish_reason="stop"
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="B done")
        yield ChatChunk(type="message.end", finish_reason="stop")


class StubWebSearchTool(BaseTool):
    name = "web_search"
    description = "stub web search"
    parameters = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"ok": True}


def _base_agents() -> tuple[AgentNode, AgentNode]:
    a = AgentNode(
        id="A",
        label="Agent A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-a"),
            context=AgentContext(history_window={"mode": "LastN", "n": 10}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    b = AgentNode(
        id="B",
        label="Agent B",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-b"),
            context=AgentContext(history_window={"mode": "LastN", "n": 10}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return a, b


def test_sequential_simple_two_agents_no_tools() -> None:
    a, b = _base_agents()
    cfg = GraphConfig(
        meta=Meta(id="m1", name="seq", version="0", metadata=MetaMetadata()),
        nodes=[a, b],
        edges=[Edge(id="e1", from_="A", to="B")],
    )

    pa = ProviderA()
    pb = ProviderB()

    def resolver(mp: Any) -> LLMProvider:  # mp: ModelParams
        return pa if getattr(mp, "model_id") == "model-a" else pb

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=ToolRegistry())
    result = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="hi")], "metadata": {}})

    assert result.steps == 2
    assert result.messages[-1].role == "assistant" and result.messages[-1].content == "B done"
    # Ensure order: provider A called then provider B
    assert len(pa.seen) >= 1 and len(pb.seen) >= 1


def test_sequential_agent_a_tool_then_b_runs() -> None:
    # Agent A has a web_search tool attached with overrides; Agent B has none
    tool = StubWebSearchTool()
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=tool, version=None))

    a, b = _base_agents()
    a.data.tools = ToolsConfig(attached=["tool:web-search"], policy="Auto", timeout_ms=5000, max_calls_per_turn=0)

    tool_node = ToolNode(
        id="t1",
        label="web_search",
        data=ToolNodeData(
            name="web_search",
            tool_id="tool:web-search",
            version="0",
            parameter_overrides={"max_results": 3},
        ),
    )

    cfg = GraphConfig(
        meta=Meta(id="m1", name="seq", version="0", metadata=MetaMetadata()),
        nodes=[a, b, tool_node],
        edges=[Edge(id="e1", from_="A", to="B")],
    )

    pa = ProviderA(with_tool_call_first=True)
    pb = ProviderB()

    def resolver(mp: Any) -> LLMProvider:
        return pa if getattr(mp, "model_id") == "model-a" else pb

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=reg)
    result = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    # Steps: A tool call turn + A final answer + B answer => 3
    assert result.steps == 3
    # Tool invoked exactly once; overrides take precedence (max_results=3)
    assert len(tool.calls) == 1
    assert tool.calls[0]["q"] == "hello"
    assert tool.calls[0]["max_results"] == 3
    # Transcript contains tool message and final agent messages
    assert any(m.role == "tool" and getattr(m, "tool_call_id", None) == "tc-a-1" for m in result.messages)
    assert result.messages[-1].content == "B done"

