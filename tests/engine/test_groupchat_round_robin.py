from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

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
from codeless_orchestrator.engine.graph_builders import groupchat as group_builder
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class ProviderA(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="A says hi"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="A says hi")
        yield ChatChunk(type="message.end", finish_reason="stop")


class ProviderB(LLMProvider):
    def __init__(self) -> None:
        self.turn = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            args = json.dumps({"q": "hello", "max_results": 9})
            tc = ToolCall(id="tc-b-1", name="web_search", arguments_json=args)
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        return ChatResponse(message=ChatMessage(role="assistant", content="B done"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="B done")
        yield ChatChunk(type="message.end", finish_reason="stop")


class ProviderC(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="C done"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="C done")
        yield ChatChunk(type="message.end", finish_reason="stop")


class StubWebSearchTool(BaseTool):
    name = "web_search"
    description = "stub web search"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"ok": True}


def _graph_cycle_with_b_tool() -> tuple[GraphConfig, ToolRegistry, StubWebSearchTool]:
    a = AgentNode(
        id="A",
        label="Agent A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-a"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
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
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["tool:web-search"], policy="Auto", timeout_ms=5000, parallelism=4),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    c = AgentNode(
        id="C",
        label="Agent C",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-c"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    tool = StubWebSearchTool()
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", version=None, impl=tool))
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
        meta=Meta(id="m1", name="groupchat", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c, tool_node],
        edges=[
            Edge(id="e1", from_="A", to="B"),
            Edge(id="e2", from_="B", to="C"),
            Edge(id="e3", from_="C", to="A"),
        ],
    )
    return cfg, reg, tool


def test_groupchat_round_robin_one_round(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, reg, tool_impl = _graph_cycle_with_b_tool()

    # Force max_turns=3 for a single round A->B->C
    monkeypatch.setattr(group_builder, "DEFAULT_MAX_TURNS", 3)

    pa = ProviderA()
    pb = ProviderB()
    pc = ProviderC()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-a":
            return pa
        if mid == "model-b":
            return pb
        return pc

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=reg)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    # Check that assistant messages include speaker attribution when there is content
    assistant_texts = [m.content for m in res.messages if m.role == "assistant" and isinstance(m.content, str) and m.content]
    # There should be at least A, B (final), C
    joined = "\n".join(assistant_texts)
    assert "[Agent: Agent A] A says hi" in joined
    assert "[Agent: Agent B] B done" in joined
    assert "[Agent: Agent C] C done" in joined

    # Tool executed for B with overrides merged
    assert len(tool_impl.calls) == 1
    assert tool_impl.calls[0]["q"] == "hello"
    assert tool_impl.calls[0]["max_results"] == 3  # overridden by tool node

