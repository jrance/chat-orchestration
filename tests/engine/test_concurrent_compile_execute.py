from __future__ import annotations

import time
from typing import Any, Iterator

from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    Edge,
    GraphConfig,
    Meta,
    MetaMetadata,
    ModelParams,
    ToolsConfig,
)
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse


class StubProviderB(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        time.sleep(0.05)
        return ChatResponse(message=ChatMessage(role="assistant", content="B result"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="B result")
        yield ChatChunk(type="message.end", finish_reason="stop")


class StubProviderC(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        time.sleep(0.01)
        return ChatResponse(message=ChatMessage(role="assistant", content="C result"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="C result")
        yield ChatChunk(type="message.end", finish_reason="stop")


def _graph_three_nodes() -> GraphConfig:
    a = AgentNode(
        id="A",
        label="Agent A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-a"),
            context=AgentContext(history_window={"mode": "LastN", "n": 5}, inject_org_preamble=True, vars={}),
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
            context=AgentContext(history_window={"mode": "LastN", "n": 5}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
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
            context=AgentContext(history_window={"mode": "LastN", "n": 5}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(
        meta=Meta(id="m1", name="concurrent", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c],
        edges=[Edge(id="e1", from_="A", to="B"), Edge(id="e2", from_="A", to="C")],
    )


def test_concurrent_fanout_aggregate_order_and_speed() -> None:
    cfg = _graph_three_nodes()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-b":
            return StubProviderB()
        if mid == "model-c":
            return StubProviderC()
        # Root A is not used in concurrent stage
        return StubProviderB()

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)

    start = time.time()
    res = engine.execute(
        compiled,
        run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}},
    )
    elapsed = time.time() - start

    # Should be faster than sequential sum (0.05 + 0.01)
    assert elapsed < 0.09
    last = res.messages[-1]
    assert isinstance(last.content, str)
    # Deterministic order by agent id (B then C)
    assert last.content.splitlines()[0].startswith("B:")
    assert "B result" in last.content and "C result" in last.content

