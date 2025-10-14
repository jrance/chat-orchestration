from __future__ import annotations

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
    ToolsConfig,
)
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.engine.graph_builders import handoff as handoff_builder
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse


class ProviderSimple(LLMProvider):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: int = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse(message=ChatMessage(role="assistant", content=self.text), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta=self.text)
        yield ChatChunk(type="message.end", finish_reason="stop")


def _agents() -> tuple[AgentNode, AgentNode, AgentNode]:
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
    return a, b, c


def test_handoff_router_selects_c(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, c = _agents()
    cfg = GraphConfig(
        meta=Meta(id="m1", name="handoff", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c],
        edges=[Edge(id="e1", from_="A", to="B"), Edge(id="e2", from_="A", to="C")],
    )

    pa = ProviderSimple("A says hi")
    pb = ProviderSimple("B done")
    pc = ProviderSimple("C done")

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-a":
            return pa
        if mid == "model-b":
            return pb
        return pc

    # Stub router to choose C deterministically
    monkeypatch.setattr(handoff_builder, "choose_next_agent", lambda **kwargs: "C")

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    assert res.steps == 2
    assert res.messages[-1].content == "C done"
    assert pa.calls == 1 and pc.calls == 1


def test_handoff_router_invalid_fallback_first_candidate(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    a, b, c = _agents()
    cfg = GraphConfig(
        meta=Meta(id="m1", name="handoff", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c],
        edges=[Edge(id="e1", from_="A", to="B"), Edge(id="e2", from_="A", to="C")],
    )

    pa = ProviderSimple("A says hi")
    pb = ProviderSimple("B done")
    pc = ProviderSimple("C done")

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-a":
            return pa
        if mid == "model-b":
            return pb
        return pc

    # Return invalid id => should fallback to first candidate ('B') and log warning
    monkeypatch.setattr(handoff_builder, "choose_next_agent", lambda **kwargs: "Z")

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)

    with caplog.at_level("WARNING"):
        res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    assert res.steps == 2
    assert res.messages[-1].content == "B done"
    # Ensure a warning was logged
    assert any("Router returned invalid id" in rec.message for rec in caplog.records)
