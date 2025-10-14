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
from codeless_orchestrator.engine.group_policy import choose_next_speaker
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class ModeratorProvider(LLMProvider):
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)

    def chat(self, req: ChatRequest) -> ChatResponse:
        # Pop next decision
        if not self.decisions:
            payload = {"end": True}
        else:
            payload = self.decisions.pop(0)
        return ChatResponse(message=ChatMessage(role="assistant", content=json.dumps(payload)), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="{}")
        yield ChatChunk(type="message.end", finish_reason="stop")


class ProviderX(LLMProvider):
    def __init__(self, with_tool: bool = False) -> None:
        self.turn = 0
        self.with_tool = with_tool

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.with_tool and self.turn == 1:
            tc = ToolCall(id="tc-x-1", name="web_search", arguments_json=json.dumps({"q": "q"}))
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        return ChatResponse(message=ChatMessage(role="assistant", content="X's answer"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="X's answer")
        yield ChatChunk(type="message.end", finish_reason="stop")


class ProviderY(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="Y's answer"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="Y's answer")
        yield ChatChunk(type="message.end", finish_reason="stop")


class StubTool(BaseTool):
    name = "web_search"
    description = "stub"
    parameters = {"type": "object", "properties": {"q": {"type": "string"}}, "additionalProperties": False}

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, args: dict[str, Any]) -> Any:
        self.calls += 1
        return {"ok": True}


def _graph_moderated() -> GraphConfig:
    m = AgentNode(
        id="M",
        label="Moderator",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="router-model"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="None"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    x = AgentNode(
        id="X",
        label="Agent X",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-x"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    y = AgentNode(
        id="Y",
        label="Agent Y",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-y"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(
        meta=Meta(id="m1", name="groupchat moderator", version="0", metadata=MetaMetadata()),
        nodes=[m, x, y],
        edges=[
            Edge(id="e1", from_="M", to="X"),
            Edge(id="e2", from_="X", to="M"),
            Edge(id="e3", from_="M", to="Y"),
            Edge(id="e4", from_="Y", to="M"),
        ],
    )


def test_groupchat_moderator_routes_then_end() -> None:
    cfg = _graph_moderated()

    # M picks Y, then X, then end
    moderator = ModeratorProvider(decisions=[{"next": "Y"}, {"next": "X"}, {"end": True}])
    px = ProviderX()
    py = ProviderY()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "router-model":
            return moderator
        if mid == "model-x":
            return px
        return py

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    assistant_texts = [m.content for m in res.messages if m.role == "assistant" and isinstance(m.content, str) and m.content]
    # Expect Y then X
    assert any("[Agent: Agent Y] Y's answer" == t for t in assistant_texts)
    assert any("[Agent: Agent X] X's answer" == t for t in assistant_texts)


def test_groupchat_moderator_invalid_choice_fallback(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _graph_moderated()

    moderator = ModeratorProvider(decisions=[{"next": "Z"}, {"end": True}])
    px = ProviderX()
    py = ProviderY()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "router-model":
            return moderator
        if mid == "model-x":
            return px
        return py

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)

    with caplog.at_level("WARNING"):
        res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})
    # Should have picked first valid participant deterministically (X)
    assert any(m.role == "assistant" and isinstance(m.content, str) and m.content.startswith("[Agent: Agent X]") for m in res.messages)
    assert any("Moderator returned invalid id" in rec.message or "Moderator parse failed" in rec.message for rec in caplog.records)


def test_groupchat_moderator_with_tool_turn() -> None:
    # Y will be chosen first and produce a tool call, ensure tool executes before next select_next
    m = ModeratorProvider(decisions=[{"next": "X"}, {"end": True}])
    px = ProviderX(with_tool=True)
    py = ProviderY()

    # Graph: Moderator hub
    cfg = GraphConfig(
        meta=Meta(id="m1", name="groupchat moderator", version="0", metadata=MetaMetadata()),
        nodes=[
            AgentNode(
                id="M",
                label="Moderator",
                data=AgentNodeData(
                    system_instructions="",
                    style_guide="",
                    model=ModelParams(provider="openai", model_id="router-model"),
                    context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
                    tools=ToolsConfig(attached=[], policy="None"),
                    structured_output={},
                    safety={"policy_ref": "default"},
                    telemetry={"labels": {}, "emit_usage": True},
                ),
            ),
            AgentNode(
                id="X",
                label="Agent X",
                data=AgentNodeData(
                    system_instructions="",
                    style_guide="",
                    model=ModelParams(provider="openai", model_id="model-x"),
                    context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
                    tools=ToolsConfig(attached=["tool:web-search"], policy="Auto"),
                    structured_output={},
                    safety={"policy_ref": "default"},
                    telemetry={"labels": {}, "emit_usage": True},
                ),
            ),
            AgentNode(
                id="Y",
                label="Agent Y",
                data=AgentNodeData(
                    system_instructions="",
                    style_guide="",
                    model=ModelParams(provider="openai", model_id="model-y"),
                    context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
                    tools=ToolsConfig(attached=[], policy="Auto"),
                    structured_output={},
                    safety={"policy_ref": "default"},
                    telemetry={"labels": {}, "emit_usage": True},
                ),
            ),
            ToolNode(
                id="t1",
                label="web_search",
                data=ToolNodeData(name="web_search", tool_id="tool:web-search", version="0", parameter_overrides={}),
            ),
        ],
        edges=[
            Edge(id="e1", from_="M", to="X"),
            Edge(id="e2", from_="X", to="M"),
            Edge(id="e3", from_="M", to="Y"),
            Edge(id="e4", from_="Y", to="M"),
        ],
    )

    reg = ToolRegistry()
    tool = StubTool()
    reg.register(ToolEntry(tool_id="tool:web-search", version=None, impl=tool))

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "router-model":
            return m
        if mid == "model-x":
            return px
        return py

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=reg)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})

    # Ensure tool message exists before the next selection/speaker turn
    assert any(m.role == "tool" for m in res.messages)
    # And final assistant content present
    assert any(m.role == "assistant" and isinstance(m.content, str) and m.content.startswith("[Agent: Agent X]") for m in res.messages)

