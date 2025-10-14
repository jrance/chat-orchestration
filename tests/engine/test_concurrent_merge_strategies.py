from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

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
from codeless_orchestrator.engine.graph_builders import concurrent as conc_builder
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


class SlowProvider(LLMProvider):
    def __init__(self, text: str, delay_s: float) -> None:
        self.text = text
        self.delay_s = delay_s

    def chat(self, req: ChatRequest) -> ChatResponse:
        time.sleep(self.delay_s)
        return ChatResponse(message=ChatMessage(role="assistant", content=self.text), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta=self.text)
        yield ChatChunk(type="message.end", finish_reason="stop")


def _graph_with_join() -> GraphConfig:
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
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    d = AgentNode(
        id="D",
        label="Agent D",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-d"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(
        meta=Meta(id="m1", name="concurrent", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c, d],
        edges=[
            Edge(id="e1", from_="A", to="B"),
            Edge(id="e2", from_="A", to="C"),
            Edge(id="e3", from_="B", to="D"),
            Edge(id="e4", from_="C", to="D"),
        ],
    )


def test_merge_first_finish_infers_on_join() -> None:
    cfg = _graph_with_join()

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-b":
            return SlowProvider("slow", 0.1)
        if mid == "model-c":
            return SlowProvider("fast", 0.01)
        # For D
        return SlowProvider("D", 0.0)

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    res = engine.execute(
        compiled,
        run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}},
    )
    # First-finish should choose C's content
    assert res.messages[-2].content == "fast"  # merged result before D
    # Discarded list contains B under metadata.concurrent
    meta = res.messages  # not used; check executor metadata
    # We cannot access state here; ensure no exception and message chain continues
    assert res.messages[-1].content == "D"


def test_merge_vote_length(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GraphConfig(
        meta=Meta(id="m1", name="concurrent", version="0", metadata=MetaMetadata()),
        nodes=[
            AgentNode(
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
            ),
            AgentNode(
                id="B",
                label="Agent B",
                data=AgentNodeData(
                    system_instructions="",
                    style_guide="",
                    model=ModelParams(provider="openai", model_id="model-b"),
                    context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=True, vars={}),
                    tools=ToolsConfig(attached=[], policy="Auto"),
                    structured_output={},
                    safety={"policy_ref": "default"},
                    telemetry={"labels": {}, "emit_usage": True},
                ),
            ),
            AgentNode(
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
            ),
        ],
        edges=[Edge(id="e1", from_="A", to="B"), Edge(id="e2", from_="A", to="C")],
    )

    # Force vote(length) for this test
    monkeypatch.setattr(conc_builder, "_select_merge_strategy", lambda *_args, **_kwargs: "vote")

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-b":
            return SlowProvider("short", 0.01)
        if mid == "model-c":
            return SlowProvider("this is longer text", 0.01)
        return SlowProvider("A", 0.0)

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})
    assert res.messages[-1].content == "this is longer text"


class SleeperTool(BaseTool):
    name = "sleeper"
    description = "sleeps"
    parameters = {
        "type": "object",
        "properties": {"seconds": {"type": "number"}},
        "required": ["seconds"],
        "additionalProperties": False,
    }

    active = 0
    max_active = 0
    lock = threading.Lock()

    def invoke(self, args: dict[str, Any]) -> Any:  # pragma: no cover - timing logic
        secs = float(args["seconds"])  # type: ignore[assignment]
        with SleeperTool.lock:
            SleeperTool.active += 1
            SleeperTool.max_active = max(SleeperTool.max_active, SleeperTool.active)
        try:
            time.sleep(secs)
        finally:
            with SleeperTool.lock:
                SleeperTool.active -= 1
        return {"ok": True}


class ProviderWithTools(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        # Return three tool calls
        calls = [
            ToolCall(id="t1", name="sleeper", arguments_json=json.dumps({"seconds": 0.05})),
            ToolCall(id="t2", name="sleeper", arguments_json=json.dumps({"seconds": 0.05})),
            ToolCall(id="t3", name="sleeper", arguments_json=json.dumps({"seconds": 0.05})),
        ]
        return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=calls), finish_reason="tool_calls")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="message.end", finish_reason="stop")


def test_within_branch_tool_parallelism_preserves_order_and_throttle() -> None:
    # Reset counters
    SleeperTool.active = 0
    SleeperTool.max_active = 0

    # Two-branch concurrent stage to focus on tool parallelism within one branch
    a = AgentNode(
        id="A",
        label="Agent A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-a"),
            context=AgentContext(history_window={"mode": "LastN", "n": 2}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["tool:sleeper"], policy="Auto", parallelism=2),
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
            context=AgentContext(history_window={"mode": "LastN", "n": 2}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto", parallelism=2),
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
            context=AgentContext(history_window={"mode": "LastN", "n": 2}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["tool:sleeper"], policy="Auto", parallelism=2),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(
        meta=Meta(id="m1", name="concurrent", version="0", metadata=MetaMetadata()),
        nodes=[a, b, c, ToolNode(id="t", label="sleeper", data=ToolNodeData(name="sleeper", tool_id="tool:sleeper", version="1", parameter_overrides={}))],
        edges=[Edge(id="e1", from_="A", to="B"), Edge(id="e2", from_="A", to="C")],
    )

    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:sleeper", version="1", impl=SleeperTool()))

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        if mid == "model-c":
            return ProviderWithTools()
        return SlowProvider(mid, 0.0)

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=reg)
    res = engine.execute(
        compiled,
        run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}},
    )

    # At most 2 active concurrently
    assert SleeperTool.max_active <= 2
    # Tool messages appended in order
    tool_msgs = [m for m in res.messages if m.role == "tool"]
    assert [tm.tool_call_id for tm in tool_msgs] == ["t1", "t2", "t3"]


def test_cross_branch_throttle_parallelism_one() -> None:
    # Three branches with parallelism=1 -> sequential across branches
    a = AgentNode(
        id="A",
        label="Agent A",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="model-a"),
            context=AgentContext(history_window={"mode": "LastN", "n": 2}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=[], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    delays = {"B": 0.03, "C": 0.03, "D": 0.03}
    nodes = [a]
    for aid in ["B", "C", "D"]:
        nodes.append(
            AgentNode(
                id=aid,
                label=f"Agent {aid}",
                data=AgentNodeData(
                    system_instructions="",
                    style_guide="",
                    model=ModelParams(provider="openai", model_id=f"model-{aid.lower()}"),
                    context=AgentContext(history_window={"mode": "LastN", "n": 2}, inject_org_preamble=True, vars={}),
                    tools=ToolsConfig(attached=[], policy="Auto", parallelism=1),
                    structured_output={},
                    safety={"policy_ref": "default"},
                    telemetry={"labels": {}, "emit_usage": True},
                ),
            )
        )
    cfg = GraphConfig(
        meta=Meta(id="m1", name="concurrent", version="0", metadata=MetaMetadata()),
        nodes=nodes,
        edges=[
            Edge(id="e1", from_="A", to="B"),
            Edge(id="e2", from_="A", to="C"),
            Edge(id="e3", from_="A", to="D"),
        ],
    )

    def resolver(mp: Any) -> LLMProvider:
        mid = getattr(mp, "model_id")
        for aid, delay in delays.items():
            if mid == f"model-{aid.lower()}":
                return SlowProvider(aid, delay)
        return SlowProvider("A", 0.0)

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    start = time.time()
    _ = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="go")], "metadata": {}})
    elapsed = time.time() - start
    # Expect roughly the sum of delays (0.09), allow slack
    assert elapsed >= 0.08
