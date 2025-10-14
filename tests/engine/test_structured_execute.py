from __future__ import annotations

import json
from typing import Any

from codeless_orchestrator.config.schema import (
    AgentContext,
    AgentNode,
    AgentNodeData,
    GraphConfig,
    Meta,
    MetaMetadata,
    ModelParams,
    ToolsConfig,
)
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse


class StubProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.seen.append(req)
        text = self.replies.pop(0) if self.replies else "{}"
        return ChatResponse(message=ChatMessage(role="assistant", content=text), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used
        yield from ()


def test_structured_execute_with_repair() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    agent = AgentNode(
        id="a1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=False, vars={}),
            tools=ToolsConfig(policy="None"),
            structured_output={"enabled": True, "schema": schema, "onViolation": "RetryAndRepair", "maxRepairAttempts": 1},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(meta=Meta(id="m", name="t", version="0", metadata=MetaMetadata()), nodes=[agent], edges=[])

    # First reply invalid, second valid
    provider = StubProvider([json.dumps({}), json.dumps({"answer": "yes"})])

    def resolver(_: Any) -> LLMProvider:
        return provider

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="q")], "metadata": {}})

    # Steps reflect two calls (repair)
    assert res.steps == 2
    # Final assistant message content is compact JSON
    final = res.messages[-1]
    assert final.role == "assistant"
    assert final.content == "{" + '"answer":"yes"' + "}"

