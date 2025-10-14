from __future__ import annotations

import json
from typing import Any
from collections.abc import Iterator

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
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class InjectTool(BaseTool):
    name = "inject"
    description = "returns sensitive data"
    parameters = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}

    def __init__(self) -> None:
        self.seen_args: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> Any:
        self.seen_args.append(dict(args))
        return {"got": args.get("key"), "secret": "4111 1111 1111 1111"}


class StubProvider(LLMProvider):
    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []
        self.turn = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        self.seen.append(req)
        if self.turn == 1:
            # Return a tool call with PII in args
            args = json.dumps({"key": "john.doe@example.com"})
            tc = ToolCall(id="t1", name="inject", arguments_json=args)
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        # Second turn returns final answer including a redaction candidate
        return ChatResponse(message=ChatMessage(role="assistant", content="Processed john@example.com"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield from ()


def test_safety_redaction_integration() -> None:
    # Build graph with safety enabled
    registry = ToolRegistry()
    tool = InjectTool()
    registry.register(ToolEntry(tool_id="tool:inject", impl=tool, version=None))
    agent = AgentNode(
        id="a1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 3}, inject_org_preamble=False, vars={}),
            tools=ToolsConfig(attached=["tool:inject"], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "enterprise-v3", "on_block": "Refuse", "pii_redaction": True, "prompt_injection_defense": True},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(meta=Meta(id="m", name="t", version="0", metadata=MetaMetadata()), nodes=[agent, ToolNode(id="tn", kind="tool", label="inject", data=ToolNodeData(name="inject", tool_id="tool:inject", version="0"))], edges=[])
    provider = StubProvider()

    def resolver(_: Any) -> LLMProvider:
        return provider

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=registry)
    res = engine.execute(compiled, run_input={"messages": [ChatMessage(role="user", content="hello; email me at john@example.com")], "metadata": {}})

    # Provider saw redacted input in request (no raw email)
    sent = json.dumps([m.__dict__ for m in provider.seen[0].messages])
    assert "john@example.com" not in sent
    assert "[REDACTED:EMAIL]" in sent
    # Tool received original args (not redacted)
    assert tool.seen_args and tool.seen_args[0]["key"] == "john.doe@example.com"
    # Transcript contains redacted tool message result
    assert any(m.role == "tool" and "[REDACTED:EMAIL]" in (m.content if isinstance(m.content, str) else "") for m in res.messages)
