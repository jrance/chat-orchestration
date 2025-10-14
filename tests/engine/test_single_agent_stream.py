from __future__ import annotations

from collections.abc import Iterator
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
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse


class StubStreamProvider(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - not used
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Hello"), finish_reason="stop"
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="Hel")
        yield ChatChunk(type="content.delta", text_delta="lo")
        yield ChatChunk(type="message.end", finish_reason="stop")


def _cfg_no_tools() -> GraphConfig:
    agent = AgentNode(
        id="agent-1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(
                history_window={"mode": "LastN", "n": 5},
                inject_org_preamble=True,
                vars={},
            ),
            tools=ToolsConfig(attached=[], policy="None"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    return GraphConfig(
        meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()),
        nodes=[agent],
        edges=[],
    )


def test_stream_basic_tokens_passthrough() -> None:
    engine = OrchestrationEngine()

    def resolver(_: Any) -> LLMProvider:
        return StubStreamProvider()

    compiled = engine.compile(_cfg_no_tools(), provider_resolver=resolver)
    chunks = list(engine.stream_execute(compiled, run_input={"messages": "hi"}))

    # Expect the same sequence
    assert len(chunks) == 4
    assert chunks[0].type == "role" and chunks[0].role == "assistant"
    assert chunks[1].type == "content.delta" and chunks[1].text_delta == "Hel"
    assert chunks[2].type == "content.delta" and chunks[2].text_delta == "lo"
    assert chunks[3].type == "message.end" and chunks[3].finish_reason == "stop"
