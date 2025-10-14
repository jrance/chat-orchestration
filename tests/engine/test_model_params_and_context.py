from __future__ import annotations

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
    ToolsConfig,
)
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatChunk, ChatMessage, ChatRequest, ChatResponse


class CaptureProvider(LLMProvider):
    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.seen.append(req)
        # Return a trivial assistant message
        return ChatResponse(message=ChatMessage(role="assistant", content="ok"), finish_reason="stop")

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="ok")
        yield ChatChunk(type="message.end", finish_reason="stop")


def test_model_params_and_context_prompt_assembly() -> None:
    # Build a single-agent graph
    agent = AgentNode(
        id="agent-ctx",
        label="Context Agent",
        data=AgentNodeData(
            system_instructions="Hello {dept}",
            style_guide="Use {tone} tone.",
            model=ModelParams(
                provider="openai",
                model_id="gpt-4o-mini",
                temperature=0.3,
                top_p=1,
                max_tokens=800,
                stop=["\n\n"],
                seed=123,
                json_mode_enabled=False,
            ),
            context=AgentContext(
                history_window={"mode": "LastN", "n": 2},
                inject_org_preamble=True,
                vars={},
            ),
            tools=ToolsConfig(policy="None"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(
        meta=Meta(id="m2", name="test", version="0", metadata=MetaMetadata()),
        nodes=[agent],
        edges=[],
    )

    provider = CaptureProvider()

    def resolver(_: Any) -> LLMProvider:
        return provider

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver)

    # History longer than 2 (excluding system)
    history = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="tool", content="t1"),
        ChatMessage(role="user", content="u2"),
    ]

    result = engine.execute(
        compiled,
        run_input={
            "messages": history,
            "metadata": {"org_preamble": "Org Preamble"},
            "vars": {"dept": "Legal", "tone": "formal"},
        },
    )

    # Provider saw a request with normalized params and assembled prompt
    assert provider.seen, "provider.chat was not called"
    req = provider.seen[0]
    assert req.model == "gpt-4o-mini"
    assert req.temperature == 0.3
    assert req.top_p == 1
    assert req.max_tokens == 800
    assert req.stop == ["\n\n"]
    assert req.seed == 123
    assert req.json_mode_enabled is False

    assert req.messages, "no messages forwarded"
    first = req.messages[0]
    assert first.role == "system"
    text = first.content if isinstance(first.content, str) else ""
    assert "Org Preamble" in text
    assert "Hello Legal" in text
    assert "Use formal tone." in text

    roles = [m.role for m in req.messages[1:]]
    # Only last 2 non-system messages should be present: tool, user
    assert roles == ["tool", "user"]

    # Engine result sanity
    assert result.steps == 1

