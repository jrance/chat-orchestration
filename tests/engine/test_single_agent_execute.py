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
from codeless_orchestrator.providers.types import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
)
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class StubProvider(LLMProvider):
    def __init__(self) -> None:
        self.turn = 0
        self.seen_requests: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        self.seen_requests.append(req)
        if self.turn == 1:
            # Return a tool call
            args = json.dumps({"q": "foo", "max_results": 10})
            tc = ToolCall(id="call_1", name="web_search", arguments_json=args)
            return ChatResponse(
                message=ChatMessage(role="assistant", content="", tool_calls=[tc]),
                finish_reason="tool_calls",
            )
        # Final assistant reply
        return ChatResponse(
            message=ChatMessage(role="assistant", content="done"), finish_reason="stop"
        )

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used here
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="done")
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
        # Record merged args
        self.calls.append(dict(args))
        # Return deterministic result
        return [{"title": "foo", "href": "https://example.com", "body": "bar", "source": "stub"}]


def _graph_config_with_tool() -> tuple[GraphConfig, ToolRegistry, StubWebSearchTool, StubProvider]:
    # Build agent and tool nodes
    tool_impl = StubWebSearchTool()
    registry = ToolRegistry()
    registry.register(ToolEntry(tool_id="tool:web-search", impl=tool_impl, version=None))

    agent = AgentNode(
        id="agent-1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="You are a helpful agent.",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(
                history_window={"mode": "LastN", "n": 10},
                inject_org_preamble=True,
                vars={},
            ),
            tools=ToolsConfig(
                attached=["tool:web-search"],
                policy="Auto",
                timeout_ms=5000,
                max_calls_per_turn=0,
            ),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    tool_node = ToolNode(
        id="tool-1",
        label="web_search",
        data=ToolNodeData(
            name="web_search",
            tool_id="tool:web-search",
            version="0",
            parameter_overrides={"max_results": 3},
        ),
    )
    cfg = GraphConfig(
        meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()),
        nodes=[agent, tool_node],
        edges=[],
    )
    provider = StubProvider()
    return cfg, registry, tool_impl, provider


def test_execute_with_tools_and_overrides() -> None:
    cfg, registry, tool_impl, provider = _graph_config_with_tool()

    def resolver(_: Any) -> LLMProvider:
        return provider

    engine = OrchestrationEngine()
    compiled = engine.compile(cfg, provider_resolver=resolver, registry=registry)

    result = engine.execute(
        compiled,
        run_input={"messages": [ChatMessage(role="user", content="search please")], "metadata": {}},
    )

    # 2 LLM steps: first returns tool call, second returns final assistant
    assert result.steps == 2
    # Tool invoked exactly once with merged args (max_results overridden to 3)
    assert len(tool_impl.calls) == 1
    assert tool_impl.calls[0]["q"] == "foo"
    assert tool_impl.calls[0]["max_results"] == 3
    # Final transcript contains the tool message and final assistant message
    messages = result.messages
    assert any(m.role == "tool" and getattr(m, "tool_call_id", None) == "call_1" for m in messages)
    assert messages[-1].role == "assistant" and (messages[-1].content == "done")
