from __future__ import annotations

from typing import Any, Iterator, Optional

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
from codeless_orchestrator.engine.compiler import compile_single_agent
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
        self.last_req: Optional[ChatRequest] = None

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.last_req = req
        # Return a simple assistant message with no tool calls
        return ChatResponse(message=ChatMessage(role="assistant", content="hi"), finish_reason="stop", usage=None)

    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:  # pragma: no cover - not used here
        self.last_req = req
        yield ChatChunk(type="role", role="assistant")
        yield ChatChunk(type="content.delta", text_delta="hi")
        yield ChatChunk(type="message.end", finish_reason="stop")


class StubTool(BaseTool):
    name = "web_search"
    description = "stub"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: dict[str, Any]) -> Any:  # pragma: no cover - not called in this test
        return {"ok": True}


def _sample_config() -> GraphConfig:
    agent = AgentNode(
        id="agent-1",
        label="Agent",
        data=AgentNodeData(
            system_instructions="You are a helpful agent.",
            style_guide="",
            model=ModelParams(provider="openai", model_id="gpt-4o-mini"),
            context=AgentContext(history_window={"mode": "LastN", "n": 10}, inject_org_preamble=True, vars={}),
            tools=ToolsConfig(attached=["tool:web-search"], policy="Auto"),
            structured_output={},
            safety={"policy_ref": "default"},
            telemetry={"labels": {}, "emit_usage": True},
        ),
    )
    cfg = GraphConfig(
        meta=Meta(id="m1", name="test", version="0", metadata=MetaMetadata()),
        nodes=[agent],
        edges=[],
    )
    return cfg


def test_compile_and_request_build_includes_tools() -> None:
    # Registry with stub tool
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=StubTool(), version=None))

    stub = StubProvider()

    def resolver(_: Any) -> LLMProvider:
        return stub

    compiled = compile_single_agent(_sample_config(), provider_resolver=resolver, registry=reg)

    # Run one agent turn to trigger provider.chat
    app = compiled.graph
    init_state = {
        "messages": [ChatMessage(role="user", content="hello")],
        "metadata": {},
        "tool_calls": None,
        "steps": 0,
    }
    out_state = app.invoke(init_state)  # type: ignore[attr-defined]

    assert isinstance(out_state, dict)
    assert stub.last_req is not None
    assert stub.last_req.tools is not None
    # One tool spec should be present and named 'web_search'
    assert len(stub.last_req.tools) == 1
    assert stub.last_req.tools[0].name == "web_search"
    # Ensure compiled artifact has the tool name mapping
    assert "web_search" in compiled.tools_attached
