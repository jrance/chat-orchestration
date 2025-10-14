from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.server import deps
from codeless_orchestrator.server.app import app
from codeless_orchestrator.tools.base import BaseTool
from codeless_orchestrator.tools.registry import ToolEntry, ToolRegistry


class StubWebSearchTool(BaseTool):
    name = "web_search"
    description = "stub web search"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"ok": True}


class StubProviderToolFirst(LLMProvider):
    def __init__(self) -> None:
        self.turn = 0

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            # First turn: request a web_search tool call with args; overrides should clamp max_results
            args = json.dumps({"q": "hello", "max_results": 10})
            tc = ToolCall(id="tc-1", name="web_search", arguments_json=args)
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        return ChatResponse(message=ChatMessage(role="assistant", content="Search complete"), finish_reason="stop")

    def stream(self, req: ChatRequest):  # pragma: no cover - not used here
        yield from ()


def test_single_agent_with_tool_and_overrides() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "examples" / "configs" / "single_agent_with_tool.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Override provider and registry
    tool = StubWebSearchTool()
    reg = ToolRegistry()
    reg.register(ToolEntry(tool_id="tool:web-search", impl=tool, version=None))

    def resolver(_: Any) -> LLMProvider:
        return StubProviderToolFirst()

    client = TestClient(app)
    app.dependency_overrides[deps.get_provider_resolver] = lambda: resolver
    app.dependency_overrides[deps.get_registry] = lambda: reg
    try:
        body = {"config": cfg, "input": {"text": "hello"}, "options": {"requestId": "e2e-1"}}
        resp = client.post("/execute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "e2e-1"
        # Tool executed once with overrides applied (max_results==3)
        assert len(tool.calls) == 1
        assert tool.calls[0]["q"] == "hello"
        assert tool.calls[0]["max_results"] == 3
        # Assistant replied
        msgs = data["messages"]
        assert any(m.get("role") == "assistant" and m.get("content") for m in msgs)
    finally:
        app.dependency_overrides.pop(deps.get_provider_resolver, None)
        app.dependency_overrides.pop(deps.get_registry, None)

