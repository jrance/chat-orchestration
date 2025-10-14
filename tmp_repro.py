import json
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.tools.registry import ToolRegistry, ToolEntry, register_default_tools, default_registry
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse
from codeless_orchestrator.config.loader import load_graph

CONFIG = {
  "meta": {
    "id": "f6ed760b-9fa2-40f9-8a1a-7d492292da3b",
    "name": "Untitled Graph",
    "version": "1.0.0",
    "metadata": {"showToolNodesOnCanvas": True}
  },
  "nodes": [
    {
      "id": "0b523619-2a84-4b5d-8ab4-77ad8a969b13",
      "kind": "agent.codeless",
      "label": "Policy Agent",
      "data": {
        "systemInstructions": "Answer the user's question",
        "styleGuide": "",
        "model": {"provider": "openai", "modelId": "gpt-4o", "temperature": 0.3, "topP": 1, "maxTokens": 800, "seed": None, "stop": [], "jsonModeEnabled": False},
        "context": {"historyWindow": {"mode": "LastN", "n": 10}, "injectOrgPreamble": True, "vars": {}},
        "tools": {"policy": "Auto", "timeoutMs": 10000, "maxCallsPerTurn": 0, "parallelism": 10, "redactPII": True, "attached": ["93d1bbce-abd8-4db9-a90d-80a8279d5e40"]},
        "structuredOutput": {"enabled": False, "schema": {}, "onViolation": "RetryAndRepair", "maxRepairAttempts": 2, "postProcess": {"normalizeWhitespace": True, "ensureMarkdown": True}},
        "safety": {"policyRef": "enterprise-v3", "onBlock": "Refuse", "piiRedaction": True, "promptInjectionDefense": True},
        "telemetry": {"labels": {}, "emitUsage": True}
      }
    },
    {
      "id": "93d1bbce-abd8-4db9-a90d-80a8279d5e40",
      "kind": "tool",
      "label": "Policy Search",
      "data": {"name": "Tool", "toolId": "tool:web-search", "version": "1.4.3", "parameterOverrides": {}}
    }
  ],
  "edges": [
    {"id": "f5821851-232e-4d1d-9b4a-5d840b9b1e7a", "from": "0b523619-2a84-4b5d-8ab4-77ad8a969b13", "to": "93d1bbce-abd8-4db9-a90d-80a8279d5e40"}
  ]
}

class StubProvider(LLMProvider):
    def chat(self, req: ChatRequest) -> ChatResponse:
        print("provider.chat tools:", None if not req.tools else [t.name for t in req.tools])
        print("provider.chat tool_choice:", req.tool_choice)
        return ChatResponse(message=ChatMessage(role="assistant", content="ok"), finish_reason="stop")
    def stream(self, req: ChatRequest):
        yield from ()

eng = OrchestrationEngine()

# ensure registry has web-search
register_default_tools(default_registry)

cfg, _ = load_graph(CONFIG, validate=True)
compiled = eng.compile(cfg, provider_resolver=lambda mp: StubProvider())
# monkeypatch provider to stub
compiled = compiled.__class__(
    agent_id=compiled.agent_id,
    graph=compiled.graph,
    provider=StubProvider(),
    tools_attached=compiled.tools_attached,
    tool_bindings_by_fn=compiled.tool_bindings_by_fn,
    tool_config=compiled.tool_config,
    model_params=compiled.model_params,
    structured_output=compiled.structured_output,
    safety=compiled.safety,
)
res = eng.execute(compiled, {"messages": [ChatMessage(role="user", content="test")]})
print("done")

