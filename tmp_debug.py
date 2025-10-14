import json
from pathlib import Path
from codeless_orchestrator.engine.executor import OrchestrationEngine
from codeless_orchestrator.tools.registry import ToolRegistry, ToolEntry
from codeless_orchestrator.providers.base import LLMProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest, ChatResponse, ToolCall
from codeless_orchestrator.tools.base import BaseTool

class StubTool(BaseTool):
    name = "web_search"
    description = "stub"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    def __init__(self):
        self.calls = []
    def invoke(self, args):
        self.calls.append(dict(args))
        return {"ok": True}

tool = StubTool()
reg = ToolRegistry()
reg.register(ToolEntry(tool_id="tool:web-search", impl=tool, version=None))

class StubProvider(LLMProvider):
    def __init__(self):
        self.turn = 0
    def chat(self, req: ChatRequest) -> ChatResponse:
        self.turn += 1
        if self.turn == 1:
            tc = ToolCall(id="tc-1", name="web_search", arguments_json=json.dumps({"q": "hello", "max_results": 10}))
            return ChatResponse(message=ChatMessage(role="assistant", content="", tool_calls=[tc]), finish_reason="tool_calls")
        return ChatResponse(message=ChatMessage(role="assistant", content="done"), finish_reason="stop")
    def stream(self, req: ChatRequest):
        if False:
            yield

root = Path('examples/configs/single_agent_with_tool.json')
cfg = json.loads(root.read_text('utf-8'))
from codeless_orchestrator.config.loader import load_graph
cfgm, _ = load_graph(cfg, validate=True)
eng = OrchestrationEngine()
compiled = eng.compile(cfgm, provider_resolver=lambda _mp: StubProvider(), registry=reg)
res = eng.execute(compiled, {"messages":[ChatMessage(role="user", content="hi")], "metadata":{}})
print('steps', res.steps)
print('calls', tool.calls)
