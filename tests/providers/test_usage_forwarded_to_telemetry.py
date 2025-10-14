from __future__ import annotations

from typing import Any

from codeless_orchestrator.providers.openai import OpenAIProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest
from codeless_orchestrator.runtime.telemetry import InMemorySink, Telemetry


class StubChatCompletions:
    def __init__(self, parent: StubClient) -> None:
        self._parent = parent

    def create(self, stream: bool = False, **kwargs: Any) -> Any:  # type: ignore[override]
        self._parent.last_stream = stream
        self._parent.last_kwargs = kwargs
        if stream:
            raise AssertionError("stream=True not expected here")

        # Build a stubbed response with usage
        msg = type("Msg", (), {"role": "assistant", "content": "Hi"})()
        choice = type("Choice", (), {"message": msg, "finish_reason": "stop"})()
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        resp = type("Resp", (), {"choices": [choice], "usage": usage})()
        return resp


class StubClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": StubChatCompletions(self)})()
        self.last_kwargs: dict[str, Any] = {}
        self.last_stream: bool | None = None

    def with_options(self, **_: Any) -> StubClient:  # pragma: no cover - not needed here
        return self


def test_usage_forwarded_to_telemetry() -> None:
    client = StubClient()
    provider = OpenAIProvider(client=client)
    sink = InMemorySink()
    tele = Telemetry(sink=sink)

    req = ChatRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.2,
        metadata={"_telemetry": tele},
    )
    provider.chat(req)

    snap = tele.snapshot()
    assert snap.get("token_usage") is not None
    assert snap["token_usage"].get("total") == 5

