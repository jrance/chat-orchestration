from __future__ import annotations

from typing import Any

from codeless_orchestrator.providers.openai import OpenAIProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest


class StubChatCompletions:
    def __init__(self, parent: StubClient) -> None:
        self._parent = parent

    def create(self, stream: bool = False, **kwargs: Any) -> Any:  # type: ignore[override]
        self._parent.last_stream = stream
        self._parent.last_kwargs = kwargs
        if stream:
            raise AssertionError("stream=True not expected here")

        # Build a stubbed response
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

    # parity with OpenAI.with_options but not used in tests
    def with_options(self, **_: Any) -> StubClient:  # pragma: no cover - not needed here
        return self


def test_openai_provider_chat_basic() -> None:
    client = StubClient()
    provider = OpenAIProvider(client=client)

    req = ChatRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.2,
    )
    res = provider.chat(req)

    assert client.last_stream is False
    assert client.last_kwargs["model"] == "gpt-4o-mini"
    assert isinstance(client.last_kwargs["messages"], list)
    assert client.last_kwargs["temperature"] == 0.2

    assert res.message.content == "Hi"
    assert res.finish_reason == "stop"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 3
    assert res.usage.completion_tokens == 2
    assert res.usage.total_tokens == 5


def test_openai_provider_chat_json_mode_flag() -> None:
    client = StubClient()
    provider = OpenAIProvider(client=client)

    req = ChatRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="{}")],
        json_mode_enabled=True,
    )
    provider.chat(req)
    assert client.last_kwargs.get("response_format") == {"type": "json_object"}
