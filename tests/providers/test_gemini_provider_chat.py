from __future__ import annotations

from typing import Any

from codeless_orchestrator.providers.gemini import GeminiProvider
from codeless_orchestrator.providers.types import ChatMessage, ChatRequest


class _Part:
    def __init__(
        self, text: str | None = None, function_call: dict[str, Any] | None = None
    ) -> None:
        self.text = text
        if function_call is not None:
            # Support both snake and camel used by transform
            self.function_call = function_call
            self.functionCall = function_call


class _Content:
    def __init__(self, role: str, parts: list[_Part]) -> None:
        self.role = role
        self.parts = parts


class _Candidate:
    def __init__(self, content: _Content, finish_reason: str | None = None) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.finishReason = finish_reason


class StubResp:
    def __init__(self, candidates: list[_Candidate]) -> None:
        self.candidates = candidates
        self.usage_metadata = {
            "prompt_token_count": 3,
            "candidates_token_count": 2,
            "total_token_count": 5,
        }


class StubGenerativeModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.last_kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> Any:  # type: ignore[override]
        self.last_kwargs = kwargs
        # Produce a default simple text response
        parts = [_Part(text="Hi")]
        content = _Content(role="model", parts=parts)
        cand = _Candidate(content=content, finish_reason="STOP")
        return StubResp([cand])


def test_gemini_provider_chat_text() -> None:
    model = StubGenerativeModel()
    provider = GeminiProvider(model_client=model)

    req = ChatRequest(
        model="gemini-1.5-pro",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.3,
    )
    res = provider.chat(req)

    # Ensure provider passed generation_config and contents history
    gen_cfg = model.last_kwargs.get("generation_config")
    assert gen_cfg and gen_cfg["temperature"] == 0.3
    contents = model.last_kwargs.get("contents")
    assert isinstance(contents, list) and contents[0]["role"] == "user"

    # Validate response mapping
    assert res.message.content == "Hi"
    assert res.finish_reason == "stop"
    assert res.usage and res.usage.total_tokens == 5


def test_gemini_provider_chat_function_call() -> None:
    class _ModelWithCall(StubGenerativeModel):
        def generate_content(self, **kwargs: Any) -> Any:  # type: ignore[override]
            self.last_kwargs = kwargs
            parts = [_Part(function_call={"name": "web_search", "args": {"q": "x"}})]
            content = _Content(role="model", parts=parts)
            cand = _Candidate(content=content, finish_reason="STOP")
            return StubResp([cand])

    model = _ModelWithCall()
    provider = GeminiProvider(model_client=model)

    req = ChatRequest(model="gemini-1.5-pro", messages=[ChatMessage(role="user", content="Hi")])
    res = provider.chat(req)
    assert res.message.tool_calls is not None
    assert len(res.message.tool_calls) == 1
    tc = res.message.tool_calls[0]
    assert tc.name == "web_search"
    assert tc.arguments_json == '{"q": "x"}' or tc.arguments_json == '{"q":"x"}'
