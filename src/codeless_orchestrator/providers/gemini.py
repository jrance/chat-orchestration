from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .base import LLMProvider
from .exceptions import ProviderError
from .settings import get_gemini_api_key, get_gemini_timeout
from .transform_gemini import (
    from_gemini_candidate,
    map_finish_reason,
    to_gemini_generation_config,
    to_gemini_history,
    to_gemini_system_instruction,
    to_gemini_tool_config,
    to_gemini_tools,
    usage_from_gemini,
)
from .types import ChatChunk, ChatRequest, ChatResponse


def _build_model(
    model_name: str,
    *,
    api_key: str | None,
    system_instruction: str | None,
    tools: list[dict[str, Any]] | None,
    tool_config: dict[str, Any] | None,
) -> Any:
    try:
        import google.generativeai as genai
    except Exception as e:  # pragma: no cover - import error path
        raise ProviderError(f"Failed to import google-generativeai: {e}") from e

    key = api_key or get_gemini_api_key()
    if key:
        try:
            genai.configure(api_key=key)
        except Exception as e:  # pragma: no cover - unlikely
            raise ProviderError(f"Failed to configure Gemini client: {e}") from e
    try:
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
            tools=tools or None,
            tool_config=tool_config or None,  # type: ignore[arg-type]
        )
    except Exception as e:  # pragma: no cover - construction failure
        raise ProviderError(f"Failed to build GenerativeModel: {e}") from e
    return model


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_client: Any | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout if timeout is not None else get_gemini_timeout()
        # In tests we pass a stub model, otherwise we construct per-call
        self._model_client = model_client

    # ------------------- Non-streaming -------------------
    def chat(self, req: ChatRequest) -> ChatResponse:
        system_instruction = to_gemini_system_instruction(req.messages)
        history = to_gemini_history(req.messages)
        tools = to_gemini_tools(req.tools or []) if req.tools else None
        tool_config = to_gemini_tool_config(req.tool_choice)
        generation_config = to_gemini_generation_config(req)

        # Per-request timeout override precedence: req.timeout > provider timeout
        timeout = req.timeout if req.timeout is not None else self._timeout
        request_options = {"timeout": timeout} if timeout is not None else {}

        # Build or reuse model client
        model = self._model_client or _build_model(
            req.model,
            api_key=self._api_key,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
        )

        try:
            resp = model.generate_content(
                contents=history,
                generation_config=generation_config or None,
                safety_settings=None,
                request_options=request_options or None,
            )
        except Exception as e:
            raise _to_provider_error(e) from e

        try:
            candidates = getattr(resp, "candidates", []) or []
            cand = candidates[0] if candidates else None
            message = from_gemini_candidate(cand)
            # Map finish reason from candidate if available
            finish_raw = None
            if cand is not None:
                finish_raw = (
                    getattr(cand, "finish_reason", None)
                    or getattr(cand, "finishReason", None)
                )
            finish_reason = map_finish_reason(finish_raw) if finish_raw else None
            usage = usage_from_gemini(resp)
            # Forward usage to telemetry if present
            try:
                tele = (req.metadata or {}).get("_telemetry") if req.metadata else None
                if tele is not None and usage is not None and hasattr(tele, "usage"):
                    tele.usage(
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        getattr(usage, "total_tokens", None),
                    )
            except Exception:
                pass
            return ChatResponse(message=message, finish_reason=finish_reason, usage=usage, raw=resp)
        except Exception as e:
            raise ProviderError(f"Failed to parse Gemini response: {e}") from e

    # ------------------- Streaming -------------------
    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        system_instruction = to_gemini_system_instruction(req.messages)
        history = to_gemini_history(req.messages)
        tools = to_gemini_tools(req.tools or []) if req.tools else None
        tool_config = to_gemini_tool_config(req.tool_choice)
        generation_config = to_gemini_generation_config(req)

        timeout = req.timeout if req.timeout is not None else self._timeout
        request_options = {"timeout": timeout} if timeout is not None else {}

        model = self._model_client or _build_model(
            req.model,
            api_key=self._api_key,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
        )

        try:
            events = model.generate_content(
                contents=history,
                generation_config=generation_config or None,
                safety_settings=None,
                request_options=request_options or None,
                stream=True,
            )
        except Exception as e:  # pragma: no cover - network exceptions via SDK
            raise _to_provider_error(e) from e

        # Track tool call stream state
        started: dict[int, bool] = {}
        tool_ids: dict[int, str | None] = {}
        tool_names: dict[int, str | None] = {}

        for ev in events:
            # We accept two kinds of stub shapes:
            # 1) ev.role set to "model" at start
            role = getattr(ev, "role", None) or getattr(getattr(ev, "content", None), "role", None)
            if role:
                # Map gemini 'model' to our 'assistant'
                yield ChatChunk(type="role", role="assistant")

            # Text delta convenience: chunk.text or first candidate/text_delta
            text = getattr(ev, "text", None)
            if text:
                yield ChatChunk(type="content.delta", text_delta=text)

            # Function call delta: ev.function_call updates or ev.tool_calls list
            calls = getattr(ev, "function_calls", None) or getattr(ev, "tool_calls", None)
            if calls:
                for c in calls:
                    index = getattr(c, "index", None)
                    if index is None:
                        index = 0
                    name = getattr(c, "name", None)
                    call_id = getattr(c, "id", None)
                    args_piece = getattr(c, "arguments", None)

                    # Start event once we see name/id
                    if not started.get(index) and (name or call_id):
                        started[index] = True
                        tool_ids[index] = call_id
                        tool_names[index] = name
                        yield ChatChunk(
                            type="tool_call.start",
                            tool_call_index=index,
                            tool_call_id=call_id,
                            tool_name=name,
                        )

                    if args_piece:
                        yield ChatChunk(
                            type="tool_call.delta",
                            tool_call_index=index,
                            tool_call_id=tool_ids.get(index),
                            tool_name=tool_names.get(index),
                            arguments_delta=args_piece,
                        )

            # Candidate-oriented streaming (alternative stub): ev.candidates[0].content.parts
            candidates = getattr(ev, "candidates", []) or []
            if candidates:
                cand = candidates[0]
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None)
                # If a role appears via candidate content on the first chunk
                r2 = getattr(content, "role", None)
                if r2 and not role:
                    yield ChatChunk(type="role", role="assistant")
                if parts:
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            yield ChatChunk(type="content.delta", text_delta=t)
                        fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
                        if fc:
                            name = getattr(fc, "name", None)
                            args_text = None
                            args = getattr(fc, "args", None)
                            if args is not None:
                                try:
                                    import json

                                    args_text = json.dumps(args)
                                except Exception:
                                    args_text = None
                            # Ensure start
                            idx = 0
                            if not started.get(idx) and (name or None):
                                started[idx] = True
                                tool_ids[idx] = None
                                tool_names[idx] = name
                                yield ChatChunk(
                                    type="tool_call.start",
                                    tool_call_index=idx,
                                    tool_call_id=None,
                                    tool_name=name,
                                )
                            if args_text:
                                yield ChatChunk(
                                    type="tool_call.delta",
                                    tool_call_index=0,
                                    tool_call_id=tool_ids.get(0),
                                    tool_name=tool_names.get(0),
                                    arguments_delta=args_text,
                                )

            # Finish reason on chunk end
            finish_raw = getattr(ev, "finish_reason", None) or getattr(
                ev, "finishReason", None
            )
            if finish_raw:
                yield ChatChunk(
                    type="message.end",
                    finish_reason=map_finish_reason(finish_raw),
                    usage=None,
                )

        # If stream ended without explicit finish reason, still close message
        # (leave finish_reason None)
        # yield ChatChunk(type="message.end", finish_reason=None, usage=None)


def _to_provider_error(err: Exception) -> ProviderError:
    # Try to propagate status information when wrapped by google core errors
    status_code: int | None = None
    code: str | None = None
    message = str(err)
    try:
        # google.api_core.exceptions.GoogleAPIError has .code and .message
        status_code = getattr(err, "code", None) or getattr(
            getattr(err, "response", None), "status_code", None
        )
        message_obj = getattr(err, "message", None)
        if message_obj:
            message = str(message_obj)
    except Exception:  # pragma: no cover - best effort only
        message = str(err)
    return ProviderError(message=message, status_code=status_code, code=code)
