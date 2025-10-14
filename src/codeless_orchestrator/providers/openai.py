from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from .base import LLMProvider
from .exceptions import ProviderError
from .settings import get_openai_client_kwargs, get_httpx_proxies, get_httpx_verify
from .transform_openai import (
    from_openai_message,
    to_openai_messages,
    to_openai_tools,
    usage_from_openai,
)
from .types import ChatChunk, ChatRequest, ChatResponse


def _build_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover - import error path
        raise ProviderError(f"Failed to import openai client: {e}") from e

    kwargs = get_openai_client_kwargs(
        api_key=api_key, base_url=base_url, organization=organization, timeout=timeout
    )
    # Optional debug of env resolution (no secrets printed). Enable with ORCH_DEBUG_ENV=1
    if os.environ.get("ORCH_DEBUG_ENV") == "1":  # pragma: no cover - debug only
        ak = os.getenv("OPENAI_API_KEY")
        bu = os.getenv("OPENAI_BASE_URL")
        oo = os.getenv("OPENAI_ORG")
        tv = os.getenv("OPENAI_TIMEOUT")
        print("[orchestrator] OpenAI env -> api_key?", bool(ak), "base_url?", bool(bu), "org?", bool(oo), "timeout?", bool(tv), flush=True)
        # Also show what kwargs we finally pass (mask api key)
        dbg = dict(kwargs)
        if "api_key" in dbg and dbg["api_key"]:
            dbg["api_key"] = "***" + str(dbg["api_key"])[-4:]
        print("[orchestrator] OpenAI client kwargs:", dbg, flush=True)
    # OpenAI initializer doesn't accept timeout directly; with_options handles it.
    timeout_value = kwargs.pop("timeout", None)
    # Optional proxy support via httpx client
    http_client = None
    try:
        proxy = get_httpx_proxies()
        verify = get_httpx_verify()
        if proxy:
            import httpx  # lazy import

            if os.environ.get("ORCH_DEBUG_ENV") == "1":
                print("[orchestrator] OpenAI proxy enabled", "verify=", verify, flush=True)
            if verify is None:
                http_client = httpx.Client(proxy=proxy)
            else:
                http_client = httpx.Client(proxy=proxy, verify=verify)
        else:
            if os.environ.get("ORCH_DEBUG_ENV") == "1":
                print("[orchestrator] OpenAI proxy disabled", flush=True)
    except Exception as e:  # Broad except to translate into ProviderError
        http_client = None

    client = OpenAI(http_client=http_client, **kwargs)
    if timeout_value is not None:
        client = client.with_options(timeout=timeout_value)
    return client


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = client or _build_client(
            api_key=api_key, base_url=base_url, organization=organization, timeout=timeout
        )

    # ------------------- Non-streaming -------------------
    def chat(self, req: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": to_openai_messages(req.messages),
        }
        if req.tools:
            kwargs["tools"] = to_openai_tools(req.tools)
        if req.tool_choice is not None:
            # Accept "auto", "none", or a dict
            kwargs["tool_choice"] = req.tool_choice  # forwarded as-is
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.stop is not None:
            kwargs["stop"] = req.stop
        if req.seed is not None:
            kwargs["seed"] = req.seed

        # Optional lightweight debug: print which high-level options are set
        if os.environ.get("ORCH_DEBUG_LLM_ARGS") == "1":  # pragma: no cover - debug only
            try:
                tool_names = [t.name for t in (req.tools or [])]
                print(
                    "[orchestrator] openai.chat args: tools=",
                    tool_names,
                    "tool_choice=",
                    req.tool_choice,
                    "temperature=",
                    req.temperature,
                    "top_p=",
                    req.top_p,
                    "max_tokens=",
                    req.max_tokens,
                    flush=True,
                )
            except Exception:
                pass

        # JSON mode support with optional schema enforcement
        if req.json_mode_enabled:
            if req.response_format_schema:
                # Attempt json_schema first; fallback to json_object if rejected by SDK
                try:
                    fmt = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": req.response_format_schema,
                            "strict": True,
                        },
                    }
                    kwargs["response_format"] = fmt
                    # Try the call early to catch InvalidRequest-like errors
                    resp = self._client.chat.completions.create(stream=False, **kwargs)
                except Exception:
                    # Fallback to generic JSON mode
                    kwargs["response_format"] = {"type": "json_object"}
                    resp = self._client.chat.completions.create(stream=False, **kwargs)
                # Parse and return
                try:
                    choice = resp.choices[0]
                    message = from_openai_message(choice.message)
                    finish_reason: str | None = getattr(choice, "finish_reason", None)
                    usage = usage_from_openai(resp)
                    return ChatResponse(message=message, finish_reason=finish_reason, usage=usage, raw=resp)
                except Exception as e:
                    raise ProviderError(f"Failed to parse OpenAI response: {e}") from e
            else:
                kwargs["response_format"] = {"type": "json_object"}

        # Per-request timeout override
        options: dict[str, Any] = {}
        if req.timeout is not None:
            options["timeout"] = req.timeout

        try:
            resp = self._client.chat.completions.create(stream=False, **kwargs, **options)
        except Exception as e:  # Broad except to translate into ProviderError
            raise _to_provider_error(e) from e

        try:
            # choices[0] must exist in successful calls
            choice = resp.choices[0]
            message = from_openai_message(choice.message)
            finish_reason: str | None = getattr(choice, "finish_reason", None)
            usage = usage_from_openai(resp)
            # Forward usage to telemetry if present in metadata
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
            raise ProviderError(f"Failed to parse OpenAI response: {e}") from e

    # ------------------- Streaming -------------------
    def stream(self, req: ChatRequest) -> Iterator[ChatChunk]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": to_openai_messages(req.messages),
            "stream": True,
        }
        if req.tools:
            kwargs["tools"] = to_openai_tools(req.tools)
        if req.tool_choice is not None:
            kwargs["tool_choice"] = req.tool_choice
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.stop is not None:
            kwargs["stop"] = req.stop
        if req.seed is not None:
            kwargs["seed"] = req.seed
        if req.json_mode_enabled:
            if req.response_format_schema:
                # Try schema, but streaming with json_schema may be unsupported; fallback transparently
                try:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": req.response_format_schema,
                            "strict": True,
                        },
                    }
                except Exception:
                    kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {"type": "json_object"}
        if req.timeout is not None:
            kwargs["timeout"] = req.timeout

        # Optional lightweight debug
        if os.environ.get("ORCH_DEBUG_LLM_ARGS") == "1":  # pragma: no cover - debug only
            try:
                tool_names = [t.name for t in (req.tools or [])]
                print(
                    "[orchestrator] openai.stream args: tools=",
                    tool_names,
                    "tool_choice=",
                    req.tool_choice,
                    flush=True,
                )
            except Exception:
                pass

        try:
            events = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # pragma: no cover - network exceptions via SDK
            raise _to_provider_error(e) from e

        # Track tool call start state by index
        started: dict[int, bool] = {}
        tool_ids: dict[int, str | None] = {}
        tool_names: dict[int, str | None] = {}

        for ev in events:
            # Pull the first choice since we always request a single completion
            choice = getattr(ev, "choices", [None])[0]
            if choice is None:
                continue
            delta = getattr(choice, "delta", None)
            finish_reason = getattr(choice, "finish_reason", None)
            if delta is None:
                # End of stream typically arrives with a finish_reason
                if finish_reason is not None:
                    yield ChatChunk(type="message.end", finish_reason=finish_reason, usage=None)
                continue

            # Role
            role = getattr(delta, "role", None)
            if role:
                yield ChatChunk(type="role", role=role)

            # Content text delta
            text = getattr(delta, "content", None)
            if text:
                yield ChatChunk(type="content.delta", text_delta=text)

            # Tool calls deltas
            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    index = getattr(tc, "index", None)
                    if index is None:
                        # Fallback if index missing: assume single tool call
                        index = 0
                    # Gather id and function info
                    tc_id = getattr(tc, "id", None)
                    function = getattr(tc, "function", None)
                    name = getattr(function, "name", None) if function is not None else None
                    args_piece = (
                        getattr(function, "arguments", None) if function is not None else None
                    )

                    # Start event if first time we see id/name for this index
                    if not started.get(index) and (tc_id or name):
                        started[index] = True
                        tool_ids[index] = tc_id
                        tool_names[index] = name
                        yield ChatChunk(
                            type="tool_call.start",
                            tool_call_index=index,
                            tool_call_id=tc_id,
                            tool_name=name,
                        )

                    # Arguments stream deltas
                    if args_piece:
                        yield ChatChunk(
                            type="tool_call.delta",
                            tool_call_index=index,
                            tool_call_id=tool_ids.get(index),
                            tool_name=tool_names.get(index),
                            arguments_delta=args_piece,
                        )

            # End event when finish_reason appears alongside empty delta
            if finish_reason is not None and not getattr(delta, "content", None) and not getattr(
                delta, "tool_calls", None
            ) and not getattr(delta, "role", None):
                yield ChatChunk(type="message.end", finish_reason=finish_reason, usage=None)


def _to_provider_error(err: Exception) -> ProviderError:
    # Try to extract status/code/message from OpenAI SDK errors
    status_code: int | None = None
    code: str | None = None
    message = str(err)
    try:
        status_code = getattr(err, "status_code", None)
        code = getattr(getattr(err, "error", None), "code", None) or getattr(err, "code", None)
        message_obj = getattr(err, "message", None)
        if message_obj is not None:
            message = str(message_obj)
    except Exception:  # pragma: no cover - best-effort
        message = str(err)
    return ProviderError(message=message, status_code=status_code, code=code)
