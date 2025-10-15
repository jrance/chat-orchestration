from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Iterator
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, Request
from starlette.responses import StreamingResponse

from ...config.loader import load_graph
from ...engine.executor import OrchestrationEngine
from ...engine.types import EngineRunInput, ProviderResolver
from ...runtime.streaming import chunk_to_event
from ...tools.registry import ToolRegistry
from .. import deps
from ..errors import to_http_error
from ..schemas import ChatMessageModel, ExecuteRequest, ExecuteResponse

router = APIRouter(prefix="", tags=["execute"])


def _request_id_from(request: Request, body_req_id: str | None) -> str:
    return body_req_id or request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _encode_sse(data: dict[str, Any]) -> bytes:
    # Minimal SSE format: one data line, then blank line
    return ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode("utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_to_run_input(inp: dict[str, Any], *, org_preamble: str | None, extra_meta: dict[str, Any] | None = None) -> EngineRunInput | dict[str, Any]:
    # Coerce ExecuteInputModel-like dict into EngineRunInput
    if "messages" in inp and inp["messages"] is not None:
        messages = [ChatMessageModel.model_validate(m).to_internal() for m in inp["messages"]]
        md = {"org_preamble": org_preamble}
        if extra_meta:
            md.update(extra_meta)
        return EngineRunInput(messages=messages, metadata=md, vars=inp.get("vars"))
    if "text" in inp and inp["text"] is not None:
        md = {"org_preamble": org_preamble}
        if extra_meta:
            md.update(extra_meta)
        return EngineRunInput(messages=inp["text"], metadata=md, vars=inp.get("vars"))
    # Fallback empty -> empty user text
    md = {"org_preamble": org_preamble}
    if extra_meta:
        md.update(extra_meta)
    return EngineRunInput(messages="", metadata=md, vars=inp.get("vars"))


@router.post("/execute", response_model=ExecuteResponse)
def execute_nonstream(
    body: ExecuteRequest,
    request: Request,
    engine: OrchestrationEngine = Depends(deps.get_engine),
    registry: ToolRegistry = Depends(deps.get_registry),
    provider_resolver: ProviderResolver = Depends(deps.get_provider_resolver),
    org_preamble: str | None = Depends(deps.get_org_preamble),
) -> ExecuteResponse:
    try:
        # Load graph; skip deep cross-validation for performance in execute
        cfg, _ = load_graph(body.config, validate=False)
        # Read telemetry settings from first agent (single-agent support in this version)
        emit_metrics = True
        labels: dict[str, str] = {}
        try:
            agent_nodes = [n for n in cfg.nodes if n.kind == "agent.codeless"]
            if agent_nodes:
                tcfg = agent_nodes[0].data.telemetry
                labels = dict(getattr(tcfg, "labels", {}) or {})
                emit_metrics = bool(getattr(tcfg, "emit_usage", True))
        except Exception:
            emit_metrics = True
            labels = {}

        req_id = _request_id_from(request, (body.options.request_id if body.options else None))
        telemetry = deps.build_telemetry(request, labels={**labels})
        # Timers around compile and execute
        t_compile = telemetry.start_timer("compile")
        compiled = engine.compile(cfg, provider_resolver=provider_resolver, registry=registry)
        telemetry.stop_timer(t_compile)

        run_input = _input_to_run_input(
            body.input.model_dump(by_alias=True),
            org_preamble=org_preamble,
            extra_meta={"_telemetry": telemetry, "request_id": req_id},
        )
        t_exec = telemetry.start_timer("execute")
        result = engine.execute(compiled, run_input)
        telemetry.stop_timer(t_exec)

        resp = ExecuteResponse.from_result(result, request_id=req_id)
        if emit_metrics:
            resp.metrics = telemetry.snapshot()
        return resp
    except Exception as e:
        raise to_http_error(e)


@router.post("/execute/stream")
async def execute_stream(
    body: ExecuteRequest,
    request: Request,
    engine: OrchestrationEngine = Depends(deps.get_engine),
    registry: ToolRegistry = Depends(deps.get_registry),
    provider_resolver: ProviderResolver = Depends(deps.get_provider_resolver),
    org_preamble: str | None = Depends(deps.get_org_preamble),
):
    try:
        # Load graph; skip deep cross-validation for performance in execute/stream
        cfg, _ = load_graph(body.config, validate=False)
        # Telemetry settings
        emit_metrics = True
        labels: dict[str, str] = {}
        try:
            agent_nodes = [n for n in cfg.nodes if n.kind == "agent.codeless"]
            if agent_nodes:
                tcfg = agent_nodes[0].data.telemetry
                labels = dict(getattr(tcfg, "labels", {}) or {})
                emit_metrics = bool(getattr(tcfg, "emit_usage", True))
        except Exception:
            emit_metrics = True
            labels = {}

        req_id = _request_id_from(request, (body.options.request_id if body.options else None))
        telemetry = deps.build_telemetry(request, labels={**labels})
        t_compile = telemetry.start_timer("compile")
        compiled = engine.compile(cfg, provider_resolver=provider_resolver, registry=registry)
        telemetry.stop_timer(t_compile)

        run_input = _input_to_run_input(
            body.input.model_dump(by_alias=True),
            org_preamble=org_preamble,
            extra_meta={"_telemetry": telemetry, "request_id": req_id},
        )
    except Exception as e:
        raise to_http_error(e)

    # Try to use sse-starlette if available for heartbeat support
    # Avoid raising ModuleNotFoundError when not installed (e.g., during debugging)
    use_sse_starlette = False
    try:  # pragma: no cover - optional path
        import importlib.util as _ilu

        use_sse_starlette = _ilu.find_spec("sse_starlette.sse") is not None
    except Exception:  # pragma: no cover - extremely defensive
        use_sse_starlette = False

    if use_sse_starlette:
        try:  # pragma: no cover - optional path
            from sse_starlette.sse import EventSourceResponse

            def _iter_events() -> Iterator[bytes]:
                try:
                    for chunk in engine.stream_execute(compiled, run_input):
                        ev = chunk_to_event(chunk)
                        ev["request_id"] = req_id
                        ev["timestamp"] = _now_iso()
                        if emit_metrics and ev.get("type") == "message.end":
                            ev["metrics"] = telemetry.snapshot()
                        yield _encode_sse(ev)
                except Exception as ex:  # emit error then end
                    yield _encode_sse({"type": "error", "error": {"message": str(ex)}, "request_id": req_id, "timestamp": _now_iso()})
                finally:
                    yield _encode_sse({"type": "end", "request_id": req_id, "timestamp": _now_iso()})

            # Yield pre-encoded SSE bytes so sse-starlette won't map dicts to ServerSentEvent(**data)
            return EventSourceResponse(_iter_events(), ping=15.0, headers={"X-Request-ID": req_id})
        except Exception:
            # Fall through to manual SSE
            pass

    # Manual SSE via StreamingResponse
    async def _gen() -> AsyncIterator[bytes]:
        try:
            for chunk in engine.stream_execute(compiled, run_input):
                ev = chunk_to_event(chunk)
                ev["request_id"] = req_id
                ev["timestamp"] = _now_iso()
                if emit_metrics and ev.get("type") == "message.end":
                    ev["metrics"] = telemetry.snapshot()
                yield _encode_sse(ev)
                # Respect disconnect if possible
                if await request.is_disconnected():
                    break
        except Exception as ex:
            yield _encode_sse({"type": "error", "error": {"message": str(ex)}, "request_id": req_id, "timestamp": _now_iso()})
        finally:
            yield _encode_sse({"type": "end", "request_id": req_id, "timestamp": _now_iso()})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Request-ID": req_id,
    }
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)
