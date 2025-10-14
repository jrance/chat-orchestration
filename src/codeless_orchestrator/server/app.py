from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _json_response_class() -> type[JSONResponse]:
    """Return ORJSONResponse if available, else FastAPI's JSONResponse.

    We import lazily to keep the optional dependency truly optional.
    """
    try:  # pragma: no cover - optional path
        from fastapi.responses import ORJSONResponse  # type: ignore

        return ORJSONResponse  # type: ignore[return-value]
    except Exception:  # pragma: no cover - fallback
        return JSONResponse


def _add_middlewares(app: FastAPI) -> None:
    # Basic permissive CORS; can be tightened via env/config later
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable[[Request], Any]):
        # Propagate request id from header or generate
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response: Response = await call_next(request)
        # Ensure header is present
        response.headers.setdefault("X-Request-ID", req_id)
        return response


def build_app() -> FastAPI:
    app = FastAPI(title="Codeless Orchestrator Server", version="0.0.1", default_response_class=_json_response_class())
    _add_middlewares(app)

    # Health endpoint
    @app.get("/healthz")
    def healthz() -> dict[str, str]:  # pragma: no cover - trivial
        return {"status": "ok"}

    # Routers
    from .routers import compile as compile_router
    from .routers import execute as execute_router
    from .routers import validate as validate_router

    app.include_router(validate_router.router)
    app.include_router(compile_router.router)
    app.include_router(execute_router.router)

    return app


# Uvicorn entrypoint
app = build_app()

