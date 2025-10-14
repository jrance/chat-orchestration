from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _load_env_from_file(path: str = ".env") -> None:
    """Best-effort .env loader so the server works outside VS Code.

    - Tries python-dotenv if installed
    - Falls back to a tiny parser (KEY=VALUE, no export, ignores # comments)
    - Never overrides already-set variables
    """
    # Try python-dotenv if available
    try:  # pragma: no cover - optional dependency path
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except Exception:
        pass

    # Lightweight fallback
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" not in s:
                        continue
                    key, val = s.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        # Silent best-effort only
        pass


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
    # Ensure env vars are available even when not launched via VS Code
    _load_env_from_file()

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
