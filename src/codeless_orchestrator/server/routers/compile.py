from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ...config.loader import load_graph
from ...engine.compiler import compile_graph
from ...engine.executor import OrchestrationEngine
from ...engine.types import ProviderResolver
from ...tools.registry import ToolRegistry
from .. import deps
from ..errors import to_http_error
from ..schemas import CompileResponse

router = APIRouter(prefix="", tags=["compile"])


def _detect_orchestration_kind(cfg: Any) -> tuple[str, dict[str, Any]]:
    # Internal-only import to avoid exporting private helper in engine.compiler
    from ...engine import compiler as _compiler

    return _compiler._detect_orchestration(cfg)  # type: ignore[attr-defined]


@router.post("/compile", response_model=CompileResponse)
def compile_config(
    config: dict[str, Any] = Body(..., description="Graph configuration JSON"),
    engine: OrchestrationEngine = Depends(deps.get_engine),
    registry: ToolRegistry = Depends(deps.get_registry),
    provider_resolver: ProviderResolver = Depends(deps.get_provider_resolver),
) -> CompileResponse:
    try:
        cfg, _ = load_graph(config, validate=True)

        # Determine orchestration
        kind, data = _detect_orchestration_kind(cfg)

        # Compile (also validates tool attachments against registry). To avoid importing
        # heavyweight provider SDKs during metadata-only compilation, use a lightweight
        # stub provider resolver.
        from ...providers.base import LLMProvider
        from ...providers.types import ChatMessage, ChatRequest, ChatResponse

        class _StubProvider(LLMProvider):  # pragma: no cover - trivial
            def chat(self, req: ChatRequest) -> ChatResponse:
                return ChatResponse(message=ChatMessage(role="assistant", content=""))

            def stream(self, req: ChatRequest):
                if False:
                    yield  # pragma: no cover - generator form

        compiled = engine.compile(cfg, provider_resolver=lambda _mp: _StubProvider(), registry=registry)

        # Agents metadata
        agents_meta: list[dict[str, Any]] = []
        for n in cfg.nodes:
            if getattr(n, "kind", None) == "agent.codeless":
                agents_meta.append(
                    {
                        "id": n.id,
                        "label": n.label,
                        "provider": n.data.model.provider,
                        "model_id": n.data.model.model_id,
                        "tools": list(n.data.tools.attached or []),
                    }
                )

        # Tools metadata from nodes
        tools_meta: list[dict[str, Any]] = []
        for n in cfg.nodes:
            if getattr(n, "kind", None) == "tool":
                tools_meta.append(
                    {
                        "id": n.id,
                        "label": n.label,
                        "tool_id": n.data.tool_id,
                        "version": n.data.version,
                    }
                )

        capabilities = {
            "streaming": True,
            "tools": any((a.get("tools") for a in agents_meta)),
        }

        notes: list[str] = []
        # Hints
        if kind == "single" and compiled.tools_attached:
            notes.append("Single-agent with tools attached")

        return CompileResponse(
            ok=True,
            orchestration=kind,  # type: ignore[arg-type]
            agents=agents_meta,
            tools=tools_meta,
            capabilities=capabilities,
            notes=notes,
        )
    except Exception as e:  # Map to HTTP error
        raise to_http_error(e)
