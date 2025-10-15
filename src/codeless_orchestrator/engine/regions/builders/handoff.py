from __future__ import annotations

from typing import Callable

from langgraph.graph import StateGraph

from ..builder import BuildChildFn, RegionBuildResult
from ..context import CompileContext
from ..types import HandoffRegion
from ...state import EngineState
from ....logging import get_logger
from ....providers.base import LLMProvider

try:  # pragma: no cover - legacy compatibility
    from ...graph_builders import handoff as _legacy_handoff_module
except Exception:  # pragma: no cover - legacy compatibility
    _legacy_handoff_module = None

from ...router import choose_next_agent as _router_choose_next_agent

_log = get_logger(__name__)


def _resolve_router_fn() -> Callable[..., str]:
    if _legacy_handoff_module is not None and hasattr(
        _legacy_handoff_module, "choose_next_agent"
    ):
        return getattr(_legacy_handoff_module, "choose_next_agent")
    return _router_choose_next_agent


def _cond_factory(
    *,
    tools_key: str,
    exit_key: str,
    region: HandoffRegion,
    provider: LLMProvider,
    candidate_keys: dict[str, str],
) -> Callable[[EngineState], str]:
    def _choose(state: EngineState) -> str:
        tcs = state.get("tool_calls") or []
        if tcs:
            return tools_key
        if not candidate_keys:
            return exit_key
        if len(candidate_keys) == 1:
            return next(iter(candidate_keys.values()))
        router_fn = _resolve_router_fn()
        try:
            choice = router_fn(
                messages=list(state.get("messages", [])),
                current_agent=region.root_agent,
                candidates=list(candidate_keys.keys()),
                provider=provider,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            _log.warning("Router failed (%s); defaulting to '%s'", exc, next(iter(candidate_keys.keys())))
            choice = next(iter(candidate_keys.keys()))
        if choice not in candidate_keys:
            first = next(iter(candidate_keys.keys()))
            _log.warning(
                "Router returned invalid id '%s'; defaulting to '%s'",
                choice,
                first,
            )
            choice = first
        return candidate_keys.get(choice) or next(iter(candidate_keys.values()))

    return _choose


def build_handoff(
    *,
    region: HandoffRegion,
    graph: StateGraph,
    ctx: CompileContext,
    build_child: BuildChildFn,
) -> RegionBuildResult:
    entry_key = ctx.region_entry_key(region.id)
    exit_key = ctx.region_exit_key(region.id)
    ctx.ensure_passthrough_node(graph, entry_key)
    ctx.ensure_passthrough_node(graph, exit_key)

    artifacts = ctx.agent_artifacts(region.root_agent)
    agent_key = ctx.agent_node_key(region.id, region.root_agent)
    tools_key = ctx.tools_node_key(region.id, region.root_agent)
    graph.add_node(agent_key, artifacts.agent_fn)
    graph.add_node(tools_key, artifacts.tools_fn)
    graph.add_edge(entry_key, agent_key)
    graph.add_edge(tools_key, agent_key)

    branch_results: dict[str, RegionBuildResult] = {}
    for br in region.branches:
        child_res = build_child(br.region, True)
        branch_results[br.entry_agent] = child_res
        graph.add_edge(child_res.exit_key, exit_key)

    candidate_map = {aid: res.entry_key for aid, res in branch_results.items()}

    graph.add_conditional_edges(
        agent_key,
        _cond_factory(
            tools_key=tools_key,
            exit_key=exit_key,
            region=region,
            provider=artifacts.provider,
            candidate_keys=candidate_map,
        ),
    )

    notes = list(region.notes)
    for res in branch_results.values():
        notes.extend(res.notes)

    return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=notes)


__all__ = ["build_handoff"]
