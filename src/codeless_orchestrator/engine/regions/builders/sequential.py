from __future__ import annotations

from typing import Callable

from langgraph.graph import StateGraph

from ..builder import BuildChildFn, RegionBuildResult
from ..context import CompileContext
from ..types import LeafRegion, SequentialRegion
from ...state import EngineState


def _attach_agent_nodes(
    *,
    graph: StateGraph,
    ctx: CompileContext,
    region_id: str,
    agent_id: str,
) -> tuple[str, str]:
    artifacts = ctx.agent_artifacts(agent_id)
    agent_key = ctx.agent_node_key(region_id, agent_id)
    tools_key = ctx.tools_node_key(region_id, agent_id)
    graph.add_node(agent_key, artifacts.agent_fn)
    graph.add_node(tools_key, artifacts.tools_fn)
    graph.add_edge(tools_key, agent_key)
    return agent_key, tools_key


def _cond_factory(tools_key: str, target_key: str) -> Callable[[EngineState], str]:
    def _should_continue(state: EngineState) -> str:
        tcs = state.get("tool_calls") or []
        if tcs:
            return tools_key
        return target_key

    return _should_continue


def build_leaf(
    *,
    region: LeafRegion,
    graph: StateGraph,
    ctx: CompileContext,
    build_child: BuildChildFn,
) -> RegionBuildResult:
    entry_key = ctx.region_entry_key(region.id)
    exit_key = ctx.region_exit_key(region.id)
    ctx.ensure_passthrough_node(graph, entry_key)
    ctx.ensure_passthrough_node(graph, exit_key)

    agent_id = region.agent_id or (region.agents[0] if region.agents else None)
    if agent_id is None:
        raise ValueError("LeafRegion requires at least one agent id")

    artifacts = ctx.agent_artifacts(agent_id)
    agent_key = ctx.agent_node_key(region.id, agent_id)
    tools_key = ctx.tools_node_key(region.id, agent_id)
    graph.add_node(agent_key, artifacts.agent_fn)
    graph.add_node(tools_key, artifacts.tools_fn)

    graph.add_edge(entry_key, agent_key)
    graph.add_edge(tools_key, agent_key)

    graph.add_conditional_edges(agent_key, _cond_factory(tools_key, exit_key))

    return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=list(region.notes))


def build_sequential(
    *,
    region: SequentialRegion,
    graph: StateGraph,
    ctx: CompileContext,
    build_child: BuildChildFn,
) -> RegionBuildResult:
    entry_key = ctx.region_entry_key(region.id)
    exit_key = ctx.region_exit_key(region.id)
    ctx.ensure_passthrough_node(graph, entry_key)
    ctx.ensure_passthrough_node(graph, exit_key)

    chain = list(region.chain or region.agents)
    if not chain:
        graph.add_edge(entry_key, exit_key)
        return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=list(region.notes))

    child_results: list[RegionBuildResult] = []
    for child in region.tail_children:
        child_res = build_child(child, True)
        child_results.append(child_res)

    # Build agent chain
    agent_keys: list[str] = []
    tool_keys: list[str] = []
    for idx, agent_id in enumerate(chain):
        agent_key, tools_key = _attach_agent_nodes(
            graph=graph, ctx=ctx, region_id=region.id, agent_id=agent_id
        )
        agent_keys.append(agent_key)
        tool_keys.append(tools_key)
    graph.add_edge(entry_key, agent_keys[0])

    # Wire conditional edges
    for idx, agent_key in enumerate(agent_keys):
        tools_key = tool_keys[idx]
        if idx + 1 < len(agent_keys):
            next_key = agent_keys[idx + 1]
        elif child_results:
            next_key = child_results[0].entry_key
        else:
            next_key = exit_key
        graph.add_conditional_edges(agent_key, _cond_factory(tools_key, next_key))

    # Connect child exits to region exit
    for child_res in child_results:
        graph.add_edge(child_res.exit_key, exit_key)

    notes = list(region.notes)
    for child_res in child_results:
        notes.extend(child_res.notes)

    return RegionBuildResult(region=region, entry_key=entry_key, exit_key=exit_key, notes=notes)


__all__ = ["build_sequential", "build_leaf"]
