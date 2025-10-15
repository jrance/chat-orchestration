from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langgraph.graph import END, START, StateGraph

from ..errors import GraphCompileError
from ..state import EngineState
from ..types import CompiledSingleAgent, ProviderResolver
from ...config.schema import GraphConfig
from ...tools.registry import ToolRegistry
from .analyzer import RegionAnalyzer
from .builder import RegionBuildResult
from .context import CompileContext
from .types import RegionBase, collect_region_tree
from .builders import handoff as handoff_builder
from .builders import sequential as sequential_builder
from .builders import concurrent as concurrent_builder
from .builders import groupchat as groupchat_builder


BuilderFn = Callable[[RegionBase, StateGraph, CompileContext, Callable[[RegionBase, bool], RegionBuildResult]], RegionBuildResult]


@dataclass(slots=True)
class _ResolverState:
    ctx: CompileContext
    builders: dict[str, BuilderFn]

    def build_inline(self, region: RegionBase, graph: StateGraph) -> RegionBuildResult:
        builder = self.builders.get(region.kind)
        if builder is None:
            raise GraphCompileError(f"No builder for region kind '{region.kind}'")

        def _build_child(child_region: RegionBase, inline: bool) -> RegionBuildResult:
            if inline:
                return self.build_inline(child_region, graph)
            return self.build_subgraph(child_region)

        return builder(region, graph, self.ctx, _build_child)

    def build_subgraph(self, region: RegionBase) -> RegionBuildResult:
        sub_graph = StateGraph(EngineState)
        sub_ctx = CompileContext(
            cfg=self.ctx.cfg,
            provider_resolver=self.ctx.provider_resolver,
            registry=self.ctx.registry,
        )
        sub_state = _ResolverState(ctx=sub_ctx, builders=self.builders)
        result = sub_state.build_inline(region, sub_graph)
        sub_graph.add_edge(START, result.entry_key)
        sub_graph.add_edge(result.exit_key, END)
        app = sub_graph.compile()
        return RegionBuildResult(
            region=region,
            entry_key=result.entry_key,
            exit_key=result.exit_key,
            notes=list(result.notes),
            app=app,
        )


class OrchestrationResolver:
    """Analyze and build a hierarchical orchestration into a LangGraph app."""

    def __init__(
        self,
        *,
        cfg: GraphConfig,
        provider_resolver: ProviderResolver,
        registry: ToolRegistry,
    ) -> None:
        self.cfg = cfg
        self.provider_resolver = provider_resolver
        self.registry = registry

    def compile(self) -> CompiledSingleAgent:
        analyzer = RegionAnalyzer(self.cfg)
        root_region = analyzer.analyze()
        ctx = CompileContext(
            cfg=self.cfg,
            provider_resolver=self.provider_resolver,
            registry=self.registry,
        )
        graph = StateGraph(EngineState)
        builders: dict[str, BuilderFn] = {
            "leaf": lambda region, g, c, build_child: sequential_builder.build_leaf(
                region=region, graph=g, ctx=c, build_child=build_child
            ),
            "sequential": lambda region, g, c, build_child: sequential_builder.build_sequential(
                region=region, graph=g, ctx=c, build_child=build_child
            ),
            "handoff": lambda region, g, c, build_child: handoff_builder.build_handoff(
                region=region, graph=g, ctx=c, build_child=build_child
            ),
            "concurrent": lambda region, g, c, build_child: concurrent_builder.build_concurrent(
                region=region, graph=g, ctx=c, build_child=build_child
            ),
            "groupchat": lambda region, g, c, build_child: groupchat_builder.build_groupchat(
                region=region, graph=g, ctx=c, build_child=build_child
            ),
        }
        resolver_state = _ResolverState(ctx=ctx, builders=builders)
        root_result = resolver_state.build_inline(root_region, graph)

        graph.add_edge(START, root_result.entry_key)
        graph.add_edge(root_result.exit_key, END)
        app = graph.compile()

        root_agent_id = next(iter(root_region.agents), None)
        if root_agent_id is None:
            raise GraphCompileError("Root region has no agents")
        artifacts = ctx.agent_artifacts(root_agent_id)

        region_tree = collect_region_tree(root_region)

        return CompiledSingleAgent(
            agent_id=root_agent_id,
            graph=app,
            provider=artifacts.provider,
            tools_attached=dict(artifacts.tool_bindings),
            tool_bindings_by_fn=dict(artifacts.bindings_by_fn),
            tool_config=artifacts.tool_cfg,
            model_params=artifacts.node.data.model,
            structured_output=artifacts.node.data.structured_output,
            safety=artifacts.node.data.safety,
            region_tree=region_tree,
            region_notes=list(root_result.notes),
        )


__all__ = ["OrchestrationResolver"]
