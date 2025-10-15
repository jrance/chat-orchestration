from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from ...config.schema import AgentNode, GraphConfig
from ..errors import GraphCompileError
from .types import (
    BranchRef,
    ConcurrentRegion,
    GroupChatRegion,
    HandoffRegion,
    LeafRegion,
    RegionBase,
    RegionId,
    SequentialRegion,
)


@dataclass(slots=True)
class AgentGraph:
    agents: dict[str, AgentNode]
    out_edges: dict[str, list[str]]
    in_edges: dict[str, list[str]]


def _build_agent_graph(cfg: GraphConfig) -> AgentGraph:
    agents: dict[str, AgentNode] = {n.id: n for n in cfg.nodes if isinstance(n, AgentNode)}
    out_edges: dict[str, list[str]] = {aid: [] for aid in agents}
    in_edges: dict[str, list[str]] = {aid: [] for aid in agents}
    for edge in cfg.edges:
        if edge.from_ in agents and edge.to in agents:
            out_edges[edge.from_].append(edge.to)
            in_edges[edge.to].append(edge.from_)
    return AgentGraph(agents=agents, out_edges=out_edges, in_edges=in_edges)


def _tarjan(nodes: Iterable[str], out_edges: dict[str, list[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in out_edges.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    for n in nodes:
        if n not in indices:
            strongconnect(n)
    return result


class RegionAnalyzer:
    def __init__(self, cfg: GraphConfig) -> None:
        self.cfg = cfg
        self.graph = _build_agent_graph(cfg)
        if not self.graph.agents:
            raise GraphCompileError("No agent.codeless nodes found")
        self._region_counters: dict[str, int] = defaultdict(int)
        self._region_cache: dict[tuple[str, frozenset[str]], RegionBase] = {}
        self._visiting: set[str] = set()
        self._scc_lookup: dict[str, list[str]] = {}
        self._scc_members: set[str] = set()
        self._order_index: dict[str, int] = {
            node.id: idx for idx, node in enumerate(self.cfg.nodes) if isinstance(node, AgentNode)
        }
        self._init_sccs()

    def _init_sccs(self) -> None:
        nodes = list(self.graph.agents.keys())
        sccs = _tarjan(nodes, self.graph.out_edges)
        for comp in sccs:
            if len(comp) >= 2 or any(v in self.graph.out_edges.get(v, []) for v in comp):
                # Sort participants deterministically by appearance order
                sorted_comp = sorted(comp, key=lambda x: self._order_index.get(x, 0))
                for member in sorted_comp:
                    self._scc_lookup[member] = sorted_comp
                    self._scc_members.add(member)

    def analyze(self) -> RegionBase:
        root_ids = [aid for aid, indeg in self.graph.in_edges.items() if indeg == []]
        if not root_ids:
            if self._scc_members:
                ordered_members = sorted(self._scc_members, key=lambda x: self._order_index.get(x, 0))
                root_id = ordered_members[0]
            else:
                raise GraphCompileError("orchestration.no_root")
        elif len(root_ids) > 1:
            raise GraphCompileError("orchestration.multiple_roots")
        else:
            root_id = root_ids[0]
        region = self._build_region(root_id, stop_nodes=frozenset())
        return region

    def _next_region_id(self, kind: str) -> RegionId:
        self._region_counters[kind] += 1
        return f"{kind}-{self._region_counters[kind]}"

    def _build_region(self, entry_id: str, *, stop_nodes: frozenset[str]) -> RegionBase:
        if entry_id in stop_nodes:
            raise GraphCompileError("orchestration.invalid_boundary")
        cache_key = (entry_id, stop_nodes)
        if cache_key in self._region_cache:
            return self._region_cache[cache_key]
        if entry_id in self._visiting:
            raise GraphCompileError("orchestration.unsupported_cycle")
        self._visiting.add(entry_id)
        try:
            if entry_id in self._scc_members:
                region = self._build_groupchat(entry_id, stop_nodes=stop_nodes)
            else:
                out_neighbors = [n for n in self.graph.out_edges.get(entry_id, []) if n not in stop_nodes]
                if not out_neighbors:
                    region = self._build_leaf(entry_id)
                elif len(out_neighbors) == 1:
                    region = self._build_sequential_chain(entry_id, stop_nodes=stop_nodes)
                else:
                    conc = self._detect_concurrent(entry_id, out_neighbors, stop_nodes=stop_nodes)
                    if conc is not None:
                        region = self._build_concurrent(entry_id, conc, stop_nodes=stop_nodes)
                    else:
                        region = self._build_handoff(entry_id, out_neighbors, stop_nodes=stop_nodes)
        finally:
            self._visiting.remove(entry_id)
        self._region_cache[cache_key] = region
        return region

    def _build_leaf(self, entry_id: str) -> LeafRegion:
        rid = self._next_region_id("leaf")
        node = self.graph.agents[entry_id]
        return LeafRegion(id=rid, agents=[entry_id], agent_id=entry_id, notes=[f"leaf:{node.label}"])

    def _build_sequential_chain(self, entry_id: str, *, stop_nodes: frozenset[str]) -> SequentialRegion:
        rid = self._next_region_id("seq")
        chain: list[str] = [entry_id]
        current = entry_id
        seen: set[str] = {entry_id}
        while True:
            neighbors = [n for n in self.graph.out_edges.get(current, []) if n not in stop_nodes]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            if nxt in seen:
                raise GraphCompileError("orchestration.unsupported_cycle")
            if nxt in self._scc_members:
                break
            chain.append(nxt)
            seen.add(nxt)
            current = nxt
        tail_neighbors = [n for n in self.graph.out_edges.get(current, []) if n not in stop_nodes and n not in chain]
        tail_children = [self._build_region(n, stop_nodes=stop_nodes) for n in dict.fromkeys(tail_neighbors)]
        return SequentialRegion(id=rid, agents=list(chain), chain=chain, tail_children=tail_children, notes=[f"sequential:length={len(chain)}"])

    def _build_handoff(self, entry_id: str, neighbors: list[str], *, stop_nodes: frozenset[str]) -> HandoffRegion:
        rid = self._next_region_id("handoff")
        branch_refs: list[BranchRef] = []
        for n in dict.fromkeys(neighbors):
            child = self._build_region(n, stop_nodes=stop_nodes)
            branch_refs.append(BranchRef(entry_agent=n, region=child))
        return HandoffRegion(id=rid, agents=[entry_id], root_agent=entry_id, branches=branch_refs)

    def _detect_concurrent(
        self,
        entry_id: str,
        neighbors: list[str],
        *,
        stop_nodes: frozenset[str],
    ) -> dict[str, Any] | None:
        if len(neighbors) <= 1:
            return None
        leaf_branches = True
        join_candidate: str | None = None
        for n in neighbors:
            outs = [m for m in self.graph.out_edges.get(n, []) if m not in stop_nodes]
            if not outs:
                continue
            leaf_branches = False
            if len(outs) != 1:
                return None
            candidate = outs[0]
            if candidate in neighbors:
                return None
            if join_candidate is None:
                join_candidate = candidate
            elif join_candidate != candidate:
                return None
        meta_name = getattr(getattr(self.cfg, "meta", None), "name", "") or ""
        if join_candidate is None and "concurrent" not in meta_name.lower():
            return None
        return {
            "branches": list(dict.fromkeys(neighbors)),
            "join": join_candidate,
        }

    def _build_concurrent(
        self,
        entry_id: str,
        data: dict[str, Any],
        *,
        stop_nodes: frozenset[str],
    ) -> ConcurrentRegion:
        rid = self._next_region_id("concurrent")
        join_id = data.get("join")
        join_stop = stop_nodes
        if join_id is not None:
            join_stop = frozenset(set(stop_nodes) | {join_id})
        branches: list[BranchRef] = []
        for n in data.get("branches", []):
            child = self._build_region(n, stop_nodes=join_stop)
            branches.append(BranchRef(entry_agent=n, region=child))
        join_region = self._build_region(join_id, stop_nodes=stop_nodes) if join_id is not None else None
        inferred_strategy = "first_finish" if join_region is not None else "aggregate"
        return ConcurrentRegion(
            id=rid,
            agents=[entry_id],
            fanout_root=entry_id,
            branches=branches,
            join=join_region,
            strategy=None,
            notes=[f"concurrent:strategy={inferred_strategy}", f"concurrent:branches={len(branches)}"],
        )

    def _build_groupchat(self, entry_id: str, *, stop_nodes: frozenset[str]) -> GroupChatRegion:
        members = self._scc_lookup.get(entry_id)
        if not members:
            raise GraphCompileError("orchestration.groupchat_resolution_failed")
        rid = self._next_region_id("groupchat")

        # Determine downstream continuation (if any)
        outgoing: list[str] = []
        for member in members:
            for nxt in self.graph.out_edges.get(member, []):
                if nxt not in members and nxt not in stop_nodes:
                    outgoing.append(nxt)
        unique_outgoing = list(dict.fromkeys(outgoing))
        if len(unique_outgoing) > 1:
            raise GraphCompileError("orchestration.ambiguous_region")
        next_region = self._build_region(unique_outgoing[0], stop_nodes=stop_nodes) if unique_outgoing else None

        # Moderator heuristics: prefer structural hub, fall back to label hints
        agents = self.graph.agents
        hub_candidates: list[str] = []
        for member in members:
            in_deg = sum(1 for src in self.graph.in_edges.get(member, []) if src in members)
            out_deg = sum(1 for dst in self.graph.out_edges.get(member, []) if dst in members)
            if in_deg >= 2 and out_deg >= 2:
                hub_candidates.append(member)
        label_candidates = [m for m in members if "moderator" in (agents[m].label or "").lower()]

        mode: Literal["moderator", "round_robin"] = "round_robin"
        moderator_id: str | None = None
        if len(hub_candidates) == 1:
            moderator_id = hub_candidates[0]
            mode = "moderator"
        elif label_candidates:
            moderator_id = label_candidates[0]
            mode = "moderator"

        participants = [m for m in members if m != moderator_id] if moderator_id is not None else list(members)
        if mode == "moderator" and not participants:
            # Fallback: no eligible speakers, revert to round-robin across the SCC
            moderator_id = None
            mode = "round_robin"
            participants = list(members)

        notes = [f"groupchat:mode={mode}"]
        if moderator_id is not None:
            notes.append(f"groupchat:moderator={moderator_id}")

        return GroupChatRegion(
            id=rid,
            agents=list(members),
            participants=participants,
            mode=mode,
            moderator_id=moderator_id,
            next_region=next_region,
            notes=notes,
        )


def analyze_regions(cfg: GraphConfig) -> RegionBase:
    return RegionAnalyzer(cfg).analyze()


__all__ = ["analyze_regions", "RegionAnalyzer"]
