from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Sequence

RegionKind = Literal["leaf", "sequential", "handoff", "concurrent", "groupchat", "external"]
RegionId = str


@dataclass(slots=True)
class RegionTree:
    """Serializable view of a region and its children."""

    id: RegionId
    kind: RegionKind
    agents: list[str] = field(default_factory=list)
    children: list["RegionTree"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegionBase:
    """Base class for all region descriptors produced by the analyzer."""

    id: RegionId
    kind: ClassVar[RegionKind]
    agents: list[str]
    notes: list[str] = field(default_factory=list)

    def to_tree(self) -> RegionTree:
        return RegionTree(id=self.id, kind=self.kind, agents=list(self.agents), children=self._tree_children(), notes=list(self.notes))

    def _tree_children(self) -> list[RegionTree]:  # pragma: no cover - overridden
        return []


@dataclass(slots=True)
class LeafRegion(RegionBase):
    kind: ClassVar[Literal["leaf"]] = "leaf"
    agent_id: str = ""

    def __post_init__(self) -> None:
        if not self.agent_id and self.agents:
            object.__setattr__(self, "agent_id", self.agents[0])
        elif not self.agents and self.agent_id:
            object.__setattr__(self, "agents", [self.agent_id])


@dataclass(slots=True)
class ExternalRegion(LeafRegion):
    """Placeholder region for external/BYOE agents."""

    kind: ClassVar[Literal["external"]] = "external"
    external_kind: Literal["remote", "byoe"] | None = None


@dataclass(slots=True)
class SequentialRegion(RegionBase):
    kind: ClassVar[Literal["sequential"]] = "sequential"
    chain: list[str] = field(default_factory=list)
    tail_children: list["RegionBase"] = field(default_factory=list)

    def _tree_children(self) -> list[RegionTree]:
        return [child.to_tree() for child in self.tail_children]


@dataclass(slots=True)
class BranchRef:
    entry_agent: str
    region: RegionBase
    label: str | None = None

    def to_tree(self) -> RegionTree:
        tree = self.region.to_tree()
        if self.label:
            tree.notes.append(f"branch:{self.label}")
        return tree


@dataclass(slots=True)
class HandoffRegion(RegionBase):
    kind: ClassVar[Literal["handoff"]] = "handoff"
    root_agent: str = ""
    branches: list[BranchRef] = field(default_factory=list)

    def _tree_children(self) -> list[RegionTree]:
        return [b.to_tree() for b in self.branches]


@dataclass(slots=True)
class ConcurrentRegion(RegionBase):
    kind: ClassVar[Literal["concurrent"]] = "concurrent"
    fanout_root: str = ""
    branches: list[BranchRef] = field(default_factory=list)
    join: RegionBase | None = None
    strategy: Literal["aggregate", "first_finish", "vote"] | None = None

    def _tree_children(self) -> list[RegionTree]:
        children = [b.to_tree() for b in self.branches]
        if self.join is not None:
            children.append(self.join.to_tree())
        return children


@dataclass(slots=True)
class GroupChatRegion(RegionBase):
    kind: ClassVar[Literal["groupchat"]] = "groupchat"
    participants: list[str] = field(default_factory=list)
    mode: Literal["moderator", "round_robin"] = "round_robin"
    moderator_id: str | None = None
    next_region: RegionBase | None = None

    def _tree_children(self) -> list[RegionTree]:
        return [self.next_region.to_tree()] if self.next_region else []


def collect_region_tree(root: RegionBase) -> RegionTree:
    return root.to_tree()


def flatten_regions(root: RegionBase) -> list[RegionBase]:
    result: list[RegionBase] = []

    def _walk(region: RegionBase) -> None:
        result.append(region)
        if isinstance(region, SequentialRegion):
            for child in region.tail_children:
                _walk(child)
        elif isinstance(region, HandoffRegion):
            for br in region.branches:
                _walk(br.region)
        elif isinstance(region, ConcurrentRegion):
            for br in region.branches:
                _walk(br.region)
            if region.join is not None:
                _walk(region.join)
        elif isinstance(region, GroupChatRegion):
            if region.next_region is not None:
                _walk(region.next_region)
        elif isinstance(region, LeafRegion):
            pass
        elif isinstance(region, ExternalRegion):
            pass

    _walk(root)
    return result


__all__ = [
    "RegionBase",
    "RegionTree",
    "RegionKind",
    "RegionId",
    "LeafRegion",
    "SequentialRegion",
    "HandoffRegion",
    "ConcurrentRegion",
    "GroupChatRegion",
    "ExternalRegion",
    "BranchRef",
    "collect_region_tree",
    "flatten_regions",
]
