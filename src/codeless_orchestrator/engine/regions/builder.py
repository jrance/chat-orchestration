from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from langgraph.graph import StateGraph

from .context import CompileContext
from .types import RegionBase


@dataclass(slots=True)
class RegionBuildResult:
    region: RegionBase
    entry_key: str
    exit_key: str
    notes: list[str] = field(default_factory=list)
    app: Any | None = None


BuildChildFn = Callable[[RegionBase, bool], RegionBuildResult]


class RegionBuilder(Protocol):
    def build(
        self,
        *,
        region: RegionBase,
        graph: StateGraph,
        ctx: CompileContext,
        build_child: BuildChildFn,
    ) -> RegionBuildResult:
        ...


__all__ = ["RegionBuilder", "RegionBuildResult", "BuildChildFn"]
