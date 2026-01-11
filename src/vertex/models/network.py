"""Pydantic models for Network Optimization problems."""

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class Arc(BaseModel):
    """Directed arc in a network."""

    source: str
    target: str
    capacity: float = Field(default=float("inf"))
    cost: float = Field(default=0)


class MaxFlowResult(BaseModel):
    """Result of maximum flow computation."""

    status: SolverStatus
    max_flow: float | None = None
    arc_flows: dict[str, float] = Field(default_factory=dict)
    source_side_cut: list[str] = Field(default_factory=list)
    sink_side_cut: list[str] = Field(default_factory=list)
    solve_time_ms: float | None = None


class MinCostFlowResult(BaseModel):
    """Result of minimum cost flow computation."""

    status: SolverStatus
    total_cost: float | None = None
    total_flow: float | None = None
    arc_flows: dict[str, float] = Field(default_factory=dict)
    solve_time_ms: float | None = None


class ShortestPathResult(BaseModel):
    """Result of shortest path computation."""

    status: SolverStatus
    distance: float | None = None
    path: list[str] = Field(default_factory=list)
    solve_time_ms: float | None = None


class MSTEdge(BaseModel):
    """Edge in minimum spanning tree."""

    source: str
    target: str
    weight: float


class MSTResult(BaseModel):
    """Result of Minimum Spanning Tree computation."""

    status: SolverStatus
    total_weight: float | None = None
    edges: list[MSTEdge] = Field(default_factory=list)
    solve_time_ms: float | None = None


class MultiCommodityFlowResult(BaseModel):
    """Result of Multi-Commodity Flow computation."""

    status: SolverStatus
    total_cost: float | None = None
    commodity_flows: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Commodity -> (arc -> flow) mapping",
    )
    solve_time_ms: float | None = None
