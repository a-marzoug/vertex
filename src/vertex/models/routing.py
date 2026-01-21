"""Pydantic models for Routing problems."""

from pydantic import BaseModel, Field

from vertex.config import SolverStatus
from vertex.models.viz import GanttChart


# TSP Models
class TSPResult(BaseModel):
    """Result of Traveling Salesman Problem."""

    status: SolverStatus
    route: list[str] = Field(
        default_factory=list, description="Ordered list of locations in tour"
    )
    total_distance: float | None = None
    solve_time_ms: float | None = None
    plot_data: dict | None = Field(
        default=None, description="Coordinates for plotting route"
    )


# VRP Models
class VRPRoute(BaseModel):
    """Single vehicle route."""

    vehicle: int
    stops: list[str] = Field(description="Ordered stops including depot")
    distance: float
    load: float = 0
    arrival_times: list[int] = Field(
        default_factory=list, description="Arrival time at each stop"
    )


class VRPResult(BaseModel):
    """Result of Vehicle Routing Problem."""

    status: SolverStatus
    routes: list[VRPRoute] = Field(default_factory=list)
    total_distance: float | None = None
    solve_time_ms: float | None = None
    visualization: GanttChart | None = None  # Gantt chart for time windows? Or map?
    # Usually VRP with TW can be visualized as a Gantt chart of routes.
