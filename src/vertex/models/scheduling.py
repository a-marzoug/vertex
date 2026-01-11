"""Pydantic models for Scheduling and Routing problems."""

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


# TSP Models
class TSPResult(BaseModel):
    """Result of Traveling Salesman Problem."""

    status: SolverStatus
    route: list[str] = Field(default_factory=list, description="Ordered list of locations in tour")
    total_distance: float | None = None
    solve_time_ms: float | None = None


# VRP Models
class VRPRoute(BaseModel):
    """Single vehicle route."""

    vehicle: int
    stops: list[str] = Field(description="Ordered stops including depot")
    distance: float
    load: float = 0


class VRPResult(BaseModel):
    """Result of Vehicle Routing Problem."""

    status: SolverStatus
    routes: list[VRPRoute] = Field(default_factory=list)
    total_distance: float | None = None
    solve_time_ms: float | None = None


# Job Shop Models
class Task(BaseModel):
    """A task in job shop scheduling."""

    machine: int
    duration: int


class ScheduledTask(BaseModel):
    """A scheduled task with timing."""

    job: int
    task: int
    machine: int
    start: int
    duration: int
    end: int


class JobShopResult(BaseModel):
    """Result of Job Shop Scheduling."""

    status: SolverStatus
    makespan: int | None = None
    schedule: list[ScheduledTask] = Field(default_factory=list)
    solve_time_ms: float | None = None
