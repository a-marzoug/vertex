"""Pydantic models for Scheduling and Routing problems."""

from typing import Any

from pydantic import BaseModel, Field

from vertex.config import SolverStatus
from vertex.models.viz import GanttChart


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
    visualization: GanttChart | None = None


# Bin Packing Models
class BinAssignment(BaseModel):
    """Items assigned to a bin."""

    bin_id: int
    items: list[str]
    total_weight: float


class BinPackingResult(BaseModel):
    """Result of Bin Packing Problem."""

    status: SolverStatus
    num_bins_used: int | None = None
    assignments: list[BinAssignment] = Field(default_factory=list)
    solve_time_ms: float | None = None


# Set Covering Models
class SetCoverResult(BaseModel):
    """Result of Set Covering Problem."""

    status: SolverStatus
    selected_sets: list[str] = Field(default_factory=list)
    total_cost: float | None = None
    solve_time_ms: float | None = None


# Graph Coloring Models
class GraphColoringResult(BaseModel):
    """Result of Graph Coloring Problem."""

    status: SolverStatus
    num_colors: int | None = None
    coloring: dict[str, int] = Field(
        default_factory=dict, description="Node to color assignment"
    )
    solve_time_ms: float | None = None


# Cutting Stock Models
class CuttingPattern(BaseModel):
    """A cutting pattern for stock material."""

    stock_id: int
    cuts: dict[str, int] = Field(description="Item to count mapping")
    waste: float


class CuttingStockResult(BaseModel):
    """Result of Cutting Stock Problem."""

    status: SolverStatus
    num_stock_used: int | None = None
    patterns: list[CuttingPattern] = Field(default_factory=list)
    total_waste: float | None = None
    solve_time_ms: float | None = None


# Flow Shop Models
class FlowShopResult(BaseModel):
    """Result of Flow Shop Scheduling."""

    status: SolverStatus
    makespan: int | None = None
    job_sequence: list[int] = Field(default_factory=list, description="Order of jobs")
    schedule: list[ScheduledTask] = Field(default_factory=list)
    solve_time_ms: float | None = None
    visualization: GanttChart | None = None


# Parallel Machine Models
class ParallelMachineResult(BaseModel):
    """Result of Parallel Machine Scheduling."""

    status: SolverStatus
    makespan: int | None = None
    machine_assignments: dict[int, list[int]] = Field(
        default_factory=dict, description="Machine to job list"
    )
    schedule: list[ScheduledTask] = Field(default_factory=list)
    solve_time_ms: float | None = None
    visualization: GanttChart | None = None


# RCPSP Models
class RCPSPResult(BaseModel):
    """Result of Resource-Constrained Project Scheduling."""

    status: SolverStatus
    makespan: int | None = None
    schedule: list[dict[str, Any]] = Field(default_factory=list)
    solve_time_ms: float | None = None
    visualization: GanttChart | None = None
