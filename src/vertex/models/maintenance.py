"""Pydantic models for Maintenance Planning."""

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class MaintenanceState(BaseModel):
    """Represents a condition state of the equipment."""

    id: int
    name: str
    description: str | None = None
    failure_probability: float = Field(
        ..., description="Probability of failure in this state if no action taken"
    )


class MaintenanceAction(BaseModel):
    """Possible maintenance action."""

    id: int
    name: str
    cost: float
    target_state: int | None = Field(
        default=None, description="Deterministic target state (e.g., 0 for replacement)"
    )
    # If probabilistic transition, we need a matrix, but simplified for now


class OptimalPolicy(BaseModel):
    """Optimal action for a given state and time."""

    state: int
    time_step: int
    action: str
    expected_cost: float


class MaintenancePlanResult(BaseModel):
    """Result of maintenance optimization."""

    status: SolverStatus
    total_expected_cost: float
    policy: list[OptimalPolicy] = Field(
        description="Optimal action for each state/time"
    )
    solve_time_ms: float | None = None
