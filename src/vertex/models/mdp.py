"""Pydantic models for Markov Decision Processes."""

from typing import Any

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class MDPResult(BaseModel):
    """Result of solving a generic MDP."""

    status: SolverStatus
    optimal_value: float = Field(
        description="Expected discounted reward from start state"
    )
    policy: dict[str, str] = Field(description="Map from state to optimal action")
    values: dict[str, float] = Field(description="Value function for each state")
    solve_time_ms: float | None = None
