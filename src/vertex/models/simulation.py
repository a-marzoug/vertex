"""Pydantic models for Simulation Optimization."""

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class SimulationParameter(BaseModel):
    """Parameter to tune in simulation."""

    name: str
    lower_bound: float
    upper_bound: float
    initial_guess: float | None = None
    is_integer: bool = False


class SimulationOptimizationResult(BaseModel):
    """Result of simulation optimization."""

    status: SolverStatus
    optimal_parameters: dict[str, float]
    optimal_objective: float
    num_evaluations: int
    solve_time_ms: float
    history: list[dict] = Field(
        default_factory=list, description="Optimization trajectory"
    )
