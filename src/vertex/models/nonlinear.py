"""Pydantic models for Nonlinear Programming."""

from typing import Literal

from pydantic import BaseModel, Field

from vertex.config import SolverStatus, VariableType


class NLPVariable(BaseModel):
    """Decision variable for NLP."""

    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    lower_bound: float | None = None
    upper_bound: float | None = None
    initial_guess: float = 0.0


class NLPConstraint(BaseModel):
    """Nonlinear constraint: expression sense rhs (e.g. x**2 + y <= 5)."""

    expression: str = Field(description="Mathematical expression (e.g., 'x**2 + y')")
    sense: Literal["<=", ">=", "="]
    rhs: float = 0.0


class NLPProblem(BaseModel):
    """Nonlinear Programming Problem."""

    variables: list[NLPVariable]
    objective_expression: str = Field(
        description="Objective function (e.g., 'x**2 + y**2')"
    )
    objective_sense: Literal["minimize", "maximize"] = "minimize"
    constraints: list[NLPConstraint] = Field(default_factory=list)


class NLPSolution(BaseModel):
    """Result of NLP solver."""

    status: SolverStatus
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    solve_time_ms: float | None = None
    message: str | None = None
