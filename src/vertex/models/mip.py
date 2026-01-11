"""Pydantic models for Mixed-Integer Programming problems."""

from pydantic import BaseModel, Field

from vertex.config import (
    ConstraintSense,
    DEFAULT_VAR_LOWER_BOUND,
    DEFAULT_VAR_UPPER_BOUND,
    ObjectiveSense,
    SolverStatus,
    VariableType,
)


class MIPVariable(BaseModel):
    """Decision variable in a MIP problem."""

    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    lower_bound: float = DEFAULT_VAR_LOWER_BOUND
    upper_bound: float = DEFAULT_VAR_UPPER_BOUND


class MIPConstraint(BaseModel):
    """Linear constraint."""

    coefficients: dict[str, float]
    sense: ConstraintSense
    rhs: float
    name: str | None = None


class MIPObjective(BaseModel):
    """Linear objective function."""

    coefficients: dict[str, float]
    sense: ObjectiveSense = ObjectiveSense.MAXIMIZE


class MIPProblem(BaseModel):
    """Mixed-Integer Programming problem."""

    variables: list[MIPVariable]
    constraints: list[MIPConstraint]
    objective: MIPObjective
    name: str | None = None


class MIPSolution(BaseModel):
    """Solution to a MIP problem."""

    status: SolverStatus
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    solve_time_ms: float | None = None
