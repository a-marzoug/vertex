"""Pydantic models for Linear Programming problems."""

from pydantic import BaseModel, Field

from vertex.config import (
    DEFAULT_VAR_LOWER_BOUND,
    DEFAULT_VAR_UPPER_BOUND,
    ConstraintSense,
    ObjectiveSense,
    SolverStatus,
)


class Variable(BaseModel):
    """Decision variable in an LP problem."""

    name: str = Field(description="Variable identifier")
    lower_bound: float = Field(
        default=DEFAULT_VAR_LOWER_BOUND,
        description="Minimum value",
    )
    upper_bound: float = Field(
        default=DEFAULT_VAR_UPPER_BOUND,
        description="Maximum value",
    )


class Constraint(BaseModel):
    """Linear constraint: sum(coefficients[var] * var) sense rhs."""

    coefficients: dict[str, float] = Field(
        description="Variable name to coefficient mapping",
    )
    sense: ConstraintSense = Field(description="Comparison operator")
    rhs: float = Field(description="Right-hand side value")
    name: str | None = Field(default=None, description="Optional constraint name")


class Objective(BaseModel):
    """Linear objective function."""

    coefficients: dict[str, float] = Field(
        description="Variable name to coefficient mapping",
    )
    sense: ObjectiveSense = Field(
        default=ObjectiveSense.MAXIMIZE,
        description="Optimization direction",
    )


class LPProblem(BaseModel):
    """Complete Linear Programming problem definition."""

    variables: list[Variable] = Field(description="Decision variables")
    constraints: list[Constraint] = Field(description="Linear constraints")
    objective: Objective = Field(description="Objective function")
    name: str | None = Field(default=None, description="Problem name")


class LPSolution(BaseModel):
    """Solution to a Linear Programming problem."""

    status: SolverStatus = Field(description="Solver result status")
    objective_value: float | None = Field(
        default=None,
        description="Optimal objective function value",
    )
    variable_values: dict[str, float] = Field(
        default_factory=dict,
        description="Optimal variable values",
    )
    shadow_prices: dict[str, float] = Field(
        default_factory=dict,
        description="Dual values for constraints (marginal value of relaxing each constraint)",
    )
    reduced_costs: dict[str, float] = Field(
        default_factory=dict,
        description="Reduced costs for variables (how much coefficient must improve to enter basis)",
    )
    solve_time_ms: float | None = Field(
        default=None,
        description="Solve time in milliseconds",
    )
    iterations: int | None = Field(
        default=None,
        description="Number of solver iterations",
    )
