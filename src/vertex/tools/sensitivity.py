"""Sensitivity analysis tools for LP solutions."""

from typing import Any

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, SolverStatus
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class SensitivityReport(BaseModel):
    """Sensitivity analysis report for an LP solution."""

    status: SolverStatus
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)

    # Constraint analysis
    binding_constraints: list[str] = Field(
        default_factory=list,
        description="Constraints at their limits (shadow price != 0)",
    )
    slack_constraints: list[str] = Field(
        default_factory=list,
        description="Constraints with slack (shadow price = 0)",
    )
    shadow_prices: dict[str, float] = Field(
        default_factory=dict,
        description="Marginal value of relaxing each constraint by one unit",
    )

    # Variable analysis
    basic_variables: list[str] = Field(
        default_factory=list,
        description="Variables in the optimal basis (value > 0)",
    )
    nonbasic_variables: list[str] = Field(
        default_factory=list,
        description="Variables at their bounds (value = 0)",
    )
    reduced_costs: dict[str, float] = Field(
        default_factory=dict,
        description="How much objective coefficient must improve for variable to enter basis",
    )


def analyze_sensitivity(
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> SensitivityReport:
    """
    Perform sensitivity analysis on an LP problem.

    Returns shadow prices (dual values) and reduced costs to help understand:
    - Which constraints are limiting the solution
    - How much the objective would improve if constraints were relaxed
    - Which variables are candidates for improvement
    """
    # Build problem
    vars_ = [Variable(**v) for v in variables]
    constrs = [
        Constraint(
            coefficients=c["coefficients"],
            sense=ConstraintSense(c["sense"]),
            rhs=c["rhs"],
            name=c.get("name"),
        )
        for c in constraints
    ]
    obj = Objective(
        coefficients=objective_coefficients,
        sense=ObjectiveSense(objective_sense),
    )
    problem = LPProblem(variables=vars_, constraints=constrs, objective=obj)

    # Solve
    solution = LinearSolver().solve(problem)

    if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        return SensitivityReport(status=solution.status)

    # Classify constraints
    binding = []
    slack = []
    for name, price in solution.shadow_prices.items():
        if abs(price) > 1e-6:
            binding.append(name)
        else:
            slack.append(name)

    # Classify variables
    basic = []
    nonbasic = []
    for name, value in solution.variable_values.items():
        if abs(value) > 1e-6:
            basic.append(name)
        else:
            nonbasic.append(name)

    return SensitivityReport(
        status=solution.status,
        objective_value=solution.objective_value,
        variable_values=solution.variable_values,
        binding_constraints=binding,
        slack_constraints=slack,
        shadow_prices=solution.shadow_prices,
        basic_variables=basic,
        nonbasic_variables=nonbasic,
        reduced_costs=solution.reduced_costs,
    )
