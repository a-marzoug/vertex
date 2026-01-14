"""Generic Linear Programming tool."""

from typing import Any

from vertex.config import (
    DEFAULT_VAR_LOWER_BOUND,
    DEFAULT_VAR_UPPER_BOUND,
    ConstraintSense,
    ObjectiveSense,
)
from vertex.models.linear import Constraint, LPProblem, LPSolution, Objective, Variable
from vertex.solvers.linear import LinearSolver


def solve_lp(
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> LPSolution:
    """
    Solve a Linear Programming problem.

    Args:
        variables: List of variables, each with 'name' and optional 'lower_bound', 'upper_bound'.
            Example: [{"name": "x", "lower_bound": 0}, {"name": "y", "lower_bound": 0}]
        constraints: List of constraints, each with 'coefficients' (dict), 'sense' (<=, >=, =), 'rhs'.
            Example: [{"coefficients": {"x": 1, "y": 2}, "sense": "<=", "rhs": 14}]
        objective_coefficients: Variable coefficients in objective function.
            Example: {"x": 3, "y": 4}
        objective_sense: Either "maximize" or "minimize".

    Returns:
        Solution with status, objective_value, variable_values, and solve_time_ms.
    """
    problem = LPProblem(
        variables=[
            Variable(
                name=v["name"],
                lower_bound=v.get("lower_bound", DEFAULT_VAR_LOWER_BOUND),
                upper_bound=v.get("upper_bound", DEFAULT_VAR_UPPER_BOUND),
            )
            for v in variables
        ],
        constraints=[
            Constraint(
                coefficients=c["coefficients"],
                sense=ConstraintSense(c["sense"]),
                rhs=c["rhs"],
                name=c.get("name"),
            )
            for c in constraints
        ],
        objective=Objective(
            coefficients=objective_coefficients,
            sense=ObjectiveSense(objective_sense),
        ),
    )

    solver = LinearSolver()
    return solver.solve(problem)
