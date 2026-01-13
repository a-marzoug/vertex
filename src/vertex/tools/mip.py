"""Generic Mixed-Integer Programming tool."""

from vertex.config import ConstraintSense, ObjectiveSense, VariableType
from vertex.models.mip import (
    MIPConstraint,
    MIPObjective,
    MIPProblem,
    MIPSolution,
    MIPVariable,
)
from vertex.solvers.mip import MIPSolver


def solve_mip(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> MIPSolution:
    """
    Solve a Mixed-Integer Programming problem.

    Args:
        variables: List of variables with 'name', 'var_type' (continuous/integer/binary),
            and optional 'lower_bound', 'upper_bound'.
            Example: [{"name": "x", "var_type": "integer"}, {"name": "y", "var_type": "binary"}]
        constraints: List of constraints with 'coefficients', 'sense', 'rhs'.
            Example: [{"coefficients": {"x": 1, "y": 2}, "sense": "<=", "rhs": 10}]
        objective_coefficients: Variable coefficients in objective.
        objective_sense: "maximize" or "minimize".

    Returns:
        MIPSolution with status, objective_value, and variable_values.
    """
    problem = MIPProblem(
        variables=[
            MIPVariable(
                name=v["name"],
                var_type=VariableType(v.get("var_type", "continuous")),
                lower_bound=v.get("lower_bound", 0),
                upper_bound=v.get("upper_bound", float("inf")),
            )
            for v in variables
        ],
        constraints=[
            MIPConstraint(
                coefficients=c["coefficients"],
                sense=ConstraintSense(c["sense"]),
                rhs=c["rhs"],
            )
            for c in constraints
        ],
        objective=MIPObjective(
            coefficients=objective_coefficients,
            sense=ObjectiveSense(objective_sense),
        ),
    )

    return MIPSolver().solve(problem)
