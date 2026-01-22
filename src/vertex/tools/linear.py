"""Generic Linear Programming tool."""

from typing import Any

from vertex.config import (
    DEFAULT_TIMEOUT_MS,
    DEFAULT_VAR_LOWER_BOUND,
    DEFAULT_VAR_UPPER_BOUND,
    ConstraintSense,
    ObjectiveSense,
    SolverType,
)
from vertex.metrics import track_solve_metrics
from vertex.models.linear import Constraint, LPProblem, LPSolution, Objective, Variable
from vertex.solvers.linear import LinearSolver
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_problem_size, validate_timeout


@track_solve_metrics(tool_name="solve_linear_program")
@validate_problem_size()
@validate_timeout()
async def solve_lp(
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
    solver_type: str = SolverType.GLOP,
    time_limit_ms: int = DEFAULT_TIMEOUT_MS,
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
        solver_type: Backend solver to use (GLOP, SCIP, etc.).
        time_limit_ms: Maximum solve time in milliseconds (default: 5 minutes).

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

    solver = LinearSolver(
        solver_type=SolverType(solver_type), time_limit_ms=time_limit_ms
    )
    # Run CPU-bound solve in executor
    return await run_in_executor(solver.solve, problem)
