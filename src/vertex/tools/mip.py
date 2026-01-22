"""Generic Mixed-Integer Programming tool."""

from typing import Any

from vertex.config import (
    DEFAULT_TIMEOUT_MS,
    ConstraintSense,
    ObjectiveSense,
    SolverType,
    VariableType,
)
from vertex.metrics import track_solve_metrics
from vertex.models.mip import (
    MIPConstraint,
    MIPObjective,
    MIPProblem,
    MIPSolution,
    MIPVariable,
)
from vertex.solvers.mip import MIPSolver
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_problem_size, validate_timeout


@track_solve_metrics(tool_name="solve_mixed_integer_program")
@validate_problem_size()
@validate_timeout()
async def solve_mip(
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
    solver_type: str = SolverType.SCIP,
    time_limit_ms: int = DEFAULT_TIMEOUT_MS,
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
        solver_type: Backend solver to use (SCIP, SAT, GLOP - though GLOP ignores integers).
        time_limit_ms: Maximum solve time in milliseconds (default: 5 minutes).

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

    solver = MIPSolver(solver_type=SolverType(solver_type), time_limit_ms=time_limit_ms)
    return await run_in_executor(solver.solve, problem)
