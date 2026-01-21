"""Nonlinear Programming tools."""

from typing import Any, Literal

from vertex.metrics import track_solve_metrics
from vertex.models.nonlinear import NLPConstraint, NLPProblem, NLPSolution, NLPVariable
from vertex.solvers.nonlinear import solve_nlp
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_timeout
from vertex.config import VariableType


@track_solve_metrics(tool_name="solve_nonlinear_program")
@validate_timeout()
async def solve_nonlinear_program(
    variables: list[dict[str, Any]],
    objective_expression: str,
    constraints: list[dict[str, Any]] | None = None,
    objective_sense: str = "minimize",
    time_limit_ms: int = 30000,
) -> NLPSolution:
    """
    Solve a Nonlinear Programming (NLP) problem.

    Args:
        variables: List of variables with optional 'lower_bound', 'upper_bound', 'initial_guess'.
            Example: [{"name": "x", "lower_bound": 0}, {"name": "y"}]
        objective_expression: Mathematical expression to optimize (e.g., "x**2 + y**2").
            Uses standard Python/SymPy syntax.
        constraints: List of constraints.
            Example: [{"expression": "x + y", "sense": ">=", "rhs": 5}]
        objective_sense: "minimize" or "maximize".
        time_limit_ms: Solver time limit in milliseconds.

    Returns:
        Optimal values and objective.
    """
    # Build problem model
    problem_vars = []
    for v in variables:
        problem_vars.append(
            NLPVariable(
                name=v["name"],
                lower_bound=v.get("lower_bound"),
                upper_bound=v.get("upper_bound"),
                initial_guess=v.get("initial_guess", 0.0),
            )
        )

    problem_constraints = []
    if constraints:
        for c in constraints:
            problem_constraints.append(
                NLPConstraint(
                    expression=c["expression"],
                    sense=c["sense"],
                    rhs=c.get("rhs", 0.0),
                )
            )

    problem = NLPProblem(
        variables=problem_vars,
        objective_expression=objective_expression,
        objective_sense=objective_sense,  # type: ignore
        constraints=problem_constraints,
    )

    return await run_in_executor(solve_nlp, problem, time_limit_ms // 1000)


@track_solve_metrics(tool_name="solve_minlp")
@validate_timeout()
async def solve_minlp(
    variables: list[dict[str, Any]],
    objective_expression: str,
    constraints: list[dict[str, Any]] | None = None,
    objective_sense: str = "minimize",
    time_limit_ms: int = 30000,
) -> NLPSolution:
    """
    Solve a Mixed-Integer Nonlinear Programming (MINLP) problem.

    Args:
        variables: List of variables with 'var_type' ("continuous", "integer", or "binary").
            Example: [{"name": "x", "var_type": "integer", "lower_bound": 0, "upper_bound": 10},
                     {"name": "y", "var_type": "continuous"}]
        objective_expression: Nonlinear objective (e.g., "x**2 + log(y+1)").
        constraints: Nonlinear constraints.
            Example: [{"expression": "x**2 + y", "sense": "<=", "rhs": 25}]
        objective_sense: "minimize" or "maximize".
        time_limit_ms: Solver time limit.

    Returns:
        Optimal solution with integer and continuous variables.
    """
    problem_vars = []
    for v in variables:
        var_type_str = v.get("var_type", "continuous")
        if var_type_str == "binary":
            var_type = VariableType.BINARY
        elif var_type_str == "integer":
            var_type = VariableType.INTEGER
        else:
            var_type = VariableType.CONTINUOUS

        problem_vars.append(
            NLPVariable(
                name=v["name"],
                var_type=var_type,
                lower_bound=v.get("lower_bound"),
                upper_bound=v.get("upper_bound"),
                initial_guess=v.get("initial_guess", 0.0),
            )
        )

    problem_constraints = []
    if constraints:
        for c in constraints:
            problem_constraints.append(
                NLPConstraint(
                    expression=c["expression"],
                    sense=c["sense"],
                    rhs=c.get("rhs", 0.0),
                )
            )

    problem = NLPProblem(
        variables=problem_vars,
        objective_expression=objective_expression,
        objective_sense=objective_sense,  # type: ignore
        constraints=problem_constraints,
    )

    return await run_in_executor(solve_nlp, problem, time_limit_ms // 1000)
