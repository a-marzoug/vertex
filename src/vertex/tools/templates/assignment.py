"""Assignment optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, VariableType
from vertex.models.mip import MIPConstraint, MIPObjective, MIPProblem, MIPVariable
from vertex.solvers.mip import MIPSolver


class AssignmentResult(BaseModel):
    """Result of assignment optimization."""

    status: str
    total_cost: float | None = None
    assignments: list[dict[str, str | float]] = Field(default_factory=list)
    solve_time_ms: float | None = None


def optimize_assignment(
    workers: list[str],
    tasks: list[str],
    costs: dict[str, dict[str, float]],
) -> AssignmentResult:
    """
    Solve assignment problem: assign workers to tasks minimizing total cost.

    Args:
        workers: List of worker names. Example: ["Alice", "Bob", "Charlie"]
        tasks: List of task names. Example: ["Task1", "Task2", "Task3"]
        costs: Cost matrix as nested dict. costs[worker][task] = cost.
            Example: {"Alice": {"Task1": 10, "Task2": 15}, "Bob": {"Task1": 12, "Task2": 8}}

    Returns:
        AssignmentResult with optimal assignments and total cost.
    """
    variables = []
    for w in workers:
        for t in tasks:
            variables.append(MIPVariable(name=f"{w}_{t}", var_type=VariableType.BINARY))

    constraints = []
    # Each worker assigned to exactly one task
    for w in workers:
        constraints.append(
            MIPConstraint(
                coefficients={f"{w}_{t}": 1 for t in tasks},
                sense=ConstraintSense.EQ,
                rhs=1,
            )
        )
    # Each task assigned to exactly one worker
    for t in tasks:
        constraints.append(
            MIPConstraint(
                coefficients={f"{w}_{t}": 1 for w in workers},
                sense=ConstraintSense.EQ,
                rhs=1,
            )
        )

    objective_coefficients = {f"{w}_{t}": costs[w][t] for w in workers for t in tasks}

    problem = MIPProblem(
        variables=variables,
        constraints=constraints,
        objective=MIPObjective(coefficients=objective_coefficients, sense=ObjectiveSense.MINIMIZE),
    )

    solution = MIPSolver().solve(problem)

    assignments = []
    if solution.variable_values:
        for w in workers:
            for t in tasks:
                if solution.variable_values.get(f"{w}_{t}", 0) > 0.5:
                    assignments.append({"worker": w, "task": t, "cost": costs[w][t]})

    return AssignmentResult(
        status=solution.status,
        total_cost=solution.objective_value,
        assignments=assignments,
        solve_time_ms=solution.solve_time_ms,
    )
