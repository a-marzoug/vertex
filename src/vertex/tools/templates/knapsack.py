"""Knapsack optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, VariableType
from vertex.models.mip import MIPConstraint, MIPObjective, MIPProblem, MIPVariable
from vertex.solvers.mip import MIPSolver


class KnapsackResult(BaseModel):
    """Result of knapsack optimization."""

    status: str
    total_value: float | None = None
    total_weight: float | None = None
    selected_items: list[str] = Field(default_factory=list)
    solve_time_ms: float | None = None


def optimize_knapsack(
    items: list[str],
    values: dict[str, float],
    weights: dict[str, float],
    capacity: float,
) -> KnapsackResult:
    """
    Solve 0/1 knapsack: select items to maximize value within capacity.

    Args:
        items: List of item names. Example: ["laptop", "camera", "phone"]
        values: Value of each item. Example: {"laptop": 1000, "camera": 500, "phone": 300}
        weights: Weight of each item. Example: {"laptop": 3, "camera": 1, "phone": 0.5}
        capacity: Maximum total weight. Example: 4

    Returns:
        KnapsackResult with selected items and total value.
    """
    problem = MIPProblem(
        variables=[MIPVariable(name=item, var_type=VariableType.BINARY) for item in items],
        constraints=[
            MIPConstraint(
                coefficients=weights,
                sense=ConstraintSense.LEQ,
                rhs=capacity,
            )
        ],
        objective=MIPObjective(coefficients=values, sense=ObjectiveSense.MAXIMIZE),
    )

    solution = MIPSolver().solve(problem)

    selected = []
    total_weight = 0.0
    if solution.variable_values:
        for item in items:
            if solution.variable_values.get(item, 0) > 0.5:
                selected.append(item)
                total_weight += weights[item]

    return KnapsackResult(
        status=solution.status,
        total_value=solution.objective_value,
        total_weight=round(total_weight, 6),
        selected_items=selected,
        solve_time_ms=solution.solve_time_ms,
    )
