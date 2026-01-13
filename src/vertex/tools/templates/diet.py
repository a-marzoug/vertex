"""Diet optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class DietResult(BaseModel):
    """Result of diet optimization."""

    status: str = Field(description="Solver status")
    total_cost: float | None = Field(description="Minimum diet cost")
    food_quantities: dict[str, float] = Field(description="Amount of each food")
    nutrient_intake: dict[str, float] = Field(description="Nutrient amounts consumed")
    solve_time_ms: float | None = Field(description="Solve time in milliseconds")


def optimize_diet(
    foods: list[str],
    nutrients: list[str],
    costs: dict[str, float],
    nutrition_values: dict[str, dict[str, float]],
    min_requirements: dict[str, float],
    max_limits: dict[str, float] | None = None,
) -> DietResult:
    """
    Find minimum cost diet meeting nutritional requirements.

    Args:
        foods: List of food names.
            Example: ["bread", "milk", "eggs"]
        nutrients: List of nutrient names.
            Example: ["calories", "protein", "calcium"]
        costs: Cost per unit of each food.
            Example: {"bread": 2.0, "milk": 1.5, "eggs": 3.0}
        nutrition_values: Nutrient content per unit of each food.
            Example: {"bread": {"calories": 200, "protein": 5}, ...}
        min_requirements: Minimum required amount of each nutrient.
            Example: {"calories": 2000, "protein": 50}
        max_limits: Optional maximum limits for nutrients.
            Example: {"calories": 2500}

    Returns:
        DietResult with optimal food quantities and cost.
    """
    constraints = []

    # Minimum nutrient requirements
    for nutrient in nutrients:
        if nutrient in min_requirements:
            constraints.append(
                Constraint(
                    coefficients={
                        f: nutrition_values[f].get(nutrient, 0) for f in foods
                    },
                    sense=ConstraintSense.GEQ,
                    rhs=min_requirements[nutrient],
                    name=f"min_{nutrient}",
                )
            )

    # Maximum nutrient limits
    if max_limits:
        for nutrient, limit in max_limits.items():
            constraints.append(
                Constraint(
                    coefficients={
                        f: nutrition_values[f].get(nutrient, 0) for f in foods
                    },
                    sense=ConstraintSense.LEQ,
                    rhs=limit,
                    name=f"max_{nutrient}",
                )
            )

    problem = LPProblem(
        variables=[Variable(name=f, lower_bound=0) for f in foods],
        constraints=constraints,
        objective=Objective(
            coefficients=costs,
            sense=ObjectiveSense.MINIMIZE,
        ),
    )

    solution = LinearSolver().solve(problem)

    # Calculate nutrient intake
    nutrient_intake = {}
    if solution.variable_values:
        for nutrient in nutrients:
            nutrient_intake[nutrient] = sum(
                nutrition_values[f].get(nutrient, 0)
                * solution.variable_values.get(f, 0)
                for f in foods
            )

    return DietResult(
        status=solution.status,
        total_cost=solution.objective_value,
        food_quantities=solution.variable_values,
        nutrient_intake=nutrient_intake,
        solve_time_ms=solution.solve_time_ms,
    )
