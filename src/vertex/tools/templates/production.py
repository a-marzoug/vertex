"""Production optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class ProductionResult(BaseModel):
    """Result of production optimization."""

    status: str = Field(description="Solver status")
    total_profit: float | None = Field(description="Maximum achievable profit")
    production_plan: dict[str, float] = Field(
        description="Units to produce per product"
    )
    resource_usage: dict[str, float] = Field(description="Resource consumption")
    solve_time_ms: float | None = Field(description="Solve time in milliseconds")


def optimize_production(
    products: list[str],
    resources: list[str],
    profits: dict[str, float],
    requirements: dict[str, dict[str, float]],
    availability: dict[str, float],
) -> ProductionResult:
    """
    Optimize production to maximize profit given resource constraints.

    Args:
        products: List of product names to produce.
            Example: ["chairs", "tables"]
        resources: List of resource names.
            Example: ["wood", "labor_hours"]
        profits: Profit per unit for each product.
            Example: {"chairs": 45, "tables": 80}
        requirements: Resource requirements per product unit.
            Example: {"chairs": {"wood": 5, "labor_hours": 2}, "tables": {"wood": 20, "labor_hours": 5}}
        availability: Available quantity of each resource.
            Example: {"wood": 400, "labor_hours": 100}

    Returns:
        ProductionResult with optimal production plan and profit.
    """
    problem = LPProblem(
        variables=[Variable(name=p, lower_bound=0) for p in products],
        constraints=[
            Constraint(
                coefficients={p: requirements[p].get(r, 0) for p in products},
                sense=ConstraintSense.LEQ,
                rhs=availability[r],
                name=f"{r}_limit",
            )
            for r in resources
        ],
        objective=Objective(
            coefficients=profits,
            sense=ObjectiveSense.MAXIMIZE,
        ),
    )

    solution = LinearSolver().solve(problem)

    # Calculate resource usage
    resource_usage = {}
    if solution.variable_values:
        for r in resources:
            resource_usage[r] = sum(
                requirements[p].get(r, 0) * solution.variable_values.get(p, 0)
                for p in products
            )

    return ProductionResult(
        status=solution.status,
        total_profit=solution.objective_value,
        production_plan=solution.variable_values,
        resource_usage=resource_usage,
        solve_time_ms=solution.solve_time_ms,
    )
