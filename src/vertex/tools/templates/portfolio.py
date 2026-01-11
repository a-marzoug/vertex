"""Portfolio optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class PortfolioResult(BaseModel):
    """Result of portfolio optimization."""

    status: str = Field(description="Solver status")
    expected_return: float | None = Field(description="Expected portfolio return")
    allocation: dict[str, float] = Field(description="Investment amount per asset")
    allocation_pct: dict[str, float] = Field(description="Allocation percentages")
    solve_time_ms: float | None = Field(description="Solve time in milliseconds")


def optimize_portfolio(
    assets: list[str],
    expected_returns: dict[str, float],
    budget: float,
    min_allocation: dict[str, float] | None = None,
    max_allocation: dict[str, float] | None = None,
) -> PortfolioResult:
    """
    Optimize portfolio allocation to maximize expected return.

    Args:
        assets: List of asset names.
            Example: ["stocks", "bonds", "real_estate"]
        expected_returns: Expected return rate for each asset.
            Example: {"stocks": 0.12, "bonds": 0.05, "real_estate": 0.08}
        budget: Total investment budget.
            Example: 100000
        min_allocation: Minimum investment per asset (optional).
            Example: {"bonds": 10000}
        max_allocation: Maximum investment per asset (optional).
            Example: {"stocks": 50000}

    Returns:
        PortfolioResult with optimal allocation and expected return.
    """
    variables = []
    for asset in assets:
        lower = (min_allocation or {}).get(asset, 0)
        upper = (max_allocation or {}).get(asset, budget)
        variables.append(Variable(name=asset, lower_bound=lower, upper_bound=upper))

    problem = LPProblem(
        variables=variables,
        constraints=[
            Constraint(
                coefficients={a: 1.0 for a in assets},
                sense=ConstraintSense.EQ,
                rhs=budget,
                name="budget_constraint",
            )
        ],
        objective=Objective(
            coefficients=expected_returns,
            sense=ObjectiveSense.MAXIMIZE,
        ),
    )

    solution = LinearSolver().solve(problem)

    # Calculate allocation percentages
    allocation_pct = {}
    if solution.variable_values and budget > 0:
        allocation_pct = {
            asset: (amount / budget) * 100
            for asset, amount in solution.variable_values.items()
        }

    return PortfolioResult(
        status=solution.status,
        expected_return=solution.objective_value,
        allocation=solution.variable_values,
        allocation_pct=allocation_pct,
        solve_time_ms=solution.solve_time_ms,
    )
