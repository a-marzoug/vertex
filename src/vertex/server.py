"""Vertex MCP Server - Operations Research tools for decision makers."""

from mcp.server.fastmcp import FastMCP

from vertex.config import DEFAULT_HOST, DEFAULT_PORT, SERVER_DESCRIPTION, SERVER_NAME
from vertex.models.linear import LPSolution
from vertex.prompts.linear import formulate_lp, interpret_solution
from vertex.tools.linear import solve_lp
from vertex.tools.templates.diet import DietResult, optimize_diet
from vertex.tools.templates.portfolio import PortfolioResult, optimize_portfolio
from vertex.tools.templates.production import ProductionResult, optimize_production

# Create MCP server
mcp = FastMCP(
    SERVER_NAME,
    instructions=SERVER_DESCRIPTION,
    stateless_http=True,
    json_response=True,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
)


# Register tools
@mcp.tool()
def solve_linear_program(
    variables: list[dict],
    constraints: list[dict],
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
    return solve_lp(variables, constraints, objective_coefficients, objective_sense)


@mcp.tool()
def optimize_production_plan(
    products: list[str],
    resources: list[str],
    profits: dict[str, float],
    requirements: dict[str, dict[str, float]],
    availability: dict[str, float],
) -> ProductionResult:
    """
    Optimize production to maximize profit given resource constraints.

    Args:
        products: List of product names. Example: ["chairs", "tables"]
        resources: List of resource names. Example: ["wood", "labor_hours"]
        profits: Profit per unit. Example: {"chairs": 45, "tables": 80}
        requirements: Resource needs per product. Example: {"chairs": {"wood": 5, "labor_hours": 2}}
        availability: Available resources. Example: {"wood": 400, "labor_hours": 100}

    Returns:
        ProductionResult with optimal production plan and total profit.
    """
    return optimize_production(products, resources, profits, requirements, availability)


@mcp.tool()
def optimize_diet_plan(
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
        foods: List of food names. Example: ["bread", "milk", "eggs"]
        nutrients: List of nutrients. Example: ["calories", "protein"]
        costs: Cost per unit of food. Example: {"bread": 2.0, "milk": 1.5}
        nutrition_values: Nutrients per food unit. Example: {"bread": {"calories": 200, "protein": 5}}
        min_requirements: Minimum nutrients needed. Example: {"calories": 2000, "protein": 50}
        max_limits: Optional maximum limits. Example: {"calories": 2500}

    Returns:
        DietResult with optimal food quantities and total cost.
    """
    return optimize_diet(
        foods, nutrients, costs, nutrition_values, min_requirements, max_limits
    )


@mcp.tool()
def optimize_investment_portfolio(
    assets: list[str],
    expected_returns: dict[str, float],
    budget: float,
    min_allocation: dict[str, float] | None = None,
    max_allocation: dict[str, float] | None = None,
) -> PortfolioResult:
    """
    Optimize portfolio allocation to maximize expected return.

    Args:
        assets: List of asset names. Example: ["stocks", "bonds", "real_estate"]
        expected_returns: Return rate per asset. Example: {"stocks": 0.12, "bonds": 0.05}
        budget: Total investment amount. Example: 100000
        min_allocation: Minimum per asset. Example: {"bonds": 10000}
        max_allocation: Maximum per asset. Example: {"stocks": 50000}

    Returns:
        PortfolioResult with optimal allocation and expected return.
    """
    return optimize_portfolio(
        assets, expected_returns, budget, min_allocation, max_allocation
    )


# Register prompts
@mcp.prompt()
def formulate_lp_problem(problem_description: str) -> str:
    """
    Help formulate a Linear Programming problem from a natural language description.

    Args:
        problem_description: Natural language description of the optimization problem.

    Returns:
        Structured guidance for formulating and solving the LP problem.
    """
    return formulate_lp(problem_description)


@mcp.prompt()
def interpret_lp_solution(
    status: str,
    objective_value: float,
    variable_values: str,
    problem_context: str = "",
) -> str:
    """
    Interpret LP solution results for decision makers.

    Args:
        status: Solver status (optimal, feasible, infeasible, unbounded).
        objective_value: The optimal objective function value.
        variable_values: JSON string of variable values.
        problem_context: Original problem description for context.

    Returns:
        Business-friendly interpretation of the solution.
    """
    import json

    values = (
        json.loads(variable_values)
        if isinstance(variable_values, str)
        else variable_values
    )
    return interpret_solution(status, objective_value, values, None, problem_context)


def main() -> None:
    """Run the Vertex MCP server."""
    import sys

    # Default to stdio for Claude Desktop, use --http flag for HTTP transport
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
