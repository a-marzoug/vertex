"""Vertex MCP Server - Operations Research tools for decision makers."""

from mcp.server.fastmcp import FastMCP

from vertex.config import DEFAULT_HOST, DEFAULT_PORT, SERVER_DESCRIPTION, SERVER_NAME
from vertex.models.linear import LPSolution
from vertex.models.mip import MIPSolution
from vertex.models.network import MaxFlowResult, MinCostFlowResult, ShortestPathResult
from vertex.prompts.linear import formulate_lp, interpret_solution
from vertex.prompts.mip import formulate_mip
from vertex.tools.linear import solve_lp
from vertex.tools.mip import solve_mip
from vertex.tools.network import compute_max_flow, compute_min_cost_flow, compute_shortest_path
from vertex.tools.sensitivity import SensitivityReport, analyze_sensitivity
from vertex.tools.templates.assignment import AssignmentResult
from vertex.tools.templates.assignment import optimize_assignment as _optimize_assignment
from vertex.tools.templates.diet import DietResult
from vertex.tools.templates.diet import optimize_diet as _optimize_diet
from vertex.tools.templates.facility import FacilityResult
from vertex.tools.templates.facility import optimize_facility_location as _optimize_facility
from vertex.tools.templates.knapsack import KnapsackResult
from vertex.tools.templates.knapsack import optimize_knapsack as _optimize_knapsack
from vertex.tools.templates.portfolio import PortfolioResult, optimize_portfolio
from vertex.tools.templates.production import ProductionResult, optimize_production

mcp = FastMCP(
    SERVER_NAME,
    instructions=SERVER_DESCRIPTION,
    stateless_http=True,
    json_response=True,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
)


# LP Tools
@mcp.tool()
def solve_linear_program(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> LPSolution:
    """Solve a Linear Programming problem with continuous variables."""
    return solve_lp(variables, constraints, objective_coefficients, objective_sense)


@mcp.tool()
def analyze_lp_sensitivity(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> SensitivityReport:
    """
    Analyze LP solution sensitivity to parameter changes.

    Returns shadow prices (marginal value of constraints) and reduced costs
    (how much variable coefficients must improve to enter the solution).
    """
    return analyze_sensitivity(variables, constraints, objective_coefficients, objective_sense)


@mcp.tool()
def optimize_production_plan(
    products: list[str],
    resources: list[str],
    profits: dict[str, float],
    requirements: dict[str, dict[str, float]],
    availability: dict[str, float],
) -> ProductionResult:
    """Maximize profit given resource constraints."""
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
    """Find minimum cost diet meeting nutritional requirements."""
    return optimize_diet(foods, nutrients, costs, nutrition_values, min_requirements, max_limits)


@mcp.tool()
def optimize_investment_portfolio(
    assets: list[str],
    expected_returns: dict[str, float],
    budget: float,
    min_allocation: dict[str, float] | None = None,
    max_allocation: dict[str, float] | None = None,
) -> PortfolioResult:
    """Maximize expected return with allocation constraints."""
    return optimize_portfolio(assets, expected_returns, budget, min_allocation, max_allocation)


# MIP Tools
@mcp.tool()
def solve_mixed_integer_program(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> MIPSolution:
    """Solve a Mixed-Integer Programming problem with integer/binary variables."""
    return solve_mip(variables, constraints, objective_coefficients, objective_sense)


@mcp.tool()
def optimize_worker_assignment(
    workers: list[str],
    tasks: list[str],
    costs: dict[str, dict[str, float]],
) -> AssignmentResult:
    """Assign workers to tasks minimizing total cost. Each worker gets one task."""
    return _optimize_assignment(workers, tasks, costs)


@mcp.tool()
def optimize_knapsack_selection(
    items: list[str],
    values: dict[str, float],
    weights: dict[str, float],
    capacity: float,
) -> KnapsackResult:
    """Select items to maximize value within weight capacity (0/1 knapsack)."""
    return _optimize_knapsack(items, values, weights, capacity)


@mcp.tool()
def optimize_facility_locations(
    facilities: list[str],
    customers: list[str],
    fixed_costs: dict[str, float],
    transport_costs: dict[str, dict[str, float]],
) -> FacilityResult:
    """Decide which facilities to open and assign customers to minimize cost."""
    return _optimize_facility(facilities, customers, fixed_costs, transport_costs)


# Network Tools
@mcp.tool()
def find_max_flow(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    sink: str,
) -> MaxFlowResult:
    """
    Find maximum flow from source to sink in a network.

    Args:
        nodes: Node names. Example: ["S", "A", "B", "T"]
        arcs: Arcs with 'source', 'target', 'capacity'.
            Example: [{"source": "S", "target": "A", "capacity": 10}]
        source: Source node.
        sink: Sink node.
    """
    return compute_max_flow(nodes, arcs, source, sink)


@mcp.tool()
def find_min_cost_flow(
    nodes: list[str],
    arcs: list[dict],
    supplies: dict[str, int],
) -> MinCostFlowResult:
    """
    Find minimum cost flow satisfying supplies and demands.

    Args:
        nodes: Node names.
        arcs: Arcs with 'source', 'target', 'capacity', 'cost'.
        supplies: Node supplies (positive) and demands (negative).
            Example: {"factory": 100, "warehouse": -100}
    """
    return compute_min_cost_flow(nodes, arcs, supplies)


@mcp.tool()
def find_shortest_path(
    nodes: list[str],
    arcs: list[dict],
    source: str,
    target: str,
) -> ShortestPathResult:
    """
    Find shortest path between two nodes.

    Args:
        nodes: Node names.
        arcs: Arcs with 'source', 'target', 'cost'.
        source: Start node.
        target: End node.
    """
    return compute_shortest_path(nodes, arcs, source, target)


# Prompts
@mcp.prompt()
def formulate_lp_problem(problem_description: str) -> str:
    """Help formulate a Linear Programming problem from natural language."""
    return formulate_lp(problem_description)


@mcp.prompt()
def formulate_mip_problem(problem_description: str) -> str:
    """Help formulate a Mixed-Integer Programming problem from natural language."""
    return formulate_mip(problem_description)


@mcp.prompt()
def interpret_lp_solution(
    status: str,
    objective_value: float,
    variable_values: str,
    problem_context: str = "",
) -> str:
    """Interpret optimization solution for decision makers."""
    import json

    values = json.loads(variable_values) if isinstance(variable_values, str) else variable_values
    return interpret_solution(status, objective_value, values, None, problem_context)


def main() -> None:
    """Run the Vertex MCP server."""
    import sys

    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
