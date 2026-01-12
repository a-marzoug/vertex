"""Vertex MCP Server - Operations Research tools for decision makers."""

from mcp.server.fastmcp import FastMCP

from vertex.config import DEFAULT_HOST, DEFAULT_PORT, SERVER_DESCRIPTION, SERVER_NAME
from vertex.models.linear import LPSolution
from vertex.models.mip import MIPSolution
from vertex.models.network import MaxFlowResult, MinCostFlowResult, MSTResult, MultiCommodityFlowResult, ShortestPathResult
from vertex.models.scheduling import BinPackingResult, CuttingStockResult, GraphColoringResult, JobShopResult, SetCoverResult, TSPResult, VRPResult
from vertex.prompts.linear import formulate_lp, interpret_solution
from vertex.prompts.mip import formulate_mip
from vertex.tools.analysis import (
    InfeasibilityResult,
    ModelStats,
    WhatIfResult,
    analyze_what_if as _analyze_what_if,
    diagnose_infeasibility as _diagnose_infeasibility,
    find_alternative_solutions as _find_alternatives,
    get_model_stats as _get_model_stats,
    solve_rcpsp as _solve_rcpsp,
)
from vertex.tools.cp import NQueensResult, SudokuResult, solve_n_queens, solve_sudoku
from vertex.tools.linear import solve_lp
from vertex.tools.mip import solve_mip
from vertex.tools.multiobjective import MultiObjectiveResult, solve_multi_objective
from vertex.tools.network import compute_max_flow, compute_min_cost_flow, compute_mst, compute_multi_commodity_flow, compute_shortest_path
from vertex.tools.scheduling import compute_flexible_job_shop
from vertex.tools.scheduling import (
    compute_bin_packing,
    compute_cutting_stock,
    compute_graph_coloring,
    compute_job_shop,
    compute_set_cover,
    compute_tsp,
    compute_vrp,
    compute_vrp_tw,
)
from vertex.tools.sensitivity import SensitivityReport, analyze_sensitivity
from vertex.tools.stochastic import (
    compute_lot_sizing,
    compute_newsvendor,
    compute_two_stage_stochastic,
)
from vertex.models.stochastic import LotSizingResult, NewsvendorResult, TwoStageResult
from vertex.tools.templates.assignment import AssignmentResult
from vertex.tools.templates.assignment import optimize_assignment as _optimize_assignment
from vertex.tools.templates.diet import DietResult
from vertex.tools.templates.diet import optimize_diet as _optimize_diet
from vertex.tools.templates.facility import FacilityResult
from vertex.tools.templates.facility import optimize_facility_location as _optimize_facility
from vertex.tools.templates.healthcare import ResourceAllocationResult, optimize_resource_allocation
from vertex.tools.templates.inventory import EOQResult, optimize_eoq
from vertex.tools.templates.knapsack import KnapsackResult
from vertex.tools.templates.knapsack import optimize_knapsack as _optimize_knapsack
from vertex.tools.templates.portfolio import PortfolioResult, optimize_portfolio
from vertex.tools.templates.production import ProductionResult, optimize_production
from vertex.tools.templates.supplychain import SupplyChainResult, optimize_supply_chain
from vertex.tools.templates.workforce import WorkforceResult, optimize_workforce_schedule

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
def analyze_what_if_scenario(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    parameter_name: str,
    parameter_values: list[float],
    objective_sense: str = "maximize",
) -> WhatIfResult:
    """
    Perform what-if analysis by varying a constraint RHS value.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions (one must have name=parameter_name).
        objective_coefficients: Objective function coefficients.
        parameter_name: Name of constraint to vary.
        parameter_values: Values to test for the constraint RHS.
        objective_sense: "maximize" or "minimize".
    """
    return _analyze_what_if(variables, constraints, objective_coefficients, parameter_name, parameter_values, objective_sense)


@mcp.tool()
def diagnose_infeasibility(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> InfeasibilityResult:
    """
    Diagnose why a problem is infeasible by finding conflicting constraints.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.
        objective_coefficients: Objective function coefficients.
        objective_sense: "maximize" or "minimize".
    """
    return _diagnose_infeasibility(variables, constraints, objective_coefficients, objective_sense)


@mcp.tool()
def get_model_statistics(
    variables: list[dict],
    constraints: list[dict],
) -> ModelStats:
    """
    Get statistics about an optimization model.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.

    Returns:
        Model size, sparsity, and type breakdown.
    """
    return _get_model_stats(variables, constraints)


@mcp.tool()
def solve_pareto_frontier(
    variables: list[dict],
    constraints: list[dict],
    objectives: dict[str, dict[str, float]],
    num_points: int = 10,
    objective_senses: dict[str, str] | None = None,
) -> MultiObjectiveResult:
    """
    Solve multi-objective optimization and find Pareto frontier.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.
        objectives: Dict of objective_name -> {var: coef} mappings.
        num_points: Number of Pareto points to generate.
        objective_senses: Dict of objective_name -> "maximize"/"minimize".
    """
    return solve_multi_objective(variables, constraints, objectives, num_points, objective_senses)


@mcp.tool()
def solve_sudoku_puzzle(grid: list[list[int]]) -> SudokuResult:
    """
    Solve a Sudoku puzzle using constraint programming.

    Args:
        grid: 9x9 grid with 0 for empty cells, 1-9 for filled cells.
    """
    return solve_sudoku(grid)


@mcp.tool()
def solve_n_queens_puzzle(n: int) -> NQueensResult:
    """
    Solve N-Queens problem - place N queens on NxN board with no attacks.

    Args:
        n: Board size and number of queens.
    """
    return solve_n_queens(n)


@mcp.tool()
def find_alternative_optimal_solutions(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
    max_solutions: int = 5,
    gap_tolerance: float = 0.01,
) -> list[dict]:
    """
    Find multiple near-optimal solutions for MIP problems.

    Args:
        variables: Variable definitions with var_type (binary/integer).
        constraints: Constraint definitions.
        objective_coefficients: Objective function coefficients.
        objective_sense: "maximize" or "minimize".
        max_solutions: Maximum solutions to return.
        gap_tolerance: Accept solutions within this fraction of optimal.
    """
    return _find_alternatives(variables, constraints, objective_coefficients, objective_sense, max_solutions, gap_tolerance)


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


@mcp.tool()
def optimize_inventory_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
    lead_time_days: float = 0,
    safety_stock: float = 0,
) -> EOQResult:
    """
    Calculate Economic Order Quantity - optimal order size minimizing total cost.

    Args:
        annual_demand: Annual demand in units.
        ordering_cost: Fixed cost per order.
        holding_cost_per_unit: Annual holding cost per unit.
        lead_time_days: Lead time for reorder point.
        safety_stock: Safety stock units.
    """
    return optimize_eoq(annual_demand, ordering_cost, holding_cost_per_unit, lead_time_days, safety_stock)


@mcp.tool()
def optimize_workforce(
    workers: list[str],
    days: int,
    shifts: list[str],
    requirements: dict[str, list[int]],
    costs: dict[str, float] | None = None,
    max_shifts_per_worker: int | None = None,
    time_limit_seconds: int = 30,
) -> WorkforceResult:
    """
    Schedule workers to shifts minimizing cost while meeting requirements.

    Args:
        workers: Worker names.
        days: Number of days to schedule.
        shifts: Shift names (e.g., ["morning", "afternoon", "night"]).
        requirements: Shift -> list of required workers per day.
        costs: Cost per worker (defaults to 1).
        max_shifts_per_worker: Maximum shifts per worker.
        time_limit_seconds: Solver time limit.
    """
    return optimize_workforce_schedule(workers, days, shifts, requirements, costs, max_shifts_per_worker, time_limit_seconds)


@mcp.tool()
def optimize_healthcare_resources(
    resources: list[str],
    locations: list[str],
    availability: dict[str, float],
    demands: dict[str, dict[str, float]],
    effectiveness: dict[str, dict[str, float]] | None = None,
    min_coverage: dict[str, float] | None = None,
) -> ResourceAllocationResult:
    """
    Allocate healthcare resources across locations to maximize coverage.

    Args:
        resources: Resource types (e.g., ["doctors", "nurses", "beds"]).
        locations: Location names (e.g., ["hospital_A", "clinic_B"]).
        availability: Total available units per resource.
        demands: demands[location][resource] = units needed.
        effectiveness: Optional effectiveness multiplier per location/resource.
        min_coverage: Optional minimum coverage ratio per location.
    """
    return optimize_resource_allocation(resources, locations, availability, demands, effectiveness, min_coverage)


@mcp.tool()
def optimize_supply_chain_network(
    facilities: list[str],
    customers: list[str],
    fixed_costs: dict[str, float],
    capacities: dict[str, float],
    demands: dict[str, float],
    transport_costs: dict[str, dict[str, float]],
    max_facilities: int | None = None,
) -> SupplyChainResult:
    """
    Design supply chain network - decide which facilities to open and how to serve customers.

    Args:
        facilities: Potential facility locations.
        customers: Customer locations.
        fixed_costs: Fixed cost to open each facility.
        capacities: Capacity of each facility.
        demands: Demand at each customer.
        transport_costs: transport_costs[facility][customer] = unit cost.
        max_facilities: Optional limit on facilities to open.
    """
    return optimize_supply_chain(facilities, customers, fixed_costs, capacities, demands, transport_costs, max_facilities)


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


@mcp.tool()
def find_minimum_spanning_tree(
    nodes: list[str],
    edges: list[dict],
) -> MSTResult:
    """
    Find Minimum Spanning Tree connecting all nodes with minimum total weight.

    Args:
        nodes: Node names.
        edges: Edges with 'source', 'target', 'weight'.
    """
    return compute_mst(nodes, edges)


@mcp.tool()
def find_multi_commodity_flow(
    nodes: list[str],
    arcs: list[dict],
    commodities: list[dict],
    time_limit_seconds: int = 30,
) -> MultiCommodityFlowResult:
    """
    Solve Multi-Commodity Flow - route multiple commodities through shared network.

    Args:
        nodes: Node names.
        arcs: Arcs with 'source', 'target', 'capacity', 'cost'.
        commodities: List with 'name', 'source', 'sink', 'demand'.
        time_limit_seconds: Solver time limit.
    """
    return compute_multi_commodity_flow(nodes, arcs, commodities, time_limit_seconds)


@mcp.tool()
def solve_transshipment(
    sources: list[str],
    transshipment_nodes: list[str],
    destinations: list[str],
    supplies: dict[str, int],
    demands: dict[str, int],
    costs: dict[str, dict[str, float]],
    capacities: dict[str, dict[str, float]] | None = None,
) -> MinCostFlowResult:
    """
    Solve Transshipment Problem - ship goods through intermediate nodes.

    Args:
        sources: Source nodes (e.g., factories).
        transshipment_nodes: Intermediate nodes (e.g., warehouses).
        destinations: Destination nodes (e.g., customers).
        supplies: Supply at each source.
        demands: Demand at each destination.
        costs: costs[from][to] = unit shipping cost.
        capacities: Optional capacities[from][to] = max flow.
    """
    from vertex.tools.network import compute_transshipment
    return compute_transshipment(sources, transshipment_nodes, destinations, supplies, demands, costs, capacities)


# Scheduling & Routing Tools
@mcp.tool()
def solve_tsp(
    locations: list[str],
    distance_matrix: list[list[float]],
    time_limit_seconds: int = 30,
) -> TSPResult:
    """
    Solve Traveling Salesman Problem - find shortest tour visiting all locations.

    Args:
        locations: Location names. First location is start/end point.
        distance_matrix: distances[i][j] = distance from location i to j.
        time_limit_seconds: Solver time limit.
    """
    return compute_tsp(locations, distance_matrix, time_limit_seconds)


@mcp.tool()
def solve_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
) -> VRPResult:
    """
    Solve Capacitated Vehicle Routing Problem.

    Args:
        locations: Location names. Index 0 is typically the depot.
        distance_matrix: distances[i][j] = distance from location i to j.
        demands: Demand at each location (depot demand should be 0).
        vehicle_capacities: Capacity of each vehicle.
        depot: Index of depot location.
        time_limit_seconds: Solver time limit.
    """
    return compute_vrp(locations, distance_matrix, demands, vehicle_capacities, depot, time_limit_seconds)


@mcp.tool()
def solve_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> JobShopResult:
    """
    Solve Job Shop Scheduling - schedule jobs on machines minimizing makespan.

    Args:
        jobs: List of jobs. Each job is list of tasks: {"machine": int, "duration": int}.
            Tasks within a job must be processed in order.
        time_limit_seconds: Solver time limit.

    Example:
        jobs = [
            [{"machine": 0, "duration": 3}, {"machine": 1, "duration": 2}],
            [{"machine": 1, "duration": 4}, {"machine": 0, "duration": 2}],
        ]
    """
    return compute_job_shop(jobs, time_limit_seconds)


@mcp.tool()
def solve_rcpsp(
    tasks: list[dict],
    resources: dict[str, int],
    time_limit_seconds: int = 30,
) -> dict:
    """
    Solve Resource-Constrained Project Scheduling Problem.

    Args:
        tasks: List of tasks with 'name', 'duration', 'resources' (dict), 'predecessors' (list).
            Example: {"name": "A", "duration": 3, "resources": {"workers": 2}, "predecessors": []}
        resources: Available capacity per resource type.
            Example: {"workers": 4, "machines": 2}
        time_limit_seconds: Solver time limit.
    """
    return _solve_rcpsp(tasks, resources, time_limit_seconds)


@mcp.tool()
def solve_flexible_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> dict:
    """
    Solve Flexible Job Shop - tasks can run on alternative machines.

    Args:
        jobs: List of jobs. Each job is list of tasks.
            Each task: {"machines": [(machine_id, duration), ...]}
        time_limit_seconds: Solver time limit.

    Example:
        jobs = [
            [{"machines": [(0, 3), (1, 2)]}, {"machines": [(1, 4)]}],  # Job 0
            [{"machines": [(0, 2), (1, 3)]}, {"machines": [(0, 3)]}],  # Job 1
        ]
    """
    return compute_flexible_job_shop(jobs, time_limit_seconds)


@mcp.tool()
def solve_vrp_time_windows(
    locations: list[str],
    time_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
) -> VRPResult:
    """
    Solve VRP with Time Windows - vehicles must arrive within time windows.

    Args:
        locations: Location names.
        time_matrix: time_matrix[i][j] = travel time from i to j.
        time_windows: (earliest, latest) arrival time for each location.
        demands: Demand at each location.
        vehicle_capacities: Capacity of each vehicle.
        depot: Index of depot location.
        time_limit_seconds: Solver time limit.
    """
    return compute_vrp_tw(locations, time_matrix, time_windows, demands, vehicle_capacities, depot, time_limit_seconds)


@mcp.tool()
def solve_bin_packing(
    items: list[str],
    weights: dict[str, float],
    bin_capacity: float,
    max_bins: int | None = None,
    time_limit_seconds: int = 30,
) -> BinPackingResult:
    """
    Solve Bin Packing - pack items into minimum number of bins.

    Args:
        items: Item names.
        weights: Weight of each item.
        bin_capacity: Capacity of each bin.
        max_bins: Maximum bins available.
        time_limit_seconds: Solver time limit.
    """
    return compute_bin_packing(items, weights, bin_capacity, max_bins, time_limit_seconds)


@mcp.tool()
def solve_set_cover(
    universe: list[str],
    sets: dict[str, list[str]],
    costs: dict[str, float],
    time_limit_seconds: int = 30,
) -> SetCoverResult:
    """
    Solve Set Covering - select minimum cost sets to cover all elements.

    Args:
        universe: Elements that must be covered.
        sets: Available sets mapping to elements they cover.
        costs: Cost of each set.
        time_limit_seconds: Solver time limit.
    """
    return compute_set_cover(universe, sets, costs, time_limit_seconds)


@mcp.tool()
def solve_graph_coloring(
    nodes: list[str],
    edges: list[tuple[str, str]],
    max_colors: int | None = None,
    time_limit_seconds: int = 30,
) -> GraphColoringResult:
    """
    Solve Graph Coloring - assign colors so adjacent nodes differ.

    Args:
        nodes: Node names.
        edges: List of (node1, node2) edges.
        max_colors: Maximum colors available.
        time_limit_seconds: Solver time limit.
    """
    return compute_graph_coloring(nodes, edges, max_colors, time_limit_seconds)


@mcp.tool()
def solve_cutting_stock(
    items: list[str],
    lengths: dict[str, int],
    demands: dict[str, int],
    stock_length: int,
    max_stock: int | None = None,
    time_limit_seconds: int = 30,
) -> CuttingStockResult:
    """
    Solve Cutting Stock - cut items from stock minimizing waste.

    Args:
        items: Item names.
        lengths: Length of each item type.
        demands: Number of each item needed.
        stock_length: Length of each stock piece.
        max_stock: Maximum stock pieces available.
        time_limit_seconds: Solver time limit.
    """
    return compute_cutting_stock(items, lengths, demands, stock_length, max_stock, time_limit_seconds)


# Stochastic & Dynamic Optimization Tools
@mcp.tool()
def solve_two_stage_stochastic(
    products: list[str],
    scenarios: list[dict],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    holding_costs: dict[str, float],
    capacity: dict[str, float] | None = None,
) -> TwoStageResult:
    """
    Solve two-stage stochastic program for production under demand uncertainty.

    First stage: Decide production before demand is known.
    Second stage: Handle shortages/surpluses after demand realizes.

    Args:
        products: Product names.
        scenarios: List of {name, probability, demand: {product: qty}}.
        production_costs: Cost per unit to produce.
        shortage_costs: Penalty per unit of unmet demand.
        holding_costs: Cost per unit of excess inventory.
        capacity: Optional production capacity limits.
    """
    return compute_two_stage_stochastic(products, scenarios, production_costs, shortage_costs, holding_costs, capacity)


@mcp.tool()
def solve_newsvendor(
    selling_price: float,
    cost: float,
    salvage_value: float,
    mean_demand: float,
    std_demand: float,
) -> NewsvendorResult:
    """
    Solve newsvendor (single-period stochastic inventory) problem.

    Classic model for perishable goods: how much to order when demand is uncertain.

    Args:
        selling_price: Revenue per unit sold.
        cost: Purchase/production cost per unit.
        salvage_value: Value recovered per unsold unit.
        mean_demand: Expected demand (normal distribution).
        std_demand: Standard deviation of demand.
    """
    return compute_newsvendor(selling_price, cost, salvage_value, mean_demand, std_demand)


@mcp.tool()
def solve_lot_sizing(
    demands: list[float],
    setup_cost: float,
    holding_cost: float,
    production_cost: float = 0,
) -> LotSizingResult:
    """
    Solve dynamic lot sizing using Wagner-Whitin algorithm.

    Determines when and how much to produce over multiple periods.

    Args:
        demands: Demand for each period.
        setup_cost: Fixed cost when production occurs.
        holding_cost: Cost per unit per period in inventory.
        production_cost: Variable cost per unit (optional).
    """
    return compute_lot_sizing(demands, setup_cost, holding_cost, production_cost)


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
