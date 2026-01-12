"""MCP tools for stochastic and dynamic optimization."""

from vertex.models.stochastic import (
    BinPacking2DResult,
    ChanceConstrainedResult,
    CrewScheduleResult,
    LotSizingResult,
    MonteCarloResult,
    MultiEchelonResult,
    NetworkDesignResult,
    NewsvendorResult,
    QAPResult,
    QueueMetrics,
    RobustResult,
    Scenario,
    SteinerTreeResult,
    TwoStageResult,
)
from vertex.solvers.stochastic import (
    solve_lot_sizing,
    solve_newsvendor,
    solve_two_stage_stochastic,
)


def compute_two_stage_stochastic(
    products: list[str],
    scenarios: list[dict],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    holding_costs: dict[str, float],
    capacity: dict[str, float] | None = None,
) -> TwoStageResult:
    """
    Solve two-stage stochastic program with recourse for production planning under demand uncertainty.
    
    First stage: Decide production quantities before demand is revealed.
    Second stage: After demand realizes, handle shortages/surpluses.
    
    Args:
        products: List of product names
        scenarios: List of scenarios, each with {name, probability, demand: {product: qty}}
        production_costs: Cost per unit to produce each product
        shortage_costs: Penalty per unit of unmet demand
        holding_costs: Cost per unit of excess inventory
        capacity: Optional production capacity limits
    
    Returns:
        Optimal first-stage production and expected recourse costs
    """
    scenario_objs = [Scenario(**s) for s in scenarios]
    return solve_two_stage_stochastic(
        products, scenario_objs, production_costs, shortage_costs, holding_costs, capacity
    )


def compute_newsvendor(
    selling_price: float,
    cost: float,
    salvage_value: float,
    mean_demand: float,
    std_demand: float,
) -> NewsvendorResult:
    """
    Solve the newsvendor (single-period stochastic inventory) problem.
    
    Classic OR model for perishable goods: how much to order when demand is uncertain
    and unsold items have salvage value.
    
    Args:
        selling_price: Revenue per unit sold
        cost: Purchase/production cost per unit
        salvage_value: Value recovered per unsold unit
        mean_demand: Expected demand (normal distribution)
        std_demand: Standard deviation of demand
    
    Returns:
        Optimal order quantity, expected profit, and critical ratio
    """
    return solve_newsvendor(selling_price, cost, salvage_value, mean_demand, std_demand)


def compute_lot_sizing(
    demands: list[float],
    setup_cost: float,
    holding_cost: float,
    production_cost: float = 0,
) -> LotSizingResult:
    """
    Solve dynamic lot sizing using Wagner-Whitin algorithm.
    
    Determines when and how much to produce over multiple periods to minimize
    total setup, production, and holding costs.
    
    Args:
        demands: Demand for each period
        setup_cost: Fixed cost incurred when production occurs
        holding_cost: Cost per unit per period held in inventory
        production_cost: Variable cost per unit produced (optional)
    
    Returns:
        Optimal production schedule and total cost
    """
    return solve_lot_sizing(demands, setup_cost, holding_cost, production_cost)


def solve_robust_optimization(
    products: list[str],
    nominal_demand: dict[str, float],
    demand_deviation: dict[str, float],
    uncertainty_budget: float,
    production_costs: dict[str, float],
    selling_prices: dict[str, float],
    capacity: dict[str, float] | None = None,
) -> RobustResult:
    """
    Solve robust optimization with budget uncertainty set.
    
    Protects against worst-case demand within uncertainty budget.
    Uses Bertsimas-Sim approach: at most Gamma parameters deviate.
    
    Args:
        products: Product names
        nominal_demand: Expected demand per product
        demand_deviation: Maximum deviation from nominal
        uncertainty_budget: Gamma - max number of deviating parameters
        production_costs: Cost per unit
        selling_prices: Revenue per unit sold
        capacity: Optional production limits
    
    Returns:
        Robust solution protecting against worst-case scenarios
    """
    from vertex.solvers.stochastic import solve_robust
    return solve_robust(
        products, nominal_demand, demand_deviation, uncertainty_budget,
        production_costs, selling_prices, capacity
    )


def analyze_queue_mm1(
    arrival_rate: float,
    service_rate: float,
) -> QueueMetrics:
    """
    Analyze M/M/1 queue (single server, Poisson arrivals, exponential service).
    
    Args:
        arrival_rate: Lambda - average arrivals per time unit
        service_rate: Mu - average services per time unit
    
    Returns:
        Queue performance metrics
    """
    from vertex.solvers.stochastic import compute_mm1_metrics
    return compute_mm1_metrics(arrival_rate, service_rate)


def analyze_queue_mmc(
    arrival_rate: float,
    service_rate: float,
    num_servers: int,
) -> QueueMetrics:
    """
    Analyze M/M/c queue (multiple servers, Poisson arrivals, exponential service).
    
    Args:
        arrival_rate: Lambda - average arrivals per time unit
        service_rate: Mu - average services per time unit per server
        num_servers: c - number of parallel servers
    
    Returns:
        Queue performance metrics
    """
    from vertex.solvers.stochastic import compute_mmc_metrics
    return compute_mmc_metrics(arrival_rate, service_rate, num_servers)


def simulate_newsvendor_monte_carlo(
    selling_price: float,
    cost: float,
    salvage_value: float,
    order_quantity: float,
    mean_demand: float,
    std_demand: float,
    num_simulations: int = 10000,
) -> "MonteCarloResult":
    """
    Run Monte Carlo simulation for newsvendor profit distribution.
    
    Evaluates a given order quantity under demand uncertainty.
    
    Args:
        selling_price: Revenue per unit sold
        cost: Purchase cost per unit
        salvage_value: Value per unsold unit
        order_quantity: Fixed order quantity to evaluate
        mean_demand: Expected demand
        std_demand: Demand standard deviation
        num_simulations: Number of simulation runs
    
    Returns:
        Profit distribution statistics and risk metrics
    """
    from vertex.solvers.stochastic import run_monte_carlo_newsvendor
    return run_monte_carlo_newsvendor(
        selling_price, cost, salvage_value, order_quantity,
        mean_demand, std_demand, num_simulations
    )


def simulate_production_monte_carlo(
    products: list[str],
    production_quantities: dict[str, float],
    mean_demands: dict[str, float],
    std_demands: dict[str, float],
    selling_prices: dict[str, float],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    num_simulations: int = 10000,
) -> "MonteCarloResult":
    """
    Run Monte Carlo simulation for multi-product production planning.
    
    Args:
        products: Product names
        production_quantities: Quantities to produce (decision to evaluate)
        mean_demands: Expected demand per product
        std_demands: Demand std dev per product
        selling_prices: Revenue per unit sold
        production_costs: Cost per unit produced
        shortage_costs: Penalty per unit of unmet demand
        num_simulations: Number of simulation runs
    
    Returns:
        Profit distribution statistics and risk metrics
    """
    from vertex.solvers.stochastic import run_monte_carlo_production
    return run_monte_carlo_production(
        products, production_quantities, mean_demands, std_demands,
        selling_prices, production_costs, shortage_costs, num_simulations
    )


def schedule_crew(
    workers: list[str],
    days: int,
    shifts: list[str],
    requirements: dict[str, list[int]],
    worker_availability: dict[str, list[tuple[int, str]]] | None = None,
    costs: dict[str, float] | None = None,
    max_shifts_per_worker: int | None = None,
    min_rest_between_shifts: int = 0,
    time_limit_seconds: int = 30,
) -> "CrewScheduleResult":
    """
    Solve crew/shift scheduling with constraints.
    
    Args:
        workers: Worker names
        days: Number of days to schedule
        shifts: Shift names (e.g., ["morning", "afternoon", "night"])
        requirements: shift -> [required workers per day]
        worker_availability: Optional worker -> [(day, shift)] availability
        costs: Cost per worker (default 1)
        max_shifts_per_worker: Maximum shifts per worker over period
        min_rest_between_shifts: Minimum rest periods between shifts
        time_limit_seconds: Solver time limit
    
    Returns:
        Worker assignments and coverage
    """
    from vertex.solvers.stochastic import solve_crew_scheduling
    return solve_crew_scheduling(
        workers, days, shifts, requirements, worker_availability,
        costs, max_shifts_per_worker, min_rest_between_shifts, time_limit_seconds
    )


def solve_chance_constrained_production(
    products: list[str],
    mean_demands: dict[str, float],
    std_demands: dict[str, float],
    production_costs: dict[str, float],
    selling_prices: dict[str, float],
    service_level: float = 0.95,
    capacity: dict[str, float] | None = None,
) -> "ChanceConstrainedResult":
    """
    Solve chance-constrained production planning.
    
    Ensures P(production >= demand) >= service_level for each product.
    
    Args:
        products: Product names
        mean_demands: Expected demand per product
        std_demands: Demand std dev per product
        production_costs: Cost per unit produced
        selling_prices: Revenue per unit sold
        service_level: Required probability of meeting demand (default 0.95)
        capacity: Optional production limits
    
    Returns:
        Production plan with constraint satisfaction probabilities
    """
    from vertex.solvers.stochastic import solve_chance_constrained
    return solve_chance_constrained(
        products, mean_demands, std_demands, production_costs,
        selling_prices, service_level, capacity
    )


def pack_rectangles_2d(
    rectangles: list[dict],
    bin_width: int,
    bin_height: int,
    max_bins: int | None = None,
    allow_rotation: bool = True,
    time_limit_seconds: int = 30,
) -> "BinPacking2DResult":
    """
    Solve 2D bin packing - pack rectangles into bins.
    
    Args:
        rectangles: List of {name, width, height}
        bin_width: Width of each bin
        bin_height: Height of each bin
        max_bins: Maximum bins available
        allow_rotation: Allow 90-degree rotation
        time_limit_seconds: Solver time limit
    
    Returns:
        Rectangle placements and bin utilization
    """
    from vertex.solvers.stochastic import solve_2d_bin_packing
    return solve_2d_bin_packing(
        rectangles, bin_width, bin_height, max_bins,
        allow_rotation, time_limit_seconds
    )


def design_network(
    nodes: list[str],
    potential_arcs: list[dict],
    commodities: list[dict],
    arc_fixed_costs: dict[str, float],
    arc_capacities: dict[str, float],
    arc_variable_costs: dict[str, float],
    time_limit_seconds: int = 30,
) -> "NetworkDesignResult":
    """
    Solve capacitated network design - decide which arcs to build.
    
    Args:
        nodes: Node names
        potential_arcs: List of {source, target} potential arcs
        commodities: List of {name, source, sink, demand}
        arc_fixed_costs: Fixed cost to open arc {"A->B": cost}
        arc_capacities: Capacity of each arc
        arc_variable_costs: Cost per unit flow
        time_limit_seconds: Solver time limit
    
    Returns:
        Which arcs to open and flow routing
    """
    from vertex.solvers.stochastic import solve_network_design
    # Convert string keys to tuples
    fixed = {tuple(k.split("->")):v for k,v in arc_fixed_costs.items()}
    caps = {tuple(k.split("->")):v for k,v in arc_capacities.items()}
    var = {tuple(k.split("->")):v for k,v in arc_variable_costs.items()}
    return solve_network_design(
        nodes, potential_arcs, commodities, fixed, caps, var, time_limit_seconds
    )


def solve_quadratic_assignment(
    facilities: list[str],
    locations: list[str],
    flow_matrix: dict[str, dict[str, float]],
    distance_matrix: dict[str, dict[str, float]],
    time_limit_seconds: int = 30,
) -> "QAPResult":
    """
    Solve Quadratic Assignment Problem - assign facilities to locations.
    
    Minimizes total cost = sum of flow[i][k] * distance[j][l] for all pairs
    where facility i is at location j and facility k is at location l.
    
    Args:
        facilities: Facility names
        locations: Location names (same count as facilities)
        flow_matrix: flow_matrix[f1][f2] = flow between facilities
        distance_matrix: distance_matrix[l1][l2] = distance between locations
        time_limit_seconds: Solver time limit
    
    Returns:
        Facility-to-location assignment minimizing flow*distance
    """
    from vertex.solvers.stochastic import solve_qap
    return solve_qap(facilities, locations, flow_matrix, distance_matrix, time_limit_seconds)


def find_steiner_tree(
    nodes: list[str],
    edges: list[dict],
    terminals: list[str],
    time_limit_seconds: int = 30,
) -> "SteinerTreeResult":
    """
    Solve Steiner Tree - connect terminal nodes with minimum total edge weight.
    
    May use non-terminal (Steiner) nodes if it reduces total cost.
    
    Args:
        nodes: All node names
        edges: List of {source, target, weight}
        terminals: Nodes that must be connected
        time_limit_seconds: Solver time limit
    
    Returns:
        Minimum weight tree connecting all terminals
    """
    from vertex.solvers.stochastic import solve_steiner_tree
    return solve_steiner_tree(nodes, edges, terminals, time_limit_seconds)


def optimize_multi_echelon_inventory(
    locations: list[str],
    parent: dict[str, str | None],
    demands: dict[str, float],
    lead_times: dict[str, float],
    holding_costs: dict[str, float],
    service_levels: dict[str, float],
) -> "MultiEchelonResult":
    """
    Optimize multi-echelon inventory - compute base-stock levels.
    
    Args:
        locations: Location names (warehouses, DCs, stores)
        parent: parent[loc] = upstream location (None for top)
        demands: Mean demand per period at each location
        lead_times: Replenishment lead time for each location
        holding_costs: Holding cost per unit at each location
        service_levels: Target service level (fill rate) per location
    
    Returns:
        Base-stock levels and expected fill rates
    """
    from vertex.solvers.stochastic import solve_multi_echelon_inventory
    return solve_multi_echelon_inventory(
        locations, parent, demands, lead_times, holding_costs, service_levels
    )
