"""MCP tools for stochastic and dynamic optimization."""

from vertex.models.stochastic import (
    LotSizingResult,
    NewsvendorResult,
    QueueMetrics,
    RobustResult,
    Scenario,
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
