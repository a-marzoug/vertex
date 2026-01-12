"""MCP tools for stochastic and dynamic optimization."""

from vertex.models.stochastic import (
    LotSizingResult,
    NewsvendorResult,
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
