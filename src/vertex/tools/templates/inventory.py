"""Inventory optimization template - Economic Order Quantity and variants."""

import math

from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class EOQResult(BaseModel):
    """Result of Economic Order Quantity calculation."""

    status: SolverStatus
    optimal_order_quantity: float | None = None
    annual_ordering_cost: float | None = None
    annual_holding_cost: float | None = None
    total_annual_cost: float | None = None
    orders_per_year: float | None = None
    reorder_point: float | None = None


class MultiItemInventoryResult(BaseModel):
    """Result of multi-item inventory optimization."""

    status: SolverStatus
    order_quantities: dict[str, float] = Field(default_factory=dict)
    total_cost: float | None = None
    item_costs: dict[str, float] = Field(default_factory=dict)


def optimize_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
    lead_time_days: float = 0,
    safety_stock: float = 0,
) -> EOQResult:
    """
    Calculate Economic Order Quantity (EOQ) - optimal order size minimizing total cost.

    Args:
        annual_demand: Annual demand in units.
        ordering_cost: Fixed cost per order.
        holding_cost_per_unit: Annual holding cost per unit.
        lead_time_days: Lead time in days for reorder point calculation.
        safety_stock: Safety stock units.

    Returns:
        EOQResult with optimal order quantity and costs.
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return EOQResult(status=SolverStatus.INFEASIBLE)

    # EOQ formula: Q* = sqrt(2*D*S/H)
    eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)

    orders_per_year = annual_demand / eoq
    annual_ordering = orders_per_year * ordering_cost
    annual_holding = (eoq / 2) * holding_cost_per_unit
    total_cost = annual_ordering + annual_holding

    # Reorder point = (daily demand * lead time) + safety stock
    daily_demand = annual_demand / 365
    reorder_point = (daily_demand * lead_time_days) + safety_stock

    return EOQResult(
        status=SolverStatus.OPTIMAL,
        optimal_order_quantity=round(eoq, 2),
        annual_ordering_cost=round(annual_ordering, 2),
        annual_holding_cost=round(annual_holding, 2),
        total_annual_cost=round(total_cost, 2),
        orders_per_year=round(orders_per_year, 2),
        reorder_point=round(reorder_point, 2),
    )


def optimize_multi_item_inventory(
    items: list[str],
    annual_demands: dict[str, float],
    ordering_costs: dict[str, float],
    holding_costs: dict[str, float],
    budget: float | None = None,
) -> MultiItemInventoryResult:
    """
    Optimize inventory for multiple items with optional budget constraint.

    Args:
        items: Item names.
        annual_demands: Annual demand per item.
        ordering_costs: Ordering cost per item.
        holding_costs: Annual holding cost per unit per item.
        budget: Optional total inventory budget.

    Returns:
        MultiItemInventoryResult with order quantities per item.
    """
    order_quantities = {}
    item_costs = {}
    total_cost = 0

    for item in items:
        d = annual_demands[item]
        s = ordering_costs[item]
        h = holding_costs[item]

        if d <= 0 or s <= 0 or h <= 0:
            continue

        eoq = math.sqrt(2 * d * s / h)
        cost = math.sqrt(2 * d * s * h)

        order_quantities[item] = round(eoq, 2)
        item_costs[item] = round(cost, 2)
        total_cost += cost

    return MultiItemInventoryResult(
        status=SolverStatus.OPTIMAL,
        order_quantities=order_quantities,
        total_cost=round(total_cost, 2),
        item_costs=item_costs,
    )
