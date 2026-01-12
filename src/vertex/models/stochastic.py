"""Models for stochastic and dynamic optimization problems."""

from pydantic import BaseModel


class Scenario(BaseModel):
    """A scenario with probability and parameter values."""
    name: str
    probability: float
    demand: dict[str, float]  # product -> demand


class TwoStageResult(BaseModel):
    """Result of two-stage stochastic programming."""
    status: str
    expected_cost: float
    first_stage_decisions: dict[str, float]  # e.g., production quantities
    recourse_decisions: dict[str, dict[str, float]]  # scenario -> variable -> value
    solve_time: float


class NewsvendorResult(BaseModel):
    """Result of newsvendor model."""
    status: str
    optimal_order_quantity: float
    expected_profit: float
    critical_ratio: float
    stockout_probability: float


class LotSizingResult(BaseModel):
    """Result of dynamic lot sizing (Wagner-Whitin)."""
    status: str
    total_cost: float
    production_plan: list[float]  # quantity per period
    inventory_levels: list[float]  # ending inventory per period
    setup_periods: list[int]  # periods with production
