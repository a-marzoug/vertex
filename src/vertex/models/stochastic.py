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


class RobustResult(BaseModel):
    """Result of robust optimization."""
    status: str
    objective_value: float
    worst_case_objective: float
    variable_values: dict[str, float]
    binding_scenarios: list[str]
    solve_time: float


class QueueMetrics(BaseModel):
    """Queueing system performance metrics."""
    utilization: float  # rho = lambda / (s * mu)
    avg_queue_length: float  # Lq
    avg_system_length: float  # L
    avg_wait_time: float  # Wq
    avg_system_time: float  # W
    prob_wait: float  # P(wait > 0)
    prob_empty: float  # P(0 customers)
