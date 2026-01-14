"""Tests for stochastic optimization."""

from vertex.models.stochastic import Scenario
from vertex.solvers.stochastic import solve_newsvendor, solve_two_stage_stochastic


def test_two_stage_stochastic():
    """Test two-stage stochastic programming."""
    scenarios = [
        Scenario(name="low", probability=0.3, demand={"P1": 50}),
        Scenario(name="high", probability=0.7, demand={"P1": 100}),
    ]

    result = solve_two_stage_stochastic(
        products=["P1"],
        scenarios=scenarios,
        production_costs={"P1": 10},
        shortage_costs={"P1": 50},
        holding_costs={"P1": 2},
    )

    assert result.status == "OPTIMAL"
    assert result.first_stage_decisions is not None


def test_newsvendor():
    """Test newsvendor problem."""
    result = solve_newsvendor(
        selling_price=20,
        cost=10,
        salvage_value=5,
        mean_demand=100,
        std_demand=20,
    )

    assert result.status == "OPTIMAL"
    assert result.optimal_order_quantity is not None
    assert result.optimal_order_quantity > 0
