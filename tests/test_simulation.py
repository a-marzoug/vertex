"""Tests for Simulation Optimization."""

import pytest

from vertex.tools.simulation import optimize_simulation_parameters


@pytest.mark.asyncio
async def test_optimize_newsvendor_simulation():
    """Optimize newsvendor order quantity via simulation."""
    # Newsvendor: mean=100, std=20, cost=10, price=15.
    # Analytical optimal: critical ratio = (15-10)/15 = 0.333.
    # z = norm.ppf(0.333) approx -0.43.
    # Q* = 100 - 0.43 * 20 = 91.4.

    fixed_args = {
        "demand_mean": 100,
        "demand_std": 20,
        "unit_cost": 10,
        "unit_price": 15,
        "holding_cost": 0,  # Simplify
    }

    # We want to maximize profit.
    # simulate_newsvendor returns NewsvendorResult which has expected_profit.

    params = [
        {
            "name": "order_quantity",
            "lower_bound": 50,
            "upper_bound": 150,
            "is_integer": True,
        }
    ]

    result = await optimize_simulation_parameters(
        simulation_name="simulate_newsvendor",
        parameters=params,
        fixed_arguments=fixed_args,
        objective_attribute="expected_profit",
        objective_sense="maximize",
        n_simulations_per_eval=500,  # Higher for accuracy
        max_evaluations=30,
    )

    assert result.status == "optimal"
    q_opt = result.optimal_parameters["order_quantity"]
    # Check if close to ~91. Allow wide margin due to noise and DE
    # 80 to 105 covers reasonable noise range for 500 sims
    assert 80 <= q_opt <= 105
