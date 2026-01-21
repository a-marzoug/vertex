"""Test maintenance planning tool."""

import pytest

from vertex.tools.maintenance import optimize_equipment_replacement


@pytest.mark.asyncio
async def test_optimize_equipment_replacement():
    """Test optimal replacement policy."""
    # Simple case: machine deteriorates.
    # State 0: New
    # State 1: Deteriorating
    # State 2: Failed

    # Transition matrix (Keep action)
    # 0 -> 0 (0.5), 1 (0.5)
    # 1 -> 1 (0.5), 2 (0.5)
    # 2 -> 2 (1.0)
    transition_matrix = [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ]

    # Costs
    cost_operating = [10, 50, 0]  # Low cost when new, high when deteriorating
    cost_replacement = 100
    cost_failure = 500  # Huge penalty

    result = await optimize_equipment_replacement(
        num_states=3,
        transition_matrix=transition_matrix,
        cost_operating=cost_operating,
        cost_replacement=cost_replacement,
        cost_failure=cost_failure,
        horizon=5,
        discount_factor=0.9,
    )

    assert result.status == "optimal"
    assert len(result.policy) > 0
    assert result.total_expected_cost > 0

    # Check logic: In state 2 (Failed), we should Replace.
    failed_policies = [p for p in result.policy if p.state == 2]
    # For the last period, maybe we don't replace if horizon ends?
    # But usually with high failure cost, we replace.
    # In finite horizon, at t=T (end), cost is 0.
    # At t=T-1, if in state 2:
    #   Keep: cost_op[2] + cost_fail = 0 + 500 = 500.
    #   Replace: cost_repl + 0 = 100.
    #   So Replace is better.

    for p in failed_policies:
        # We check periods before the very end where replacement makes sense
        if p.time_step < 5:
            assert p.action == "Replace"
