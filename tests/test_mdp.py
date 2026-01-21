"""Test generic MDP solver."""

import pytest

from vertex.tools.mdp import solve_discrete_mdp


@pytest.mark.asyncio
async def test_solve_grid_world_mdp():
    """Test simple Grid World MDP."""
    # States: s0, s1, s2(goal)
    # s0 -> s1 -> s2
    states = ["s0", "s1", "s2"]
    actions = ["right", "stay"]

    # Transitions
    transitions = {
        "s0": {
            "right": {"s1": 1.0},
            "stay": {"s0": 1.0},
        },
        "s1": {
            "right": {"s2": 1.0},
            "stay": {"s1": 1.0},
        },
        "s2": {
            "right": {"s2": 1.0},
            "stay": {"s2": 1.0},
        },
    }

    # Rewards
    # Reaching s2 gives 10. Staying in s2 gives 0.
    # Step cost -1.
    rewards = {
        "s0": {"right": -1.0, "stay": -1.0},
        "s1": {"right": 10.0, "stay": -1.0},  # Transition to s2 gets 10
        "s2": {"right": 0.0, "stay": 0.0},
    }

    result = await solve_discrete_mdp(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards,
        discount_factor=0.9,
    )

    assert result.status == "optimal"

    # Check policy
    # s1 -> right (get 10)
    assert result.policy["s1"] == "right"
    # s0 -> right (get -1 + 0.9 * 10 = 8) vs stay (-1 + 0.9*V(s0))
    assert result.policy["s0"] == "right"

    # Check values
    # V(s2) = 0
    # V(s1) = 10 + 0.9 * 0 = 10
    # V(s0) = -1 + 0.9 * 10 = 8
    assert abs(result.values["s2"] - 0.0) < 0.1
    assert abs(result.values["s1"] - 10.0) < 0.1
    assert abs(result.values["s0"] - 8.0) < 0.1
