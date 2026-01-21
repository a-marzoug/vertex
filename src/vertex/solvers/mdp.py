"""Solver for Markov Decision Processes (MDP)."""

import time
from collections.abc import Mapping
from typing import Any

from vertex.config import SolverStatus
from vertex.models.mdp import MDPResult


def solve_mdp(
    states: list[str],
    actions: list[str],
    transitions: dict[str, dict[str, dict[str, float]]],
    rewards: dict[str, dict[str, float]],
    discount_factor: float = 0.9,
    epsilon: float = 1e-4,
    max_iterations: int = 1000,
) -> MDPResult:
    """
    Solve infinite horizon MDP using Value Iteration.

    Args:
        states: List of state names.
        actions: List of action names.
        transitions: Map state -> action -> next_state -> probability.
        rewards: Map state -> action -> reward.
        discount_factor: Gamma (0 < gamma < 1).
        epsilon: Convergence threshold.
        max_iterations: Maximum iterations.

    Returns:
        Optimal policy and values.
    """
    start_time = time.time()

    # Initialize Value function
    V = {s: 0.0 for s in states}

    # Value Iteration
    for _ in range(max_iterations):
        delta = 0.0
        new_V = V.copy()

        for s in states:
            # Find max over actions
            max_val = float("-inf")

            # If no actions available from state (terminal), value is 0 (or reward)
            available_actions = transitions.get(s, {})
            if not available_actions:
                new_V[s] = 0.0
                continue

            for a in available_actions:
                # Calculate Q(s, a)
                # Reward: R(s, a)
                r = rewards.get(s, {}).get(a, 0.0)

                # Expected future value
                future_val = sum(
                    prob * V.get(next_s, 0.0)
                    for next_s, prob in available_actions[a].items()
                )

                q_val = r + discount_factor * future_val
                if q_val > max_val:
                    max_val = q_val

            new_V[s] = max_val
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < epsilon:
            break

    # Extract Policy
    policy = {}
    for s in states:
        best_a = None
        max_val = float("-inf")

        available_actions = transitions.get(s, {})
        if not available_actions:
            policy[s] = "None"
            continue

        for a in available_actions:
            r = rewards.get(s, {}).get(a, 0.0)
            future_val = sum(
                prob * V.get(next_s, 0.0)
                for next_s, prob in available_actions[a].items()
            )
            q_val = r + discount_factor * future_val
            if q_val > max_val:
                max_val = q_val
                best_a = a

        policy[s] = best_a or "None"

    elapsed = (time.time() - start_time) * 1000

    return MDPResult(
        status=SolverStatus.OPTIMAL,
        optimal_value=V[states[0]] if states else 0.0,
        policy=policy,
        values={k: round(v, 4) for k, v in V.items()},
        solve_time_ms=elapsed,
    )
