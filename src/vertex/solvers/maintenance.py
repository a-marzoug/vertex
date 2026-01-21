"""Solver for Maintenance Planning problems (MDP)."""

import time

from vertex.config import SolverStatus
from vertex.models.maintenance import MaintenancePlanResult, OptimalPolicy


def solve_maintenance_mdp(
    num_states: int,
    transition_matrix: list[list[float]],
    cost_operating: list[float],
    cost_replacement: float,
    cost_failure: float,
    horizon: int = 10,
    discount_factor: float = 0.95,
) -> MaintenancePlanResult:
    """
    Solve maintenance planning using Value Iteration (Finite Horizon).

    State 0 = New.
    State N-1 = Failed/Worst condition.

    Actions:
    - Keep: Incur operating cost, transition according to matrix.
    - Replace: Incur replacement cost, transition to State 0.

    Args:
        num_states: Number of condition states.
        transition_matrix: P[i][j] probability of moving from i to j if Kept.
        cost_operating: Operating cost per period for each state.
        cost_replacement: Cost to replace equipment.
        cost_failure: Penalty cost if in failure state (usually last state).
        horizon: Planning horizon (periods).
        discount_factor: Discount rate for future costs.

    Returns:
        Optimal policy and total expected cost.
    """
    start_time = time.time()

    # V[t][s] = min expected cost from time t onwards starting in state s
    # Initialize terminal costs (0 or salvage, assumed 0)
    V = [[0.0] * num_states for _ in range(horizon + 1)]
    policy = []

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for s in range(num_states):
            # Option 1: Keep
            # Expected future cost
            future_cost_keep = sum(
                transition_matrix[s][next_s] * V[t + 1][next_s]
                for next_s in range(num_states)
            )

            # Immediate cost
            immediate_cost_keep = cost_operating[s]
            if s == num_states - 1:  # Failure state
                immediate_cost_keep += cost_failure

            value_keep = immediate_cost_keep + discount_factor * future_cost_keep

            # Option 2: Replace
            # Immediate cost is replacement. Next state is 0.
            # We assume replacement happens instantly or takes 1 period?
            # Standard assumption: Replace takes place, start next period in state 0 (or new unit operates this period?)
            # Let's assume Replace -> Pay Replacement Cost -> New unit (State 0) operates this period (so pay Op cost of state 0 + future from state 0).
            # OR Replace -> Pay Replacement, Next period is State 0.
            # Let's assume: Replace -> Pay R, Transition to 0 immediately (effectively) or next period.
            # Simplest: Replace -> Pay R + C(0) + discount * V[t+1][trans from 0].
            # Actually, "Replace" usually means you effectively restart the process.
            # Let's model: Replace action transitions to State 0 with prob 1.

            future_cost_replace = V[t + 1][0]  # Next period we are in state 0
            # If we assume replacement is instantaneous, we might also pay operating cost of new machine?
            # Let's keep it simple: Cost = Replacement Cost. Next State = 0.
            value_replace = cost_replacement + discount_factor * future_cost_replace

            if value_keep <= value_replace:
                V[t][s] = value_keep
                action = "Keep"
            else:
                V[t][s] = value_replace
                action = "Replace"

            policy.append(
                OptimalPolicy(
                    state=s,
                    time_step=t,
                    action=action,
                    expected_cost=round(V[t][s], 2),
                )
            )

    elapsed = (time.time() - start_time) * 1000

    # Sort policy by time then state
    policy.sort(key=lambda p: (p.time_step, p.state))

    return MaintenancePlanResult(
        status=SolverStatus.OPTIMAL,
        total_expected_cost=round(V[0][0], 2),  # Expected cost starting new
        policy=policy,
        solve_time_ms=elapsed,
    )
