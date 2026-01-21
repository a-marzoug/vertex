"""Maintenance planning tools."""

from vertex.metrics import track_solve_metrics
from vertex.models.maintenance import MaintenancePlanResult
from vertex.solvers.maintenance import solve_maintenance_mdp
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_non_negative, validate_positive


@track_solve_metrics(tool_name="optimize_equipment_replacement")
@validate_positive("num_states", "horizon")
@validate_non_negative("cost_replacement", "cost_failure")
async def optimize_equipment_replacement(
    num_states: int,
    transition_matrix: list[list[float]],
    cost_operating: list[float],
    cost_replacement: float,
    cost_failure: float,
    horizon: int = 10,
    discount_factor: float = 0.95,
) -> MaintenancePlanResult:
    """
    Optimize equipment replacement policy based on condition states.

    Uses Markov Decision Process (MDP) to find the optimal action (Keep vs Replace)
    for each condition state and time period to minimize total expected cost.

    Args:
        num_states: Number of condition states (0=New, N-1=Failed).
        transition_matrix: Probability matrix P[i][j] of moving from state i to j
            in one period if 'Keep' action is chosen.
        cost_operating: Operating cost per period for each state.
        cost_replacement: Cost to replace equipment (resets state to 0).
        cost_failure: Additional penalty cost if equipment is in failure state.
        horizon: Planning horizon in periods (default: 10).
        discount_factor: Discount rate for future costs (default: 0.95).

    Returns:
        Optimal policy (action for each state/time) and expected total cost.
    """
    return await run_in_executor(
        solve_maintenance_mdp,
        num_states,
        transition_matrix,
        cost_operating,
        cost_replacement,
        cost_failure,
        horizon,
        discount_factor,
    )
