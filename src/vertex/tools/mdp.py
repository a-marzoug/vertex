"""Markov Decision Process tools."""

from vertex.metrics import track_solve_metrics
from vertex.models.mdp import MDPResult
from vertex.solvers.mdp import solve_mdp
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_positive


@track_solve_metrics(tool_name="solve_discrete_mdp")
@validate_positive("max_iterations")
async def solve_discrete_mdp(
    states: list[str],
    actions: list[str],
    transitions: dict[str, dict[str, dict[str, float]]],
    rewards: dict[str, dict[str, float]],
    discount_factor: float = 0.9,
    epsilon: float = 1e-4,
    max_iterations: int = 1000,
) -> MDPResult:
    """
    Solve a discrete Markov Decision Process (MDP) using Value Iteration.

    Args:
        states: List of state names (e.g., ["s1", "s2"]).
        actions: List of action names (e.g., ["a1", "a2"]).
        transitions: Transition probabilities.
            Format: {state: {action: {next_state: prob, ...}, ...}, ...}
        rewards: Immediate rewards.
            Format: {state: {action: reward, ...}, ...}
        discount_factor: Discount factor gamma (default: 0.9).
        epsilon: Convergence threshold (default: 1e-4).
        max_iterations: Maximum iterations to run (default: 1000).

    Returns:
        Optimal policy and value function.
    """
    return await run_in_executor(
        solve_mdp,
        states,
        actions,
        transitions,
        rewards,
        discount_factor,
        epsilon,
        max_iterations,
    )
