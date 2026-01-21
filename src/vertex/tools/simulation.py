"""Simulation Optimization tools."""

from typing import Any

from vertex.metrics import track_solve_metrics
from vertex.models.simulation import SimulationOptimizationResult, SimulationParameter
from vertex.solvers.simulation import optimize_simulation
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_positive, validate_timeout


@track_solve_metrics(tool_name="optimize_simulation_parameters")
@validate_positive("n_simulations_per_eval", "max_evaluations")
@validate_timeout()
async def optimize_simulation_parameters(
    simulation_name: str,
    parameters: list[dict[str, Any]],
    fixed_arguments: dict[str, Any],
    objective_attribute: str,
    objective_sense: str = "minimize",
    n_simulations_per_eval: int = 100,
    max_evaluations: int = 50,
    time_limit_ms: int = 60000,
) -> SimulationOptimizationResult:
    """
    Optimize parameters for a simulation model using black-box optimization.

    Args:
        simulation_name: Name of simulation ("simulate_newsvendor").
        parameters: List of params to tune.
            Example: [{"name": "order_quantity", "lower_bound": 0, "upper_bound": 100, "is_integer": False}]
        fixed_arguments: Static arguments for simulation.
        objective_attribute: Result field to optimize (e.g., "expected_profit").
        objective_sense: "minimize" or "maximize".
        n_simulations_per_eval: Monte Carlo runs per trial.
        max_evaluations: Max optimizer iterations.
        time_limit_ms: Time budget.

    Returns:
        Optimal parameters and objective.
    """
    # Parse parameters
    sim_params = []
    for p in parameters:
        sim_params.append(
            SimulationParameter(
                name=p["name"],
                lower_bound=p["lower_bound"],
                upper_bound=p["upper_bound"],
                initial_guess=p.get("initial_guess"),
                is_integer=p.get("is_integer", False),
            )
        )

    return await run_in_executor(
        optimize_simulation,
        simulation_name,
        sim_params,
        fixed_arguments,
        objective_attribute,
        objective_sense,
        n_simulations_per_eval,
        max_evaluations,
        time_limit_ms // 1000,
    )
