"""Solver for Simulation Optimization."""

import time
from typing import Any

import scipy.optimize

from vertex.config import SolverStatus
from vertex.models.simulation import SimulationOptimizationResult, SimulationParameter
from vertex.solvers.stochastic import (
    run_monte_carlo_newsvendor,
    run_monte_carlo_production,
)


def _simulate_newsvendor_wrapper(
    order_quantity: float,
    demand_mean: float,
    demand_std: float,
    unit_cost: float,
    unit_price: float,
    holding_cost: float = 0.0,
    num_simulations: int = 10000,
):
    """Wrapper to adapt newsvendor parameters for simulation optimization."""
    result = run_monte_carlo_newsvendor(
        selling_price=unit_price,
        cost=unit_cost,
        salvage_value=0.0,  # Simplified: no salvage value
        order_quantity=order_quantity,
        mean_demand=demand_mean,
        std_demand=demand_std,
        num_simulations=num_simulations,
    )
    # Create a simple object with expected_profit attribute
    class Result:
        def __init__(self, mean_obj):
            self.expected_profit = mean_obj
            self.mean_objective = mean_obj

    return Result(result.mean_objective)


def _simulate_production_wrapper(
    production_quantities: dict[str, float],
    products: list[str],
    mean_demands: dict[str, float],
    std_demands: dict[str, float],
    selling_prices: dict[str, float],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    num_simulations: int = 10000,
):
    """Wrapper to adapt production parameters for simulation optimization."""
    result = run_monte_carlo_production(
        products=products,
        production_quantities=production_quantities,
        mean_demands=mean_demands,
        std_demands=std_demands,
        selling_prices=selling_prices,
        production_costs=production_costs,
        shortage_costs=shortage_costs,
        num_simulations=num_simulations,
    )
    # Create a simple object with expected_profit attribute
    class Result:
        def __init__(self, mean_obj):
            self.expected_profit = mean_obj
            self.mean_objective = mean_obj

    return Result(result.mean_objective)


# Registry of supported simulations
SIMULATIONS = {
    "simulate_production": _simulate_production_wrapper,
    "simulate_newsvendor": _simulate_newsvendor_wrapper,
}


def optimize_simulation(
    simulation_name: str,
    parameters: list[SimulationParameter],
    fixed_arguments: dict[str, Any],
    objective_attribute: str,
    objective_sense: str = "minimize",
    n_simulations_per_eval: int = 100,
    max_evaluations: int = 50,
    time_limit_seconds: int = 60,
) -> SimulationOptimizationResult:
    """
    Optimize parameters of a simulation model.

    Args:
        simulation_name: Name of the simulation function (e.g., "simulate_newsvendor").
        parameters: List of parameters to tune (ranges, types).
        fixed_arguments: Other arguments to pass to the simulation.
        objective_attribute: Name of the attribute in result to optimize (e.g., "mean_objective").
        objective_sense: "minimize" or "maximize".
        n_simulations_per_eval: Number of Monte Carlo runs per evaluation (higher = less noise).
        max_evaluations: Budget for optimizer.
        time_limit_seconds: Time budget.

    Returns:
        Optimization result.
    """
    start_time = time.time()

    if simulation_name not in SIMULATIONS:
        return SimulationOptimizationResult(
            status=SolverStatus.ERROR,
            optimal_parameters={},
            optimal_objective=0.0,
            num_evaluations=0,
            solve_time_ms=0.0,
            history=[],
        )

    sim_func = SIMULATIONS[simulation_name]
    history = []

    # Setup bounds
    bounds = [(p.lower_bound, p.upper_bound) for p in parameters]
    param_names = [p.name for p in parameters]

    sign = 1.0 if objective_sense == "minimize" else -1.0

    def objective(x):
        # Check time limit
        if time.time() - start_time > time_limit_seconds:
            return 1e9 * sign  # Soft penalty to stop? DE doesn't stop easily.

        current_args = fixed_arguments.copy()

        current_params = {}
        for i, param in enumerate(parameters):
            val = x[i]
            if param.is_integer:
                val = int(round(val))
            current_args[param.name] = val
            current_params[param.name] = val

        # Ensure num_simulations is passed
        current_args["num_simulations"] = n_simulations_per_eval

        try:
            result = sim_func(**current_args)
            obj_val = getattr(result, objective_attribute)

            # If objective is None (e.g. infeasible), penalize
            if obj_val is None:
                return 1e9 * sign

            history.append({"parameters": current_params, "objective": obj_val})

            return sign * obj_val
        except Exception:
            return 1e9 * sign

    # Use Differential Evolution as it's robust to noise (stochasticity)
    res = scipy.optimize.differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_evaluations
        // 10,  # popsize default is 15, so maxiter * 15 * len(x) evals
        popsize=5,
        seed=42,
        polish=False,  # Polishing uses gradient, bad for stochastic
    )

    elapsed = (time.time() - start_time) * 1000

    # Extract optimal
    optimal_params = {}
    for i, param in enumerate(parameters):
        val = res.x[i]
        if param.is_integer:
            val = int(round(val))
        optimal_params[param.name] = val

    return SimulationOptimizationResult(
        status=SolverStatus.OPTIMAL,
        optimal_parameters=optimal_params,
        optimal_objective=sign * res.fun,
        num_evaluations=res.nfev,
        solve_time_ms=elapsed,
        history=history[-50:],  # Keep last 50
    )
