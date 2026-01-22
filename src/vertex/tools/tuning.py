"""Self-tuning and solver selection tools."""

from typing import Any

from vertex.models.tuning import ProblemCharacteristics, SolverRecommendation
from vertex.solvers.tuning import recommend_solver


def select_solver(
    num_variables: int,
    num_constraints: int,
    has_integer_variables: bool = False,
    has_binary_variables: bool = False,
    is_nonlinear: bool = False,
    is_quadratic: bool = False,
    is_network_flow: bool = False,
    is_scheduling: bool = False,
    is_routing: bool = False,
    has_uncertainty: bool = False,
    sparsity: float | None = None,
) -> SolverRecommendation:
    """
    Automatically select the best solver based on problem characteristics.

    Analyzes problem structure and recommends the most appropriate tool and solver
    configuration for optimal performance.

    Args:
        num_variables: Number of decision variables.
        num_constraints: Number of constraints.
        has_integer_variables: Problem includes integer variables.
        has_binary_variables: Problem includes binary (0/1) variables.
        is_nonlinear: Objective or constraints are nonlinear.
        is_quadratic: Objective is quadratic (x^T Q x).
        is_network_flow: Problem is a network flow (graph-based).
        is_scheduling: Problem involves scheduling/sequencing.
        is_routing: Problem involves vehicle routing or TSP.
        has_uncertainty: Problem involves stochastic/uncertain parameters.
        sparsity: Fraction of zero coefficients (0-1), if known.

    Returns:
        Solver recommendation with tool name, reasoning, and configuration hints.

    Example:
        >>> select_solver(
        ...     num_variables=100,
        ...     num_constraints=50,
        ...     has_integer_variables=True,
        ...     is_scheduling=True
        ... )
        SolverRecommendation(
            recommended_tool="solve_job_shop",
            reasoning="Scheduling problem with integer variables...",
            expected_performance="fast"
        )
    """
    characteristics = ProblemCharacteristics(
        num_variables=num_variables,
        num_constraints=num_constraints,
        has_integer_variables=has_integer_variables,
        has_binary_variables=has_binary_variables,
        is_nonlinear=is_nonlinear,
        is_quadratic=is_quadratic,
        is_network_flow=is_network_flow,
        is_scheduling=is_scheduling,
        is_routing=is_routing,
        has_uncertainty=has_uncertainty,
        sparsity=sparsity,
    )

    return recommend_solver(characteristics)
