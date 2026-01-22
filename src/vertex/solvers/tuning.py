"""Solver for self-tuning and automatic solver selection."""

from vertex.models.tuning import ProblemCharacteristics, SolverRecommendation


def recommend_solver(characteristics: ProblemCharacteristics) -> SolverRecommendation:
    """
    Recommend the best solver based on problem characteristics.

    Args:
        characteristics: Problem characteristics.

    Returns:
        Solver recommendation with reasoning.
    """
    num_vars = characteristics.num_variables
    num_cons = characteristics.num_constraints
    has_int = characteristics.has_integer_variables
    has_bin = characteristics.has_binary_variables
    is_nl = characteristics.is_nonlinear
    is_qp = characteristics.is_quadratic
    is_network = characteristics.is_network_flow
    is_sched = characteristics.is_scheduling
    is_route = characteristics.is_routing
    has_unc = characteristics.has_uncertainty

    # Network flow problems
    if is_network:
        if is_route:
            return SolverRecommendation(
                recommended_tool="compute_vrp" if has_int else "compute_tsp",
                alternative_tools=["compute_multi_depot_vrp", "compute_pickup_delivery"],
                reasoning="Network routing problem detected. VRP/TSP solvers use specialized algorithms (OR-Tools routing).",
                solver_hints={"use_guided_local_search": "true for large instances"},
                expected_performance="medium",
            )
        return SolverRecommendation(
            recommended_tool="compute_max_flow",
            alternative_tools=["compute_min_cost_flow", "compute_multi_commodity_flow"],
            reasoning="Network flow problem. Specialized network simplex algorithms are fastest.",
            solver_hints={"algorithm": "network_simplex"},
            expected_performance="fast",
        )

    # Scheduling problems
    if is_sched:
        if num_vars > 500:
            return SolverRecommendation(
                recommended_tool="solve_job_shop",
                alternative_tools=["solve_flexible_job_shop", "solve_rcpsp"],
                reasoning="Large scheduling problem. CP-SAT excels at constraint-heavy scheduling.",
                solver_hints={"use_cp_sat": "true", "time_limit": "increase for large problems"},
                expected_performance="medium",
            )
        return SolverRecommendation(
            recommended_tool="solve_job_shop",
            alternative_tools=["solve_flow_shop", "solve_parallel_machines"],
            reasoning="Scheduling problem. CP-SAT handles precedence and resource constraints well.",
            solver_hints={"use_cp_sat": "true"},
            expected_performance="fast",
        )

    # Stochastic/uncertain problems
    if has_unc:
        return SolverRecommendation(
            recommended_tool="solve_two_stage_stochastic",
            alternative_tools=["solve_robust_production", "solve_chance_constrained"],
            reasoning="Problem involves uncertainty. Two-stage stochastic programming handles scenarios.",
            solver_hints={"num_scenarios": "balance accuracy vs speed"},
            expected_performance="medium",
        )

    # Nonlinear problems
    if is_nl:
        if has_int or has_bin:
            return SolverRecommendation(
                recommended_tool="solve_minlp",
                alternative_tools=["solve_nonlinear_program"],
                reasoning="Mixed-integer nonlinear problem. Uses Differential Evolution for global search.",
                solver_hints={
                    "maxiter": "increase for complex landscapes",
                    "popsize": "increase for better exploration",
                },
                expected_performance="slow",
            )
        return SolverRecommendation(
            recommended_tool="solve_nonlinear_program",
            alternative_tools=["solve_qp" if is_qp else "solve_minlp"],
            reasoning="Nonlinear continuous problem. SLSQP handles smooth nonlinear constraints.",
            solver_hints={"initial_guess": "provide good starting point"},
            expected_performance="medium",
        )

    # Quadratic problems
    if is_qp:
        return SolverRecommendation(
            recommended_tool="solve_qp",
            alternative_tools=["optimize_portfolio_variance"],
            reasoning="Quadratic problem. OSQP solver is specialized for convex QP.",
            solver_hints={"ensure_convex": "true"},
            expected_performance="fast",
        )

    # Mixed-integer linear problems
    if has_int or has_bin:
        if num_vars > 10000:
            return SolverRecommendation(
                recommended_tool="solve_mixed_integer_program",
                alternative_tools=["solve_linear_program"],
                reasoning="Large MIP. SCIP uses branch-and-cut with advanced preprocessing.",
                solver_hints={
                    "use_scip": "true",
                    "presolve": "aggressive",
                    "time_limit": "set reasonable limit for large problems",
                },
                expected_performance="slow",
            )
        if num_vars > 1000:
            return SolverRecommendation(
                recommended_tool="solve_mixed_integer_program",
                alternative_tools=["solve_linear_program"],
                reasoning="Medium MIP. SCIP handles efficiently with default settings.",
                solver_hints={"use_scip": "true"},
                expected_performance="medium",
            )
        return SolverRecommendation(
            recommended_tool="solve_mixed_integer_program",
            alternative_tools=["solve_linear_program"],
            reasoning="Small MIP. SCIP solves quickly with branch-and-bound.",
            solver_hints={"use_scip": "true"},
            expected_performance="fast",
        )

    # Linear programming
    if num_vars > 50000:
        return SolverRecommendation(
            recommended_tool="solve_linear_program",
            alternative_tools=[],
            reasoning="Very large LP. GLOP (Google's LP solver) uses dual simplex for large sparse problems.",
            solver_hints={
                "use_glop": "true",
                "scaling": "equilibration recommended",
            },
            expected_performance="medium",
        )

    return SolverRecommendation(
        recommended_tool="solve_linear_program",
        alternative_tools=["solve_mixed_integer_program"],
        reasoning="Linear programming problem. GLOP uses primal/dual simplex efficiently.",
        solver_hints={"use_glop": "true"},
        expected_performance="fast",
    )
