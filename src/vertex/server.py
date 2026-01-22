"""Vertex MCP Server - Operations Research tools for decision makers."""

import os

from mcp.server.fastmcp import FastMCP

from vertex.config import DEFAULT_HOST, DEFAULT_PORT, SERVER_DESCRIPTION, SERVER_NAME
from vertex.logging import configure_logging, get_logger
from vertex.metrics import get_metrics
from vertex.prompts.linear import formulate_lp, interpret_solution
from vertex.prompts.mip import formulate_mip
from vertex.prompts.network import formulate_network_problem
from vertex.prompts.scheduling import formulate_scheduling_problem
from vertex.prompts.selection import select_optimization_approach
from vertex.prompts.sensitivity import interpret_sensitivity_analysis
from vertex.tools.analysis import (
    analyze_what_if,
    diagnose_infeasibility,
    find_alternative_solutions,
    get_model_stats,
)
from vertex.tools.cp import solve_n_queens, solve_sudoku
from vertex.tools.linear import solve_lp
from vertex.tools.maintenance import optimize_equipment_replacement
from vertex.tools.mdp import solve_discrete_mdp
from vertex.tools.mip import solve_mip
from vertex.tools.multiobjective import solve_multi_objective
from vertex.tools.network import (
    compute_max_flow,
    compute_min_cost_flow,
    compute_mst,
    compute_multi_commodity_flow,
    compute_shortest_path,
    compute_transshipment,
)
from vertex.tools.nonlinear import solve_minlp, solve_nonlinear_program
from vertex.tools.routing import (
    compute_multi_depot_vrp,
    compute_pickup_delivery,
    compute_tsp,
    compute_vrp,
    compute_vrp_tw,
)
from vertex.tools.scheduling import (
    compute_bin_packing,
    compute_cutting_stock,
    compute_flexible_job_shop,
    compute_flow_shop,
    compute_graph_coloring,
    compute_job_shop,
    compute_parallel_machines,
    compute_set_cover,
    solve_rcpsp,
)
from vertex.tools.sensitivity import analyze_sensitivity
from vertex.tools.simulation import optimize_simulation_parameters
from vertex.tools.stochastic import (
    analyze_queue_mm1,
    analyze_queue_mmc,
    compute_lot_sizing,
    compute_newsvendor,
    compute_two_stage_stochastic,
    design_network,
    find_steiner_tree,
    optimize_multi_echelon_inventory,
    optimize_portfolio_qp,
    pack_rectangles_2d,
    schedule_crew,
    simulate_newsvendor_monte_carlo,
    simulate_production_monte_carlo,
    solve_chance_constrained_production,
    solve_quadratic_assignment,
    solve_quadratic_program,
    solve_robust_optimization,
)
from vertex.tools.tuning import select_solver
from vertex.tools.templates.assignment import optimize_assignment
from vertex.tools.templates.diet import optimize_diet
from vertex.tools.templates.facility import optimize_facility_location
from vertex.tools.templates.healthcare import optimize_resource_allocation
from vertex.tools.templates.inventory import optimize_eoq
from vertex.tools.templates.knapsack import optimize_knapsack
from vertex.tools.templates.portfolio import optimize_portfolio
from vertex.tools.templates.production import optimize_production
from vertex.tools.templates.supplychain import optimize_supply_chain
from vertex.tools.templates.workforce import optimize_workforce_schedule

mcp = FastMCP(
    SERVER_NAME,
    instructions=SERVER_DESCRIPTION,
    stateless_http=True,
    json_response=True,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
)


def get_system_metrics() -> str:
    """Get current system metrics (Prometheus format)."""
    data, _ = get_metrics()
    return data.decode("utf-8")


# Tools
mcp.add_tool(solve_lp, name="solve_linear_program")
mcp.add_tool(analyze_sensitivity, name="analyze_lp_sensitivity")
mcp.add_tool(analyze_what_if, name="analyze_what_if_scenario")
mcp.add_tool(diagnose_infeasibility)
mcp.add_tool(get_model_stats, name="get_model_statistics")
mcp.add_tool(solve_multi_objective, name="solve_pareto_frontier")
mcp.add_tool(solve_sudoku, name="solve_sudoku_puzzle")
mcp.add_tool(solve_n_queens, name="solve_n_queens_puzzle")
mcp.add_tool(find_alternative_solutions, name="find_alternative_optimal_solutions")
mcp.add_tool(optimize_production, name="optimize_production_plan")
mcp.add_tool(optimize_diet, name="optimize_diet_plan")
mcp.add_tool(optimize_portfolio, name="optimize_investment_portfolio")
mcp.add_tool(solve_mip, name="solve_mixed_integer_program")
mcp.add_tool(optimize_assignment, name="optimize_worker_assignment")
mcp.add_tool(optimize_knapsack, name="optimize_knapsack_selection")
mcp.add_tool(optimize_facility_location, name="optimize_facility_locations")
mcp.add_tool(optimize_eoq, name="optimize_inventory_eoq")
mcp.add_tool(optimize_workforce_schedule, name="optimize_workforce")
mcp.add_tool(optimize_resource_allocation, name="optimize_healthcare_resources")
mcp.add_tool(optimize_supply_chain, name="optimize_supply_chain_network")
mcp.add_tool(compute_max_flow, name="find_max_flow")
mcp.add_tool(compute_min_cost_flow, name="find_min_cost_flow")
mcp.add_tool(compute_shortest_path, name="find_shortest_path")
mcp.add_tool(compute_mst, name="find_minimum_spanning_tree")
mcp.add_tool(compute_multi_commodity_flow, name="find_multi_commodity_flow")
mcp.add_tool(compute_transshipment, name="solve_transshipment")
mcp.add_tool(compute_tsp, name="solve_tsp")
mcp.add_tool(compute_vrp, name="solve_vrp")
mcp.add_tool(compute_job_shop, name="solve_job_shop")
mcp.add_tool(solve_rcpsp)
mcp.add_tool(compute_flexible_job_shop, name="solve_flexible_job_shop")
mcp.add_tool(compute_vrp_tw, name="solve_vrp_time_windows")
mcp.add_tool(compute_pickup_delivery, name="solve_pickup_delivery")
mcp.add_tool(compute_multi_depot_vrp, name="solve_multi_depot_vrp")
mcp.add_tool(compute_bin_packing, name="solve_bin_packing")
mcp.add_tool(compute_set_cover, name="solve_set_cover")
mcp.add_tool(compute_graph_coloring, name="solve_graph_coloring")
mcp.add_tool(compute_cutting_stock, name="solve_cutting_stock")
mcp.add_tool(compute_two_stage_stochastic, name="solve_two_stage_stochastic")
mcp.add_tool(compute_newsvendor, name="solve_newsvendor")
mcp.add_tool(compute_lot_sizing, name="solve_lot_sizing")
mcp.add_tool(solve_robust_optimization, name="solve_robust_production")
mcp.add_tool(analyze_queue_mm1, name="analyze_mm1_queue")
mcp.add_tool(analyze_queue_mmc, name="analyze_mmc_queue")
mcp.add_tool(compute_flow_shop, name="solve_flow_shop")
mcp.add_tool(compute_parallel_machines, name="solve_parallel_machines")
mcp.add_tool(simulate_newsvendor_monte_carlo, name="simulate_newsvendor")
mcp.add_tool(simulate_production_monte_carlo, name="simulate_production")
mcp.add_tool(schedule_crew, name="solve_crew_schedule")
mcp.add_tool(solve_chance_constrained_production, name="solve_chance_constrained")
mcp.add_tool(pack_rectangles_2d, name="solve_2d_bin_packing")
mcp.add_tool(design_network, name="solve_network_design")
mcp.add_tool(solve_quadratic_assignment, name="solve_quadratic_assignment_problem")
mcp.add_tool(find_steiner_tree, name="solve_steiner_tree")
mcp.add_tool(optimize_multi_echelon_inventory, name="optimize_multi_echelon")
mcp.add_tool(solve_quadratic_program, name="solve_qp")
mcp.add_tool(optimize_portfolio_qp, name="optimize_portfolio_variance")
mcp.add_tool(optimize_equipment_replacement, name="optimize_equipment_replacement")
mcp.add_tool(solve_discrete_mdp, name="solve_discrete_mdp")
mcp.add_tool(solve_nonlinear_program, name="solve_nonlinear_program")
mcp.add_tool(solve_minlp, name="solve_minlp")
mcp.add_tool(optimize_simulation_parameters, name="optimize_simulation_parameters")
mcp.add_tool(select_solver, name="select_solver")
mcp.add_tool(get_system_metrics, name="get_system_metrics")

# Prompts
formulate_lp_problem = mcp.prompt(name="formulate_lp_problem")(formulate_lp)
formulate_mip_problem = mcp.prompt(name="formulate_mip_problem")(formulate_mip)
interpret_lp_solution = mcp.prompt(name="interpret_lp_solution")(interpret_solution)
select_approach = mcp.prompt(name="select_optimization_approach")(
    select_optimization_approach
)
formulate_network = mcp.prompt(name="formulate_network_problem")(
    formulate_network_problem
)
formulate_scheduling = mcp.prompt(name="formulate_scheduling_problem")(
    formulate_scheduling_problem
)
interpret_sensitivity = mcp.prompt(name="interpret_sensitivity_analysis")(
    interpret_sensitivity_analysis
)


def main() -> None:
    """Run the Vertex MCP server."""
    import sys
    from typing import Literal

    # Configure logging based on environment
    log_level = os.getenv("VERTEX_LOG_LEVEL", "INFO")
    json_logs = os.getenv("VERTEX_JSON_LOGS", "true").lower() == "true"
    configure_logging(level=log_level, json_format=json_logs)

    logger = get_logger(__name__)
    logger.info(
        "server_starting",
        server_name=SERVER_NAME,
        transport="stdio" if "--http" not in sys.argv else "streamable-http",
    )

    transport: Literal["stdio", "sse", "streamable-http"] = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
