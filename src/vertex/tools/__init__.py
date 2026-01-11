"""MCP tools for optimization."""

from vertex.tools.analysis import (
    InfeasibilityResult,
    ModelStats,
    WhatIfResult,
    analyze_what_if,
    diagnose_infeasibility,
    get_model_stats,
    solve_rcpsp,
)
from vertex.tools.cp import NQueensResult, SudokuResult, solve_n_queens, solve_sudoku
from vertex.tools.linear import solve_lp
from vertex.tools.mip import solve_mip
from vertex.tools.multiobjective import MultiObjectiveResult, ParetoPoint, solve_multi_objective
from vertex.tools.network import (
    compute_max_flow,
    compute_min_cost_flow,
    compute_mst,
    compute_multi_commodity_flow,
    compute_shortest_path,
    compute_transshipment,
)
from vertex.tools.scheduling import (
    compute_bin_packing,
    compute_cutting_stock,
    compute_graph_coloring,
    compute_job_shop,
    compute_set_cover,
    compute_tsp,
    compute_vrp,
    compute_vrp_tw,
)
from vertex.tools.sensitivity import SensitivityReport, analyze_sensitivity

__all__ = [
    "analyze_sensitivity",
    "analyze_what_if",
    "compute_bin_packing",
    "compute_cutting_stock",
    "compute_graph_coloring",
    "compute_job_shop",
    "compute_max_flow",
    "compute_min_cost_flow",
    "compute_mst",
    "compute_multi_commodity_flow",
    "compute_set_cover",
    "compute_shortest_path",
    "compute_transshipment",
    "compute_tsp",
    "compute_vrp",
    "compute_vrp_tw",
    "diagnose_infeasibility",
    "get_model_stats",
    "InfeasibilityResult",
    "ModelStats",
    "MultiObjectiveResult",
    "NQueensResult",
    "ParetoPoint",
    "SensitivityReport",
    "solve_lp",
    "solve_mip",
    "solve_multi_objective",
    "solve_n_queens",
    "solve_rcpsp",
    "solve_sudoku",
    "SudokuResult",
    "WhatIfResult",
]
