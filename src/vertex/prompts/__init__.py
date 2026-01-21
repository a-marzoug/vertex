"""MCP prompts for OR problem formulation."""

from vertex.prompts.linear import formulate_lp, interpret_solution
from vertex.prompts.mip import formulate_mip
from vertex.prompts.network import formulate_network_problem
from vertex.prompts.scheduling import formulate_scheduling_problem
from vertex.prompts.selection import select_optimization_approach
from vertex.prompts.sensitivity import interpret_sensitivity_analysis

__all__ = [
    "formulate_lp",
    "formulate_mip",
    "formulate_network_problem",
    "formulate_scheduling_problem",
    "interpret_sensitivity_analysis",
    "interpret_solution",
    "select_optimization_approach",
]
