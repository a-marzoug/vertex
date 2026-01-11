"""Domain-specific optimization templates."""

from vertex.tools.templates.assignment import AssignmentResult, optimize_assignment
from vertex.tools.templates.diet import DietResult, optimize_diet
from vertex.tools.templates.facility import FacilityResult, optimize_facility_location
from vertex.tools.templates.inventory import (
    EOQResult,
    MultiItemInventoryResult,
    optimize_eoq,
    optimize_multi_item_inventory,
)
from vertex.tools.templates.knapsack import KnapsackResult, optimize_knapsack
from vertex.tools.templates.portfolio import PortfolioResult, optimize_portfolio
from vertex.tools.templates.production import ProductionResult, optimize_production
from vertex.tools.templates.workforce import (
    ShiftAssignment,
    WorkforceResult,
    optimize_workforce_schedule,
)

__all__ = [
    "AssignmentResult",
    "DietResult",
    "EOQResult",
    "FacilityResult",
    "KnapsackResult",
    "MultiItemInventoryResult",
    "PortfolioResult",
    "ProductionResult",
    "ShiftAssignment",
    "WorkforceResult",
    "optimize_assignment",
    "optimize_diet",
    "optimize_eoq",
    "optimize_facility_location",
    "optimize_knapsack",
    "optimize_multi_item_inventory",
    "optimize_portfolio",
    "optimize_production",
    "optimize_workforce_schedule",
]
