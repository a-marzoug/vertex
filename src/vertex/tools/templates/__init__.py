"""Domain-specific optimization templates."""

from vertex.tools.templates.assignment import AssignmentResult, optimize_assignment
from vertex.tools.templates.diet import DietResult, optimize_diet
from vertex.tools.templates.facility import FacilityResult, optimize_facility_location
from vertex.tools.templates.knapsack import KnapsackResult, optimize_knapsack
from vertex.tools.templates.portfolio import PortfolioResult, optimize_portfolio
from vertex.tools.templates.production import ProductionResult, optimize_production

__all__ = [
    "AssignmentResult",
    "DietResult",
    "FacilityResult",
    "KnapsackResult",
    "PortfolioResult",
    "ProductionResult",
    "optimize_assignment",
    "optimize_diet",
    "optimize_facility_location",
    "optimize_knapsack",
    "optimize_portfolio",
    "optimize_production",
]
