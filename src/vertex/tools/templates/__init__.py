"""Domain-specific optimization templates."""

from vertex.tools.templates.diet import optimize_diet
from vertex.tools.templates.portfolio import optimize_portfolio
from vertex.tools.templates.production import optimize_production

__all__ = ["optimize_diet", "optimize_portfolio", "optimize_production"]
