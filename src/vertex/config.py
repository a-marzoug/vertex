"""Configuration constants for Vertex OR Server."""

from enum import StrEnum


class SolverType(StrEnum):
    """Supported solver backends."""

    GLOP = "GLOP"  # Google's LP solver
    PDLP = "PDLP"  # Primal-Dual LP solver


class SolverStatus(StrEnum):
    """Solver result status codes."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


class ObjectiveSense(StrEnum):
    """Optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintSense(StrEnum):
    """Constraint comparison operators."""

    LEQ = "<="  # Less than or equal
    GEQ = ">="  # Greater than or equal
    EQ = "="  # Equal


# Default bounds
DEFAULT_VAR_LOWER_BOUND = 0.0
DEFAULT_VAR_UPPER_BOUND = float("inf")

# Server configuration
SERVER_NAME = "Vertex"
SERVER_DESCRIPTION = "Operations Research tools for decision makers"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
