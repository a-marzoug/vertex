"""Configuration constants for Vertex OR Server."""

from enum import StrEnum


class SolverType(StrEnum):
    """Supported solver backends."""

    GLOP = "GLOP"
    SCIP = "SCIP"
    SAT = "SAT"


class VariableType(StrEnum):
    """Variable domain types."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


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

    LEQ = "<="
    GEQ = ">="
    EQ = "="


DEFAULT_VAR_LOWER_BOUND = 0.0
DEFAULT_VAR_UPPER_BOUND = float("inf")

SERVER_NAME = "Vertex"
SERVER_DESCRIPTION = "Operations Research tools for decision makers"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
