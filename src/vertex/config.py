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


class RoutingStrategy(StrEnum):
    """First solution strategies for routing."""

    PATH_CHEAPEST_ARC = "PATH_CHEAPEST_ARC"
    SAVINGS = "SAVINGS"
    SWEEP = "SWEEP"
    CHRISTOFIDES = "CHRISTOFIDES"
    ALL_UNPERFORMED = "ALL_UNPERFORMED"
    BEST_INSERTION = "BEST_INSERTION"
    PARALLEL_CHEAPEST_INSERTION = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_CHEAPEST_INSERTION = "LOCAL_CHEAPEST_INSERTION"
    GLOBAL_CHEAPEST_ARC = "GLOBAL_CHEAPEST_ARC"
    LOCAL_CHEAPEST_ARC = "LOCAL_CHEAPEST_ARC"
    FIRST_UNBOUND_MIN_VALUE = "FIRST_UNBOUND_MIN_VALUE"
    AUTOMATIC = "AUTOMATIC"


class RoutingMetaheuristic(StrEnum):
    """Local search metaheuristics for routing."""

    GREEDY_DESCENT = "GREEDY_DESCENT"
    GUIDED_LOCAL_SEARCH = "GUIDED_LOCAL_SEARCH"
    SIMULATED_ANNEALING = "SIMULATED_ANNEALING"
    TABU_SEARCH = "TABU_SEARCH"
    GENERIC_TABU_SEARCH = "GENERIC_TABU_SEARCH"
    AUTOMATIC = "AUTOMATIC"


DEFAULT_VAR_LOWER_BOUND = 0.0
DEFAULT_VAR_UPPER_BOUND = float("inf")

SERVER_NAME = "Vertex"
SERVER_DESCRIPTION = "Operations Research tools for decision makers"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Resource limits for production safety
MAX_VARIABLES = 100_000
MAX_CONSTRAINTS = 100_000
MAX_GRAPH_NODES = 50_000
MAX_GRAPH_EDGES = 200_000
MAX_JOBS = 10_000
MAX_MACHINES = 1_000
DEFAULT_TIMEOUT_MS = 300_000  # 5 minutes
MAX_TIMEOUT_MS = 3_600_000  # 1 hour
