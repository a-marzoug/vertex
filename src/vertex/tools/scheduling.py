"""Scheduling and Routing MCP tools."""

from vertex.models.scheduling import JobShopResult, TSPResult, VRPResult
from vertex.solvers.scheduling import solve_job_shop, solve_tsp, solve_vrp


def compute_tsp(
    locations: list[str],
    distance_matrix: list[list[float]],
    time_limit_seconds: int = 30,
) -> TSPResult:
    """
    Solve Traveling Salesman Problem - find shortest tour visiting all locations.

    Args:
        locations: Location names. First is start/end point.
        distance_matrix: distances[i][j] = distance from location i to j.
        time_limit_seconds: Solver time limit.

    Returns:
        Optimal tour and total distance.
    """
    return solve_tsp(locations, distance_matrix, time_limit_seconds)


def compute_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
) -> VRPResult:
    """
    Solve Capacitated Vehicle Routing Problem.

    Args:
        locations: Location names. Index 0 is typically the depot.
        distance_matrix: distances[i][j] = distance from location i to j.
        demands: Demand at each location (depot demand should be 0).
        vehicle_capacities: Capacity of each vehicle.
        depot: Index of depot location.
        time_limit_seconds: Solver time limit.

    Returns:
        Routes for each vehicle with stops and distances.
    """
    return solve_vrp(locations, distance_matrix, demands, vehicle_capacities, depot, time_limit_seconds)


def compute_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> JobShopResult:
    """
    Solve Job Shop Scheduling Problem - schedule jobs on machines minimizing makespan.

    Args:
        jobs: List of jobs. Each job is a list of tasks: {"machine": int, "duration": int}.
            Tasks within a job must be processed in order.
        time_limit_seconds: Solver time limit.

    Returns:
        Schedule with start times for each task and total makespan.

    Example:
        jobs = [
            [{"machine": 0, "duration": 3}, {"machine": 1, "duration": 2}],  # Job 0
            [{"machine": 1, "duration": 4}, {"machine": 0, "duration": 2}],  # Job 1
        ]
    """
    # Convert dict format to tuple format
    jobs_tuples = [[(t["machine"], t["duration"]) for t in job] for job in jobs]
    return solve_job_shop(jobs_tuples, time_limit_seconds)
