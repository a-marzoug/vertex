"""Routing MCP tools."""

from vertex.config import RoutingMetaheuristic, RoutingStrategy
from vertex.metrics import track_solve_metrics
from vertex.models.routing import TSPResult, VRPResult
from vertex.solvers.routing import (
    solve_multi_depot_vrp,
    solve_pickup_delivery,
    solve_tsp,
    solve_vrp,
    solve_vrp_time_windows,
)
from vertex.utils.async_utils import run_in_executor
from vertex.validation import validate_problem_size, validate_timeout


@track_solve_metrics(tool_name="solve_tsp")
@validate_problem_size(max_variables=1000)  # Graph nodes
@validate_timeout()
async def compute_tsp(
    locations: list[str],
    distance_matrix: list[list[float]],
    time_limit_seconds: int = 30,
    search_strategy: str = RoutingStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: str = RoutingMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> TSPResult:
    """
    Solve Traveling Salesman Problem - find shortest tour visiting all locations.

    Args:
        locations: Location names. First is start/end point.
        distance_matrix: distances[i][j] = distance from location i to j.
        time_limit_seconds: Solver time limit.
        search_strategy: First solution strategy (PATH_CHEAPEST_ARC, SAVINGS, etc.).
        local_search_metaheuristic: Metaheuristic (GUIDED_LOCAL_SEARCH, TABU_SEARCH, etc.).

    Returns:
        Optimal tour and total distance.
    """
    return await run_in_executor(
        solve_tsp,
        locations,
        distance_matrix,
        time_limit_seconds,
        search_strategy,
        local_search_metaheuristic,
    )


@track_solve_metrics(tool_name="solve_vrp")
@validate_problem_size(max_variables=1000)
@validate_timeout()
async def compute_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
    search_strategy: str = RoutingStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: str = RoutingMetaheuristic.GUIDED_LOCAL_SEARCH,
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
        search_strategy: First solution strategy.
        local_search_metaheuristic: Local search metaheuristic.

    Returns:
        Routes for each vehicle with stops and distances.
    """
    return await run_in_executor(
        solve_vrp,
        locations,
        distance_matrix,
        demands,
        vehicle_capacities,
        depot,
        time_limit_seconds,
        search_strategy,
        local_search_metaheuristic,
    )


@track_solve_metrics(tool_name="solve_vrp_time_windows")
@validate_problem_size(max_variables=1000)
@validate_timeout()
async def compute_vrp_tw(
    locations: list[str],
    time_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
    search_strategy: str = RoutingStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: str = RoutingMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> VRPResult:
    """
    Solve VRP with Time Windows - vehicles must arrive within time windows.

    Args:
        locations: Location names.
        time_matrix: time_matrix[i][j] = travel time from i to j.
        time_windows: (earliest, latest) arrival time for each location.
        demands: Demand at each location.
        vehicle_capacities: Capacity of each vehicle.
        depot: Index of depot location.
        time_limit_seconds: Solver time limit.
        search_strategy: First solution strategy.
        local_search_metaheuristic: Local search metaheuristic.

    Returns:
        Routes with arrival times at each stop.
    """
    return await run_in_executor(
        solve_vrp_time_windows,
        locations,
        time_matrix,
        time_windows,
        demands,
        vehicle_capacities,
        depot,
        time_limit_seconds,
        search_strategy,
        local_search_metaheuristic,
    )


@track_solve_metrics(tool_name="solve_pickup_delivery")
@validate_problem_size(max_variables=1000)
@validate_timeout()
async def compute_pickup_delivery(
    locations: list[str],
    distance_matrix: list[list[float]],
    pickups_deliveries: list[list[int]],
    num_vehicles: int,
    depot: int = 0,
    time_limit_seconds: int = 30,
    search_strategy: str = RoutingStrategy.PARALLEL_CHEAPEST_INSERTION,
    local_search_metaheuristic: str = RoutingMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> VRPResult:
    """
    Solve VRP with Pickup and Delivery constraints.

    Args:
        locations: Location names.
        distance_matrix: distances[i][j] = distance from location i to j.
        pickups_deliveries: List of [pickup_index, delivery_index] pairs.
            Indices refer to the locations list.
        num_vehicles: Number of vehicles available.
        depot: Index of depot location.
        time_limit_seconds: Solver time limit.
        search_strategy: First solution strategy.
        local_search_metaheuristic: Local search metaheuristic.

    Returns:
        Routes respecting P&D constraints.
    """
    return await run_in_executor(
        solve_pickup_delivery,
        locations,
        distance_matrix,
        pickups_deliveries,
        num_vehicles,
        depot,
        time_limit_seconds,
        search_strategy,
        local_search_metaheuristic,
    )


@track_solve_metrics(tool_name="solve_multi_depot_vrp")
@validate_problem_size(max_variables=1000)
@validate_timeout()
async def compute_multi_depot_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    vehicle_starts: list[int],
    vehicle_ends: list[int],
    time_limit_seconds: int = 30,
    search_strategy: str = RoutingStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: str = RoutingMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> VRPResult:
    """
    Solve VRP with Multiple Depots.

    Args:
        locations: Location names.
        distance_matrix: distances[i][j] = distance from location i to j.
        demands: Demand at each location.
        vehicle_capacities: Capacity of each vehicle.
        vehicle_starts: Start location index for each vehicle.
        vehicle_ends: End location index for each vehicle.
        time_limit_seconds: Solver time limit.
        search_strategy: First solution strategy.
        local_search_metaheuristic: Local search metaheuristic.

    Returns:
        Routes starting and ending at specified depots.
    """
    return await run_in_executor(
        solve_multi_depot_vrp,
        locations,
        distance_matrix,
        demands,
        vehicle_capacities,
        vehicle_starts,
        vehicle_ends,
        time_limit_seconds,
        search_strategy,
        local_search_metaheuristic,
    )
