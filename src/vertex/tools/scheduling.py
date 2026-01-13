"""Scheduling and Routing MCP tools."""

from vertex.models.scheduling import (
    BinPackingResult,
    CuttingStockResult,
    FlowShopResult,
    GraphColoringResult,
    JobShopResult,
    ParallelMachineResult,
    SetCoverResult,
    TSPResult,
    VRPResult,
)
from vertex.solvers.scheduling import (
    solve_bin_packing,
    solve_cutting_stock,
    solve_flow_shop,
    solve_graph_coloring,
    solve_job_shop,
    solve_parallel_machines,
    solve_set_cover,
    solve_tsp,
    solve_vrp,
    solve_vrp_time_windows,
)


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
    return solve_vrp(
        locations,
        distance_matrix,
        demands,
        vehicle_capacities,
        depot,
        time_limit_seconds,
    )


def compute_vrp_tw(
    locations: list[str],
    time_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
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

    Returns:
        Routes with arrival times at each stop.
    """
    return solve_vrp_time_windows(
        locations,
        time_matrix,
        time_windows,
        demands,
        vehicle_capacities,
        depot,
        time_limit_seconds,
    )


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
    """
    jobs_tuples = [[(t["machine"], t["duration"]) for t in job] for job in jobs]
    return solve_job_shop(jobs_tuples, time_limit_seconds)


def compute_bin_packing(
    items: list[str],
    weights: dict[str, float],
    bin_capacity: float,
    max_bins: int | None = None,
    time_limit_seconds: int = 30,
) -> BinPackingResult:
    """
    Solve Bin Packing Problem - pack items into minimum number of bins.

    Args:
        items: Item names.
        weights: Weight of each item.
        bin_capacity: Capacity of each bin.
        max_bins: Maximum bins available (defaults to number of items).
        time_limit_seconds: Solver time limit.

    Returns:
        Bin assignments and number of bins used.
    """
    return solve_bin_packing(items, weights, bin_capacity, max_bins, time_limit_seconds)


def compute_set_cover(
    universe: list[str],
    sets: dict[str, list[str]],
    costs: dict[str, float],
    time_limit_seconds: int = 30,
) -> SetCoverResult:
    """
    Solve Set Covering Problem - select minimum cost sets to cover all elements.

    Args:
        universe: Elements that must be covered.
        sets: Available sets, each mapping to list of elements it covers.
        costs: Cost of each set.
        time_limit_seconds: Solver time limit.

    Returns:
        Selected sets and total cost.
    """
    return solve_set_cover(universe, sets, costs, time_limit_seconds)


def compute_graph_coloring(
    nodes: list[str],
    edges: list[tuple[str, str]],
    max_colors: int | None = None,
    time_limit_seconds: int = 30,
) -> GraphColoringResult:
    """
    Solve Graph Coloring - assign colors to nodes so adjacent nodes differ.

    Args:
        nodes: Node names.
        edges: List of (node1, node2) edges.
        max_colors: Maximum colors available.
        time_limit_seconds: Solver time limit.

    Returns:
        Color assignment minimizing number of colors used.
    """
    return solve_graph_coloring(nodes, edges, max_colors, time_limit_seconds)


def compute_cutting_stock(
    items: list[str],
    lengths: dict[str, int],
    demands: dict[str, int],
    stock_length: int,
    max_stock: int | None = None,
    time_limit_seconds: int = 30,
) -> CuttingStockResult:
    """
    Solve Cutting Stock - cut items from stock material minimizing waste.

    Args:
        items: Item names.
        lengths: Length of each item type.
        demands: Number of each item needed.
        stock_length: Length of each stock piece.
        max_stock: Maximum stock pieces available.
        time_limit_seconds: Solver time limit.

    Returns:
        Cutting patterns and total waste.
    """
    return solve_cutting_stock(
        items, lengths, demands, stock_length, max_stock, time_limit_seconds
    )


def compute_flexible_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> dict:
    """
    Solve Flexible Job Shop - tasks can run on alternative machines.

    Args:
        jobs: List of jobs. Each job is list of tasks.
            Each task: {"machines": [(machine_id, duration), ...]}
        time_limit_seconds: Solver time limit.

    Returns:
        Dict with makespan and machine assignments.
    """
    from vertex.solvers.scheduling import solve_flexible_job_shop

    return solve_flexible_job_shop(jobs, time_limit_seconds)


def compute_flow_shop(
    processing_times: list[list[int]],
    time_limit_seconds: int = 30,
) -> FlowShopResult:
    """
    Solve Flow Shop Scheduling - all jobs follow same machine sequence.

    Args:
        processing_times: processing_times[job][machine] = duration.
            All jobs visit machines 0, 1, 2, ... in order.
        time_limit_seconds: Solver time limit.

    Returns:
        Optimal job sequence and makespan.
    """
    return solve_flow_shop(processing_times, time_limit_seconds)


def compute_parallel_machines(
    job_durations: list[int],
    num_machines: int,
    time_limit_seconds: int = 30,
) -> ParallelMachineResult:
    """
    Solve Parallel Machine Scheduling - assign jobs to identical machines.

    Args:
        job_durations: Duration of each job.
        num_machines: Number of identical parallel machines.
        time_limit_seconds: Solver time limit.

    Returns:
        Machine assignments and makespan.
    """
    return solve_parallel_machines(job_durations, num_machines, time_limit_seconds)
