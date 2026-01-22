"""Scheduling and Routing MCP tools."""

from typing import Any

from vertex.models.scheduling import (
    BinPackingResult,
    CuttingStockResult,
    FlowShopResult,
    GraphColoringResult,
    JobShopResult,
    ParallelMachineResult,
    RCPSPResult,
    SetCoverResult,
)
from vertex.solvers.scheduling import (
    solve_bin_packing,
    solve_cutting_stock,
    solve_flow_shop,
    solve_graph_coloring,
    solve_job_shop,
    solve_parallel_machines,
    solve_set_cover,
)
from vertex.utils.async_utils import run_in_executor
from vertex.utils.visualization import create_gantt_chart


async def compute_job_shop(
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
    return await run_in_executor(solve_job_shop, jobs_tuples, time_limit_seconds)


async def compute_bin_packing(
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
    return await run_in_executor(
        solve_bin_packing, items, weights, bin_capacity, max_bins, time_limit_seconds
    )


async def compute_set_cover(
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
    return await run_in_executor(
        solve_set_cover, universe, sets, costs, time_limit_seconds
    )


async def compute_graph_coloring(
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
    return await run_in_executor(
        solve_graph_coloring, nodes, edges, max_colors, time_limit_seconds
    )


async def compute_cutting_stock(
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
    return await run_in_executor(
        solve_cutting_stock,
        items,
        lengths,
        demands,
        stock_length,
        max_stock,
        time_limit_seconds,
    )


async def compute_flexible_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> dict[str, Any]:
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

    return await run_in_executor(solve_flexible_job_shop, jobs, time_limit_seconds)


async def compute_flow_shop(
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
    return await run_in_executor(solve_flow_shop, processing_times, time_limit_seconds)


async def compute_parallel_machines(
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
    return await run_in_executor(
        solve_parallel_machines, job_durations, num_machines, time_limit_seconds
    )


async def solve_rcpsp(
    tasks: list[dict[str, Any]],
    resources: dict[str, int],
    time_limit_seconds: int = 30,
) -> RCPSPResult:
    """
    Solve Resource-Constrained Project Scheduling Problem.

    Args:
        tasks: List of tasks with 'name', 'duration', 'resources' (dict), 'predecessors' (list).
        resources: Available capacity per resource type.
        time_limit_seconds: Solver time limit.

    Returns:
        RCPSPResult with status, makespan, schedule, and visualization.
    """

    # Define inner blocking function to run in executor
    def _solve_blocking() -> RCPSPResult:
        import time

        from ortools.sat.python import cp_model

        from vertex.config import SolverStatus

        start_time = time.time()
        model = cp_model.CpModel()

        horizon = sum(t["duration"] for t in tasks)

        # Variables
        starts = {}
        ends = {}
        intervals = {}

        for t in tasks:
            name = t["name"]
            starts[name] = model.new_int_var(0, horizon, f"start_{name}")
            ends[name] = model.new_int_var(0, horizon, f"end_{name}")
            intervals[name] = model.new_interval_var(
                starts[name], t["duration"], ends[name], f"interval_{name}"
            )

        # Precedence constraints
        for t in tasks:
            for pred in t.get("predecessors", []):
                model.add(starts[t["name"]] >= ends[pred])

        # Resource constraints using cumulative
        for res_name, capacity in resources.items():
            task_intervals = []
            demands = []
            for t in tasks:
                if t.get("resources", {}).get(res_name, 0) > 0:
                    task_intervals.append(intervals[t["name"]])
                    demands.append(t["resources"][res_name])
            if task_intervals:
                model.add_cumulative(task_intervals, demands, capacity)

        # Minimize makespan
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, [ends[t["name"]] for t in tasks])
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        status = solver.solve(model)
        elapsed = (time.time() - start_time) * 1000

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return RCPSPResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

        schedule = [
            {
                "task": t["name"],
                "start": solver.value(starts[t["name"]]),
                "end": solver.value(ends[t["name"]]),
                "duration": t["duration"],
                "resource": "Project",  # Placeholder for visualization lane
            }
            for t in tasks
        ]

        status_enum = (
            SolverStatus.OPTIMAL
            if status == cp_model.OPTIMAL
            else SolverStatus.FEASIBLE
        )

        result = RCPSPResult(
            status=status_enum,
            makespan=solver.value(makespan),
            schedule=sorted(schedule, key=lambda x: x["start"]),
            solve_time_ms=elapsed,
        )

        result.visualization = create_gantt_chart(
            result.schedule,
            title="Project Schedule",
            resource_key="resource",
            task_key="task",
        )

        return result

    return await run_in_executor(_solve_blocking)
