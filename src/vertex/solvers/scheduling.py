"""Scheduling and Routing solvers using OR-Tools."""

import time
from typing import Any

from ortools.sat.python import cp_model

from vertex.config import SolverStatus
from vertex.models.scheduling import (
    BinAssignment,
    BinPackingResult,
    CuttingPattern,
    CuttingStockResult,
    FlowShopResult,
    GraphColoringResult,
    JobShopResult,
    ParallelMachineResult,
    ScheduledTask,
    SetCoverResult,
)
from vertex.utils.visualization import create_gantt_chart


def solve_job_shop(
    jobs: list[list[tuple[int, int]]],
    time_limit_seconds: int = 30,
) -> JobShopResult:
    """Solve Job Shop Scheduling Problem using CP-SAT."""
    start_time = time.time()
    model = cp_model.CpModel()

    num_machines = 1 + max(task[0] for job in jobs for task in job)
    horizon = sum(task[1] for job in jobs for task in job)

    all_tasks = {}
    machine_to_intervals = [[] for _ in range(num_machines)]

    for job_id, job in enumerate(jobs):
        for task_id, (machine, duration) in enumerate(job):
            start_var = model.new_int_var(0, horizon, f"start_{job_id}_{task_id}")
            end_var = model.new_int_var(0, horizon, f"end_{job_id}_{task_id}")
            interval_var = model.new_interval_var(
                start_var, duration, end_var, f"interval_{job_id}_{task_id}"
            )
            all_tasks[(job_id, task_id)] = (
                start_var,
                end_var,
                interval_var,
                machine,
                duration,
            )
            machine_to_intervals[machine].append(interval_var)

    for intervals in machine_to_intervals:
        model.add_no_overlap(intervals)

    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            model.add(
                all_tasks[(job_id, task_id + 1)][0] >= all_tasks[(job_id, task_id)][1]
            )

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        makespan,
        [all_tasks[(job_id, len(job) - 1)][1] for job_id, job in enumerate(jobs)],
    )
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return JobShopResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    schedule = []
    for (job_id, task_id), (
        start_var,
        end_var,
        _,
        machine,
        duration,
    ) in all_tasks.items():
        schedule.append(
            ScheduledTask(
                job=job_id,
                task=task_id,
                machine=machine,
                start=solver.value(start_var),
                duration=duration,
                end=solver.value(end_var),
            )
        )

    result = JobShopResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        makespan=solver.value(makespan),
        schedule=sorted(schedule, key=lambda t: (t.job, t.task)),
        solve_time_ms=elapsed,
    )
    result.visualization = create_gantt_chart(
        result.schedule,
        title="Job Shop Schedule",
        resource_key="machine",
        task_key="task",
    )
    return result


def solve_bin_packing(
    items: list[str],
    weights: dict[str, float],
    bin_capacity: float,
    max_bins: int | None = None,
    time_limit_seconds: int = 30,
) -> BinPackingResult:
    """Solve Bin Packing Problem using CP-SAT."""
    start_time = time.time()
    model = cp_model.CpModel()

    n_items = len(items)
    n_bins = max_bins or n_items

    # x[i, b] = 1 if item i in bin b
    x = {}
    for i in range(n_items):
        for b in range(n_bins):
            x[(i, b)] = model.new_bool_var(f"x_{i}_{b}")

    # y[b] = 1 if bin b is used
    y = [model.new_bool_var(f"y_{b}") for b in range(n_bins)]

    # Each item in exactly one bin
    for i in range(n_items):
        model.add_exactly_one(x[(i, b)] for b in range(n_bins))

    # Capacity constraint
    for b in range(n_bins):
        model.add(
            sum(int(weights[items[i]]) * x[(i, b)] for i in range(n_items))
            <= int(bin_capacity) * y[b]
        )

    # Minimize bins used
    model.minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return BinPackingResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    assignments = []
    for b in range(n_bins):
        if solver.value(y[b]):
            bin_items = [items[i] for i in range(n_items) if solver.value(x[(i, b)])]
            total_weight = sum(weights[item] for item in bin_items)
            assignments.append(
                BinAssignment(bin_id=b, items=bin_items, total_weight=total_weight)
            )

    return BinPackingResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        num_bins_used=len(assignments),
        assignments=assignments,
        solve_time_ms=elapsed,
    )


def solve_set_cover(
    universe: list[str],
    sets: dict[str, list[str]],
    costs: dict[str, float],
    time_limit_seconds: int = 30,
) -> SetCoverResult:
    """Solve Set Covering Problem using CP-SAT."""
    start_time = time.time()
    model = cp_model.CpModel()

    set_names = list(sets.keys())
    element_to_sets = {e: [] for e in universe}
    for name, elements in sets.items():
        for e in elements:
            if e in element_to_sets:
                element_to_sets[e].append(name)

    # x[s] = 1 if set s is selected
    x = {name: model.new_bool_var(f"x_{name}") for name in set_names}

    # Each element must be covered
    for element, covering_sets in element_to_sets.items():
        model.add(sum(x[s] for s in covering_sets) >= 1)

    # Minimize cost
    model.minimize(sum(int(costs[s] * 1000) * x[s] for s in set_names))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return SetCoverResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    selected = [s for s in set_names if solver.value(x[s])]
    total_cost = sum(costs[s] for s in selected)

    return SetCoverResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        selected_sets=selected,
        total_cost=total_cost,
        solve_time_ms=elapsed,
    )


def solve_graph_coloring(
    nodes: list[str],
    edges: list[tuple[str, str]],
    max_colors: int | None = None,
    time_limit_seconds: int = 30,
) -> "GraphColoringResult":
    """Solve Graph Coloring Problem using CP-SAT."""
    from vertex.models.scheduling import GraphColoringResult

    start_time = time.time()
    model = cp_model.CpModel()

    n_colors = max_colors or len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # color[i] = color of node i
    color = [model.new_int_var(0, n_colors - 1, f"color_{n}") for n in nodes]

    # Adjacent nodes must have different colors
    for u, v in edges:
        model.add(color[node_idx[u]] != color[node_idx[v]])

    # Minimize max color used
    max_color = model.new_int_var(0, n_colors - 1, "max_color")
    model.add_max_equality(max_color, color)
    model.minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return GraphColoringResult(
            status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed
        )

    coloring = {n: solver.value(color[i]) for i, n in enumerate(nodes)}
    num_colors = solver.value(max_color) + 1

    return GraphColoringResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        num_colors=num_colors,
        coloring=coloring,
        solve_time_ms=elapsed,
    )


def solve_cutting_stock(
    items: list[str],
    lengths: dict[str, int],
    demands: dict[str, int],
    stock_length: int,
    max_stock: int | None = None,
    time_limit_seconds: int = 30,
) -> "CuttingStockResult":
    """Solve Cutting Stock Problem using CP-SAT."""
    from vertex.models.scheduling import CuttingStockResult

    start_time = time.time()
    model = cp_model.CpModel()

    n_items = len(items)
    n_stock = max_stock or sum(demands.values())

    # x[s, i] = number of item i cut from stock s
    x = {}
    for s in range(n_stock):
        for i, item in enumerate(items):
            max_cuts = stock_length // lengths[item]
            x[(s, i)] = model.new_int_var(0, max_cuts, f"x_{s}_{i}")

    # y[s] = 1 if stock s is used
    y = [model.new_bool_var(f"y_{s}") for s in range(n_stock)]

    # Meet demand for each item
    for i, item in enumerate(items):
        model.add(sum(x[(s, i)] for s in range(n_stock)) >= demands[item])

    # Capacity constraint per stock
    for s in range(n_stock):
        model.add(
            sum(lengths[items[i]] * x[(s, i)] for i in range(n_items))
            <= stock_length * y[s]
        )

    # Minimize stock used
    model.minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return CuttingStockResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    patterns = []
    total_waste = 0
    for s in range(n_stock):
        if solver.value(y[s]):
            cuts = {
                items[i]: solver.value(x[(s, i)])
                for i in range(n_items)
                if solver.value(x[(s, i)]) > 0
            }
            used = sum(lengths[item] * count for item, count in cuts.items())
            waste = stock_length - used
            patterns.append(CuttingPattern(stock_id=s, cuts=cuts, waste=waste))
            total_waste += waste

    return CuttingStockResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        num_stock_used=len(patterns),
        patterns=patterns,
        total_waste=total_waste,
        solve_time_ms=elapsed,
    )


def solve_flexible_job_shop(
    jobs: list[list[dict]],
    time_limit_seconds: int = 30,
) -> dict[str, Any]:
    """
    Solve Flexible Job Shop - tasks can run on multiple machines.

    Args:
        jobs: List of jobs. Each job is list of tasks.
            Each task: {"machines": [(machine_id, duration), ...]}
        time_limit_seconds: Solver time limit.

    Returns:
        Dict with status, makespan, and schedule.
    """
    import time

    from ortools.sat.python import cp_model

    start_time = time.time()
    model = cp_model.CpModel()

    # Compute horizon
    horizon = sum(
        max(dur for _, dur in task["machines"]) for job in jobs for task in job
    )

    all_tasks = {}
    machine_intervals = {}

    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            task_alts = []
            presences = []
            for machine, duration in task["machines"]:
                suffix = f"_{job_id}_{task_id}_{machine}"
                start = model.new_int_var(0, horizon, f"start{suffix}")
                end = model.new_int_var(0, horizon, f"end{suffix}")
                present = model.new_bool_var(f"present{suffix}")
                interval = model.new_optional_interval_var(
                    start, duration, end, present, f"interval{suffix}"
                )
                task_alts.append((start, end, interval, machine, duration, present))
                presences.append(present)
                machine_intervals.setdefault(machine, []).append(interval)
            all_tasks[(job_id, task_id)] = task_alts
            model.add_exactly_one(presences)

    # No overlap on machines
    for intervals in machine_intervals.values():
        model.add_no_overlap(intervals)

    # Precedence within jobs
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            for alt1 in all_tasks[(job_id, task_id)]:
                for alt2 in all_tasks[(job_id, task_id + 1)]:
                    model.add(alt2[0] >= alt1[1]).only_enforce_if(alt1[5], alt2[5])

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    last_ends = []
    for job_id, job in enumerate(jobs):
        for alt in all_tasks[(job_id, len(job) - 1)]:
            last_ends.append(alt[1])
    model.add_max_equality(makespan, last_ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return {"status": "infeasible", "solve_time_ms": elapsed}

    schedule = []
    for (job_id, task_id), alts in all_tasks.items():
        for start, end, interval, machine, duration, present in alts:
            if solver.value(present):
                schedule.append(
                    {
                        "job": job_id,
                        "task": task_id,
                        "machine": machine,
                        "start": solver.value(start),
                        "end": solver.value(end),
                    }
                )

    return {
        "status": "optimal" if status == cp_model.OPTIMAL else "feasible",  # type: ignore[comparison-overlap]
        "makespan": solver.value(makespan),
        "schedule": sorted(schedule, key=lambda x: (x["job"], x["task"])),
        "solve_time_ms": elapsed,
    }


def solve_flow_shop(
    processing_times: list[list[int]],
    time_limit_seconds: int = 30,
) -> "FlowShopResult":
    """
    Solve Flow Shop Scheduling - all jobs follow same machine sequence.

    Args:
        processing_times: processing_times[job][machine] = duration
        time_limit_seconds: Solver time limit.
    """
    from vertex.models.scheduling import FlowShopResult, ScheduledTask

    start_time = time.time()
    model = cp_model.CpModel()

    n_jobs = len(processing_times)
    n_machines = len(processing_times[0]) if n_jobs > 0 else 0
    horizon = sum(sum(job) for job in processing_times)

    # Variables: start[j][m] = start time of job j on machine m
    starts = {}
    ends = {}
    intervals = {}

    for j in range(n_jobs):
        for m in range(n_machines):
            dur = processing_times[j][m]
            starts[(j, m)] = model.new_int_var(0, horizon, f"start_{j}_{m}")
            ends[(j, m)] = model.new_int_var(0, horizon, f"end_{j}_{m}")
            intervals[(j, m)] = model.new_interval_var(
                starts[(j, m)], dur, ends[(j, m)], f"interval_{j}_{m}"
            )

    # Precedence: job j must finish machine m before starting m+1
    for j in range(n_jobs):
        for m in range(n_machines - 1):
            model.add(starts[(j, m + 1)] >= ends[(j, m)])

    # No overlap on each machine
    for m in range(n_machines):
        model.add_no_overlap([intervals[(j, m)] for j in range(n_jobs)])

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[(j, n_machines - 1)] for j in range(n_jobs)])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return FlowShopResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    # Extract job sequence (order on first machine)
    first_machine_starts = [(j, solver.value(starts[(j, 0)])) for j in range(n_jobs)]
    job_sequence = [j for j, _ in sorted(first_machine_starts, key=lambda x: x[1])]

    schedule = []
    for j in range(n_jobs):
        for m in range(n_machines):
            schedule.append(
                ScheduledTask(
                    job=j,
                    task=m,
                    machine=m,
                    start=solver.value(starts[(j, m)]),
                    duration=processing_times[j][m],
                    end=solver.value(ends[(j, m)]),
                )
            )

    result = FlowShopResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        makespan=solver.value(makespan),
        job_sequence=job_sequence,
        schedule=sorted(schedule, key=lambda t: (t.job, t.task)),
        solve_time_ms=elapsed,
    )
    result.visualization = create_gantt_chart(
        result.schedule,
        title="Flow Shop Schedule",
        resource_key="machine",
        task_key="task",
    )
    return result


def solve_parallel_machines(
    job_durations: list[int],
    num_machines: int,
    time_limit_seconds: int = 30,
) -> "ParallelMachineResult":
    """
    Solve Parallel Machine Scheduling - assign jobs to identical machines.

    Args:
        job_durations: Duration of each job.
        num_machines: Number of identical machines.
        time_limit_seconds: Solver time limit.
    """
    from vertex.models.scheduling import ParallelMachineResult, ScheduledTask

    start_time = time.time()
    model = cp_model.CpModel()

    n_jobs = len(job_durations)
    horizon = sum(job_durations)

    # Variables
    starts = [model.new_int_var(0, horizon, f"start_{j}") for j in range(n_jobs)]
    ends = [model.new_int_var(0, horizon, f"end_{j}") for j in range(n_jobs)]
    machine_assign = [
        model.new_int_var(0, num_machines - 1, f"machine_{j}") for j in range(n_jobs)
    ]

    # Intervals per machine (optional)
    intervals_per_machine = [[] for _ in range(num_machines)]
    presences = {}

    for j in range(n_jobs):
        for m in range(num_machines):
            present = model.new_bool_var(f"present_{j}_{m}")
            interval = model.new_optional_interval_var(
                starts[j], job_durations[j], ends[j], present, f"interval_{j}_{m}"
            )
            intervals_per_machine[m].append(interval)
            presences[(j, m)] = present
            model.add(machine_assign[j] == m).only_enforce_if(present)
        model.add_exactly_one(presences[(j, m)] for m in range(num_machines))

    # No overlap on each machine
    for m in range(num_machines):
        model.add_no_overlap(intervals_per_machine[m])

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return ParallelMachineResult(
            status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed
        )

    machine_assignments = {m: [] for m in range(num_machines)}
    schedule = []
    for j in range(n_jobs):
        m = solver.value(machine_assign[j])
        machine_assignments[m].append(j)
        schedule.append(
            ScheduledTask(
                job=j,
                task=0,
                machine=m,
                start=solver.value(starts[j]),
                duration=job_durations[j],
                end=solver.value(ends[j]),
            )
        )

    result = ParallelMachineResult(
        status=SolverStatus.OPTIMAL
        if status == cp_model.OPTIMAL  # type: ignore[comparison-overlap]
        else SolverStatus.FEASIBLE,
        makespan=solver.value(makespan),
        machine_assignments=machine_assignments,
        schedule=sorted(schedule, key=lambda t: (t.machine, t.start)),
        solve_time_ms=elapsed,
    )
    result.visualization = create_gantt_chart(
        result.schedule,
        title="Parallel Machine Schedule",
        resource_key="machine",
        task_key="job",
    )
    return result
