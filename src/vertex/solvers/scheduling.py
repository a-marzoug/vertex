"""Scheduling and Routing solvers using OR-Tools."""

import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from vertex.config import SolverStatus
from vertex.models.scheduling import (
    JobShopResult,
    ScheduledTask,
    TSPResult,
    VRPResult,
    VRPRoute,
)


def solve_tsp(
    locations: list[str],
    distance_matrix: list[list[float]],
    time_limit_seconds: int = 30,
) -> TSPResult:
    """Solve Traveling Salesman Problem using OR-Tools routing."""
    start_time = time.time()
    n = len(locations)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_params)
    elapsed = (time.time() - start_time) * 1000

    if not solution:
        return TSPResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(locations[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    route.append(locations[manager.IndexToNode(index)])

    return TSPResult(
        status=SolverStatus.OPTIMAL,
        route=route,
        total_distance=solution.ObjectiveValue(),
        solve_time_ms=elapsed,
    )


def solve_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
) -> VRPResult:
    """Solve Capacitated Vehicle Routing Problem."""
    start_time = time.time()
    n = len(locations)
    num_vehicles = len(vehicle_capacities)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(idx):
        return demands[manager.IndexToNode(idx)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, vehicle_capacities, True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_params)
    elapsed = (time.time() - start_time) * 1000

    if not solution:
        return VRPResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    routes = []
    total_distance = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        stops = []
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stops.append(locations[node])
            route_load += demands[node]
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, v)
        stops.append(locations[manager.IndexToNode(index)])
        if len(stops) > 2:  # Has actual stops beyond depot
            routes.append(VRPRoute(vehicle=v, stops=stops, distance=route_distance, load=route_load))
            total_distance += route_distance

    return VRPResult(
        status=SolverStatus.OPTIMAL,
        routes=routes,
        total_distance=total_distance,
        solve_time_ms=elapsed,
    )


def solve_job_shop(
    jobs: list[list[tuple[int, int]]],
    time_limit_seconds: int = 30,
) -> JobShopResult:
    """
    Solve Job Shop Scheduling Problem using CP-SAT.

    Args:
        jobs: List of jobs, each job is list of (machine_id, duration) tuples.
        time_limit_seconds: Solver time limit.
    """
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
            interval_var = model.new_interval_var(start_var, duration, end_var, f"interval_{job_id}_{task_id}")
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var, machine, duration)
            machine_to_intervals[machine].append(interval_var)

    # No overlap on machines
    for intervals in machine_to_intervals:
        model.add_no_overlap(intervals)

    # Precedence within jobs
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            model.add(all_tasks[(job_id, task_id + 1)][0] >= all_tasks[(job_id, task_id)][1])

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [all_tasks[(job_id, len(job) - 1)][1] for job_id, job in enumerate(jobs)])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return JobShopResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    schedule = []
    for (job_id, task_id), (start_var, end_var, _, machine, duration) in all_tasks.items():
        schedule.append(ScheduledTask(
            job=job_id,
            task=task_id,
            machine=machine,
            start=solver.value(start_var),
            duration=duration,
            end=solver.value(end_var),
        ))

    return JobShopResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
        makespan=solver.value(makespan),
        schedule=sorted(schedule, key=lambda t: (t.job, t.task)),
        solve_time_ms=elapsed,
    )
