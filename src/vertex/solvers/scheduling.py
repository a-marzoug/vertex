"""Scheduling and Routing solvers using OR-Tools."""

import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from vertex.config import SolverStatus
from vertex.models.scheduling import (
    BinAssignment,
    BinPackingResult,
    JobShopResult,
    ScheduledTask,
    SetCoverResult,
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


def solve_vrp_time_windows(
    locations: list[str],
    time_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    demands: list[int],
    vehicle_capacities: list[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
) -> VRPResult:
    """Solve VRP with Time Windows."""
    start_time = time.time()
    n = len(locations)
    num_vehicles = len(vehicle_capacities)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return time_matrix[from_node][to_node]

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Time dimension
    max_time = max(tw[1] for tw in time_windows) + sum(max(row) for row in time_matrix)
    routing.AddDimension(transit_idx, max_time, max_time, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for loc_idx in range(n):
        index = manager.NodeToIndex(loc_idx)
        time_dimension.CumulVar(index).SetRange(time_windows[loc_idx][0], time_windows[loc_idx][1])

    # Capacity
    def demand_callback(idx):
        return demands[manager.IndexToNode(idx)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, vehicle_capacities, True, "Capacity")

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
        arrival_times = []
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stops.append(locations[node])
            arrival_times.append(solution.Value(time_dimension.CumulVar(index)))
            route_load += demands[node]
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, v)
        stops.append(locations[manager.IndexToNode(index)])
        arrival_times.append(solution.Value(time_dimension.CumulVar(index)))
        if len(stops) > 2:
            routes.append(VRPRoute(vehicle=v, stops=stops, distance=route_distance, load=route_load, arrival_times=arrival_times))
            total_distance += route_distance

    return VRPResult(status=SolverStatus.OPTIMAL, routes=routes, total_distance=total_distance, solve_time_ms=elapsed)


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
            interval_var = model.new_interval_var(start_var, duration, end_var, f"interval_{job_id}_{task_id}")
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var, machine, duration)
            machine_to_intervals[machine].append(interval_var)

    for intervals in machine_to_intervals:
        model.add_no_overlap(intervals)

    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            model.add(all_tasks[(job_id, task_id + 1)][0] >= all_tasks[(job_id, task_id)][1])

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
            job=job_id, task=task_id, machine=machine,
            start=solver.value(start_var), duration=duration, end=solver.value(end_var),
        ))

    return JobShopResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
        makespan=solver.value(makespan),
        schedule=sorted(schedule, key=lambda t: (t.job, t.task)),
        solve_time_ms=elapsed,
    )


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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return BinPackingResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    assignments = []
    for b in range(n_bins):
        if solver.value(y[b]):
            bin_items = [items[i] for i in range(n_items) if solver.value(x[(i, b)])]
            total_weight = sum(weights[item] for item in bin_items)
            assignments.append(BinAssignment(bin_id=b, items=bin_items, total_weight=total_weight))

    return BinPackingResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SetCoverResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    selected = [s for s in set_names if solver.value(x[s])]
    total_cost = sum(costs[s] for s in selected)

    return SetCoverResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return GraphColoringResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    coloring = {n: solver.value(color[i]) for i, n in enumerate(nodes)}
    num_colors = solver.value(max_color) + 1

    return GraphColoringResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
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
    from vertex.models.scheduling import CuttingPattern, CuttingStockResult
    
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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return CuttingStockResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    patterns = []
    total_waste = 0
    for s in range(n_stock):
        if solver.value(y[s]):
            cuts = {items[i]: solver.value(x[(s, i)]) for i in range(n_items) if solver.value(x[(s, i)]) > 0}
            used = sum(lengths[item] * count for item, count in cuts.items())
            waste = stock_length - used
            patterns.append(CuttingPattern(stock_id=s, cuts=cuts, waste=waste))
            total_waste += waste

    return CuttingStockResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
        num_stock_used=len(patterns),
        patterns=patterns,
        total_waste=total_waste,
        solve_time_ms=elapsed,
    )
