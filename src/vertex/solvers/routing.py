"""Routing solvers using OR-Tools."""

import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vertex.config import RoutingMetaheuristic, RoutingStrategy, SolverStatus
from vertex.models.routing import TSPResult, VRPResult, VRPRoute


def _get_routing_enums(strategy_name: str, metaheuristic_name: str) -> tuple[int, int]:
    """Map string config to OR-Tools enums."""
    # Strategy
    if strategy_name == "AUTOMATIC":
        strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    else:
        try:
            strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, strategy_name)
        except AttributeError:
            strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Metaheuristic
    if metaheuristic_name == "AUTOMATIC":
        metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    else:
        try:
            metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic_name
            )
        except AttributeError:
            metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )

    return strategy, metaheuristic


def solve_tsp(
    locations: list[str],
    distance_matrix: list[list[float]],
    time_limit_seconds: int = 30,
    search_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
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

    strategy, metaheuristic = _get_routing_enums(
        search_strategy, local_search_metaheuristic
    )
    search_params.first_solution_strategy = strategy
    search_params.local_search_metaheuristic = metaheuristic
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
    search_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
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
    strategy, metaheuristic = _get_routing_enums(
        search_strategy, local_search_metaheuristic
    )
    search_params.first_solution_strategy = strategy
    search_params.local_search_metaheuristic = metaheuristic
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
            routes.append(
                VRPRoute(
                    vehicle=v, stops=stops, distance=route_distance, load=route_load
                )
            )
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
    search_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
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
        time_dimension.CumulVar(index).SetRange(
            time_windows[loc_idx][0], time_windows[loc_idx][1]
        )

    # Capacity
    def demand_callback(idx):
        return demands[manager.IndexToNode(idx)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, vehicle_capacities, True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    strategy, metaheuristic = _get_routing_enums(
        search_strategy, local_search_metaheuristic
    )
    search_params.first_solution_strategy = strategy
    search_params.local_search_metaheuristic = metaheuristic
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
            routes.append(
                VRPRoute(
                    vehicle=v,
                    stops=stops,
                    distance=route_distance,
                    load=route_load,
                    arrival_times=arrival_times,
                )
            )
            total_distance += route_distance

    return VRPResult(
        status=SolverStatus.OPTIMAL,
        routes=routes,
        total_distance=total_distance,
        solve_time_ms=elapsed,
    )


def solve_pickup_delivery(
    locations: list[str],
    distance_matrix: list[list[float]],
    pickups_deliveries: list[list[int]],
    num_vehicles: int,
    depot: int = 0,
    time_limit_seconds: int = 30,
    search_strategy: str = "PARALLEL_CHEAPEST_INSERTION",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> VRPResult:
    """Solve VRP with Pickup and Delivery constraints."""
    start_time = time.time()
    n = len(locations)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Add Distance Dimension to enforce Precedence (Pickup before Delivery)
    routing.AddDimension(
        transit_idx,
        0,  # no slack
        3000000,  # max distance per vehicle (large number)
        True,  # start cumul to zero
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")

    for request in pickups_deliveries:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])

        routing.AddPickupAndDelivery(pickup_index, delivery_index)

        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )

        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index)
            <= distance_dimension.CumulVar(delivery_index)
        )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    strategy, metaheuristic = _get_routing_enums(
        search_strategy, local_search_metaheuristic
    )
    search_params.first_solution_strategy = strategy
    search_params.local_search_metaheuristic = metaheuristic
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
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stops.append(locations[node])
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, v)
        stops.append(locations[manager.IndexToNode(index)])
        if len(stops) > 2:
            routes.append(
                VRPRoute(
                    vehicle=v,
                    stops=stops,
                    distance=route_distance,
                )
            )
            total_distance += route_distance

    return VRPResult(
        status=SolverStatus.OPTIMAL,
        routes=routes,
        total_distance=total_distance,
        solve_time_ms=elapsed,
    )


def solve_multi_depot_vrp(
    locations: list[str],
    distance_matrix: list[list[float]],
    demands: list[int],
    vehicle_capacities: list[int],
    vehicle_starts: list[int],
    vehicle_ends: list[int],
    time_limit_seconds: int = 30,
    search_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> VRPResult:
    """Solve VRP with Multiple Depots."""
    start_time = time.time()
    n = len(locations)
    num_vehicles = len(vehicle_capacities)

    manager = pywrapcp.RoutingIndexManager(
        n, num_vehicles, vehicle_starts, vehicle_ends
    )
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
    strategy, metaheuristic = _get_routing_enums(
        search_strategy, local_search_metaheuristic
    )
    search_params.first_solution_strategy = strategy
    search_params.local_search_metaheuristic = metaheuristic
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
        if len(stops) > 2:
            routes.append(
                VRPRoute(
                    vehicle=v, stops=stops, distance=route_distance, load=route_load
                )
            )
            total_distance += route_distance

    return VRPResult(
        status=SolverStatus.OPTIMAL,
        routes=routes,
        total_distance=total_distance,
        solve_time_ms=elapsed,
    )
