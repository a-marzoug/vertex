"""Tests for routing tools."""

import pytest

from vertex.tools.routing import compute_multi_depot_vrp, compute_pickup_delivery


@pytest.mark.asyncio
async def test_pickup_delivery():
    """Test VRP with Pickup and Delivery."""
    # 0: Depot
    # 1: Pickup A
    # 2: Delivery A
    # 3: Pickup B
    # 4: Delivery B
    locations = ["Depot", "P_A", "D_A", "P_B", "D_B"]

    # Distance matrix (symmetric for simplicity)
    # Depot close to P_A and P_B.
    # P_A close to D_A.
    # P_B close to D_B.
    # A and B far apart.
    dist = [
        [0, 10, 20, 10, 20],  # Depot
        [10, 0, 10, 50, 60],  # P_A
        [20, 10, 0, 60, 50],  # D_A
        [10, 50, 60, 0, 10],  # P_B
        [20, 60, 50, 10, 0],  # D_B
    ]

    # Requests: [Pickup Index, Delivery Index]
    pickups_deliveries = [
        [1, 2],  # A
        [3, 4],  # B
    ]

    # 2 Vehicles. Ideally one takes A, one takes B.
    result = await compute_pickup_delivery(
        locations=locations,
        distance_matrix=dist,
        pickups_deliveries=pickups_deliveries,
        num_vehicles=2,
        depot=0,
    )

    assert result.status == "optimal"
    assert len(result.routes) <= 2

    # Check precedence in routes
    for route in result.routes:
        stops = route.stops
        # If P_A is in route, D_A must be later
        if "P_A" in stops:
            assert "D_A" in stops
            assert stops.index("P_A") < stops.index("D_A")
        if "P_B" in stops:
            assert "D_B" in stops
            assert stops.index("P_B") < stops.index("D_B")


@pytest.mark.asyncio
async def test_tsp_config():
    """Test TSP with specific search strategy."""
    from vertex.config import RoutingMetaheuristic, RoutingStrategy
    from vertex.tools.routing import compute_tsp

    locations = ["A", "B", "C"]
    dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

    # Just ensure it runs without error
    result = await compute_tsp(
        locations=locations,
        distance_matrix=dist,
        search_strategy=RoutingStrategy.SAVINGS,
        local_search_metaheuristic=RoutingMetaheuristic.GREEDY_DESCENT,
    )
    assert result.status == "optimal"


@pytest.mark.asyncio
async def test_multi_depot_vrp():
    """Test VRP with multiple depots."""
    # 0: D1 (0,0)
    # 1: D2 (10,10)
    # 2: C1 (1,1)
    # 3: C2 (9,9)
    locations = ["D1", "D2", "C1", "C2"]

    # Distance matrix
    # D1->C1=2, D1->C2=18
    # D2->C1=18, D2->C2=2
    # C1->C2=16
    dist = [
        [0, 20, 2, 18],  # D1
        [20, 0, 18, 2],  # D2
        [2, 18, 0, 16],  # C1
        [18, 2, 16, 0],  # C2
    ]

    demands = [0, 0, 1, 1]
    vehicle_capacities = [5, 5]
    vehicle_starts = [0, 1]  # V1 at D1, V2 at D2
    vehicle_ends = [0, 1]

    result = await compute_multi_depot_vrp(
        locations=locations,
        distance_matrix=dist,
        demands=demands,
        vehicle_capacities=vehicle_capacities,
        vehicle_starts=vehicle_starts,
        vehicle_ends=vehicle_ends,
    )

    assert result.status == "optimal"
    assert len(result.routes) == 2

    # Check V1 (Vehicle 0) visits C1 (Index 2)
    # Check V2 (Vehicle 1) visits C2 (Index 3)
    for route in result.routes:
        if route.vehicle == 0:
            assert "C1" in route.stops
            assert "C2" not in route.stops
        elif route.vehicle == 1:
            assert "C2" in route.stops
            assert "C1" not in route.stops
