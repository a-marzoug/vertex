"""Tests for scheduling optimization tools."""

from vertex.tools.scheduling import (
    compute_job_shop,
    compute_tsp,
    compute_vrp,
)


def test_tsp():
    """Test Traveling Salesman Problem."""
    locations = ["A", "B", "C", "D"]
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]

    result = compute_tsp(locations, distance_matrix)

    assert result.status.value == "optimal"
    assert result.total_distance is not None
    assert result.total_distance > 0


def test_vrp():
    """Test Vehicle Routing Problem."""
    locations = ["depot", "A", "B", "C"]
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    demands = [0, 1, 1, 2]

    result = compute_vrp(
        locations=locations,
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacities=[3, 3],
    )

    assert result.status.value in ["optimal", "feasible"]
    assert result.routes is not None


def test_job_shop():
    """Test Job Shop Scheduling."""
    jobs = [
        [{"machine": 0, "duration": 3}, {"machine": 1, "duration": 2}],
        [{"machine": 1, "duration": 2}, {"machine": 0, "duration": 1}],
    ]

    result = compute_job_shop(jobs)

    assert result.status.value in ["optimal", "feasible"]
    assert result.makespan is not None
