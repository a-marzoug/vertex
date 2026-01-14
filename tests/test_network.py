"""Tests for network optimization solvers."""

from vertex.config import SolverStatus
from vertex.solvers.network import (
    solve_max_flow,
    solve_min_cost_flow,
    solve_shortest_path,
)


def test_max_flow():
    """Test maximum flow computation."""
    nodes = ["s", "a", "b", "t"]
    arcs = [
        {"source": "s", "target": "a", "capacity": 10},
        {"source": "s", "target": "b", "capacity": 10},
        {"source": "a", "target": "t", "capacity": 10},
        {"source": "b", "target": "t", "capacity": 10},
    ]

    result = solve_max_flow(nodes, arcs, "s", "t")

    assert result.status == SolverStatus.OPTIMAL
    assert result.max_flow == 20


def test_min_cost_flow():
    """Test minimum cost flow."""
    nodes = ["s", "a", "t"]
    arcs = [
        {"source": "s", "target": "a", "capacity": 10, "cost": 1},
        {"source": "a", "target": "t", "capacity": 10, "cost": 1},
    ]
    supplies = {"s": 5, "t": -5}

    result = solve_min_cost_flow(nodes, arcs, supplies)

    assert result.status == SolverStatus.OPTIMAL
    assert result.total_cost == 10


def test_shortest_path():
    """Test shortest path computation."""
    nodes = ["a", "b", "c"]
    arcs = [
        {"source": "a", "target": "b", "cost": 1},
        {"source": "b", "target": "c", "cost": 2},
        {"source": "a", "target": "c", "cost": 5},
    ]

    result = solve_shortest_path(nodes, arcs, "a", "c")

    assert result.status == SolverStatus.OPTIMAL
    assert result.distance == 3
    assert result.path == ["a", "b", "c"]
