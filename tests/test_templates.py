"""Tests for template optimization functions."""

from vertex.config import SolverStatus
from vertex.tools.templates.assignment import optimize_assignment
from vertex.tools.templates.knapsack import optimize_knapsack
from vertex.tools.templates.production import optimize_production


def test_production_optimization():
    """Test production planning optimization."""
    result = optimize_production(
        products=["A", "B"],
        resources=["labor", "material"],
        profits={"A": 40, "B": 30},
        requirements={
            "A": {"labor": 1, "material": 2},
            "B": {"labor": 2, "material": 1},
        },
        availability={"labor": 100, "material": 100},
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.total_profit is not None
    assert result.total_profit > 0


def test_assignment_optimization():
    """Test worker assignment optimization."""
    result = optimize_assignment(
        workers=["W1", "W2"],
        tasks=["T1", "T2"],
        costs={"W1": {"T1": 10, "T2": 20}, "W2": {"T1": 15, "T2": 10}},
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.total_cost is not None
    assert len(result.assignments) == 2


def test_knapsack_optimization():
    """Test knapsack selection optimization."""
    result = optimize_knapsack(
        items=["item1", "item2", "item3"],
        values={"item1": 60, "item2": 100, "item3": 120},
        weights={"item1": 10, "item2": 20, "item3": 30},
        capacity=50,
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.total_value is not None
    assert result.total_value > 0
