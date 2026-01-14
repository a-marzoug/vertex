"""Tests for multi-objective optimization."""

from vertex.tools.multiobjective import solve_multi_objective


def test_multi_objective():
    """Test multi-objective optimization."""
    variables = [
        {"name": "x", "lower_bound": 0, "upper_bound": 10},
        {"name": "y", "lower_bound": 0, "upper_bound": 10},
    ]
    constraints = [{"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}]
    objectives = {
        "profit": {"x": 2, "y": 3},
        "quality": {"x": 3, "y": 1},
    }

    result = solve_multi_objective(
        variables=variables,
        constraints=constraints,
        objectives=objectives,
        num_points=3,
    )

    assert result.status.value in ["optimal", "feasible"]
    assert len(result.pareto_points) > 0
