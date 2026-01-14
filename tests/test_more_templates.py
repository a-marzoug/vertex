"""Tests for additional template optimizations."""

from vertex.tools.templates.diet import optimize_diet


def test_diet_optimization():
    """Test diet optimization."""
    result = optimize_diet(
        foods=["rice", "beans"],
        nutrients=["protein", "calories"],
        costs={"rice": 2, "beans": 3},
        nutrition_values={
            "rice": {"protein": 5, "calories": 200},
            "beans": {"protein": 10, "calories": 150},
        },
        min_requirements={"protein": 50, "calories": 1000},
    )

    assert result.status in ["optimal", "feasible"]
    assert result.total_cost is not None
