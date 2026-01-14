"""Tests for analysis tools."""

from vertex.tools.analysis import get_model_stats
from vertex.tools.sensitivity import analyze_sensitivity


def test_sensitivity_analysis():
    """Test sensitivity analysis."""
    variables = [
        {"name": "x", "lower_bound": 0},
        {"name": "y", "lower_bound": 0},
    ]
    constraints = [{"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}]
    objective = {"x": 1, "y": 1}

    result = analyze_sensitivity(
        variables=variables,
        constraints=constraints,
        objective_coefficients=objective,
        objective_sense="maximize",
    )

    assert result.status.value == "optimal"
    assert result.shadow_prices is not None


def test_model_statistics():
    """Test model statistics computation."""
    variables = [
        {"name": "x", "var_type": "continuous"},
        {"name": "y", "var_type": "integer"},
    ]
    constraints = [
        {"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10},
        {"coefficients": {"x": 2}, "sense": ">=", "rhs": 5},
    ]

    result = get_model_stats(variables, constraints)

    assert result.num_variables == 2
    assert result.num_constraints == 2
    assert result.variable_types["continuous"] == 1
    assert result.variable_types["integer"] == 1
