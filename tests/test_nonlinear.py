"""Tests for Nonlinear Programming tool."""

import pytest

from vertex.tools.nonlinear import solve_minlp, solve_nonlinear_program


@pytest.mark.asyncio
async def test_nlp_unconstrained():
    """Test unconstrained optimization (Rosenbrock)."""
    variables = [
        {"name": "x", "initial_guess": 0.0},
        {"name": "y", "initial_guess": 0.0},
    ]
    objective = "(1-x)**2 + 100*(y-x**2)**2"

    result = await solve_nonlinear_program(
        variables=variables, objective_expression=objective
    )

    # Note: SLSQP might stop at local minima or close to optimal
    # With starting point (0,0), it should reach (1,1)
    assert result.status == "optimal"
    assert abs(result.variable_values["x"] - 1.0) < 0.01
    assert abs(result.variable_values["y"] - 1.0) < 0.01
    assert result.objective_value < 0.01


@pytest.mark.asyncio
async def test_nlp_constrained():
    """Test constrained optimization."""
    # Min x^2 + y^2 s.t. x+y >= 5
    variables = [
        {"name": "x", "initial_guess": 5.0},
        {"name": "y", "initial_guess": 5.0},
    ]
    objective = "x**2 + y**2"
    constraints = [{"expression": "x + y", "sense": ">=", "rhs": 5}]

    result = await solve_nonlinear_program(
        variables=variables,
        objective_expression=objective,
        constraints=constraints,
    )

    assert result.status == "optimal"
    assert abs(result.variable_values["x"] - 2.5) < 0.01
    assert abs(result.variable_values["y"] - 2.5) < 0.01
    assert abs(result.objective_value - 12.5) < 0.1


@pytest.mark.asyncio
async def test_minlp_simple():
    """Test MINLP with integer variables."""
    # Min x^2 + y^2 s.t. x+y >= 5, x integer, y continuous
    variables = [
        {"name": "x", "var_type": "integer", "lower_bound": 0, "upper_bound": 10},
        {"name": "y", "var_type": "continuous", "lower_bound": 0, "upper_bound": 10},
    ]
    objective = "x**2 + y**2"
    constraints = [{"expression": "x + y", "sense": ">=", "rhs": 5}]

    result = await solve_minlp(
        variables=variables,
        objective_expression=objective,
        constraints=constraints,
    )

    assert result.status in ["optimal", "feasible"]
    # Optimal should be x=2 or x=3, y=3 or y=2
    x_val = result.variable_values["x"]
    y_val = result.variable_values["y"]
    assert x_val == int(x_val)  # x should be integer
    assert x_val + y_val >= 4.9  # Constraint satisfied
    assert result.objective_value < 15  # Reasonable objective
