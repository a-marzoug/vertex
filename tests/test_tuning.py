"""Tests for solver selection and self-tuning."""

import pytest

from vertex.tools.tuning import select_solver


def test_select_solver_lp():
    """Test LP solver selection."""
    result = select_solver(num_variables=100, num_constraints=50)

    assert result.recommended_tool == "solve_linear_program"
    assert "GLOP" in result.reasoning
    assert result.expected_performance == "fast"


def test_select_solver_mip_small():
    """Test small MIP solver selection."""
    result = select_solver(
        num_variables=50, num_constraints=30, has_integer_variables=True
    )

    assert result.recommended_tool == "solve_mixed_integer_program"
    assert "SCIP" in result.reasoning
    assert result.expected_performance == "fast"


def test_select_solver_mip_large():
    """Test large MIP solver selection."""
    result = select_solver(
        num_variables=15000, num_constraints=8000, has_integer_variables=True
    )

    assert result.recommended_tool == "solve_mixed_integer_program"
    assert "SCIP" in result.reasoning
    assert result.expected_performance == "slow"
    assert "time_limit" in result.solver_hints


def test_select_solver_nlp():
    """Test NLP solver selection."""
    result = select_solver(
        num_variables=10, num_constraints=5, is_nonlinear=True
    )

    assert result.recommended_tool == "solve_nonlinear_program"
    assert "SLSQP" in result.reasoning
    assert result.expected_performance == "medium"


def test_select_solver_minlp():
    """Test MINLP solver selection."""
    result = select_solver(
        num_variables=20,
        num_constraints=10,
        is_nonlinear=True,
        has_integer_variables=True,
    )

    assert result.recommended_tool == "solve_minlp"
    assert "Differential Evolution" in result.reasoning
    assert result.expected_performance == "slow"


def test_select_solver_qp():
    """Test QP solver selection."""
    result = select_solver(
        num_variables=50, num_constraints=20, is_quadratic=True
    )

    assert result.recommended_tool == "solve_qp"
    assert "OSQP" in result.reasoning
    assert result.expected_performance == "fast"


def test_select_solver_network():
    """Test network flow solver selection."""
    result = select_solver(
        num_variables=100, num_constraints=80, is_network_flow=True
    )

    assert result.recommended_tool == "compute_max_flow"
    assert "network simplex" in result.reasoning
    assert result.expected_performance == "fast"


def test_select_solver_routing():
    """Test routing solver selection."""
    result = select_solver(
        num_variables=50,
        num_constraints=30,
        is_network_flow=True,
        is_routing=True,
        has_integer_variables=True,
    )

    assert result.recommended_tool == "compute_vrp"
    assert "routing" in result.reasoning.lower()


def test_select_solver_scheduling():
    """Test scheduling solver selection."""
    result = select_solver(
        num_variables=200, num_constraints=150, is_scheduling=True
    )

    assert result.recommended_tool == "solve_job_shop"
    assert "CP-SAT" in result.reasoning
    assert result.expected_performance == "fast"


def test_select_solver_stochastic():
    """Test stochastic problem solver selection."""
    result = select_solver(
        num_variables=50, num_constraints=30, has_uncertainty=True
    )

    assert result.recommended_tool == "solve_two_stage_stochastic"
    assert "uncertainty" in result.reasoning.lower()
    assert result.expected_performance == "medium"
