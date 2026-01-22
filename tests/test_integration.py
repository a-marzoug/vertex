"""Integration tests for the full tool stack."""

import pytest

from vertex.metrics import SOLVE_REQUESTS
from vertex.tools.linear import solve_lp
from vertex.validation import ErrorCode, ValidationError


@pytest.mark.asyncio
async def test_solve_lp_stack_integration():
    """Test the full LP stack: Validation -> Metrics -> Async -> Solver."""

    # 1. Valid Request
    variables = [{"name": "x", "lower_bound": 0}, {"name": "y", "lower_bound": 0}]
    constraints = [{"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}]
    objective = {"x": 1, "y": 1}

    # Should run async and succeed
    result = await solve_lp(
        variables=variables,
        constraints=constraints,
        objective_coefficients=objective,
        objective_sense="maximize",
    )

    assert result.status == "optimal"
    assert result.objective_value == 10.0

    # Verify Metrics
    # Note: REGISTRY is global, so we check if the counter incremented
    # We might need to check the specific sample value
    before = SOLVE_REQUESTS.labels(
        tool="solve_linear_program", status="optimal"
    )._value.get()

    # Run again to increment
    await solve_lp(
        variables=variables,
        constraints=constraints,
        objective_coefficients=objective,
        objective_sense="maximize",
    )

    after = SOLVE_REQUESTS.labels(
        tool="solve_linear_program", status="optimal"
    )._value.get()
    assert after > before


@pytest.mark.asyncio
async def test_solve_lp_validation_integration():
    """Test validation layer in the stack."""

    # Create invalid request (too many variables)
    # Mocking config limit would be cleaner, but we can just use a large number if limits are defaults
    # Or rely on the decorator working. Let's rely on the validation error.

    # Let's pass a negative timeout which triggers validation
    variables = [{"name": "x"}]
    constraints = []
    objective = {"x": 1}

    with pytest.raises(ValidationError) as exc:
        await solve_lp(
            variables=variables,
            constraints=constraints,
            objective_coefficients=objective,
            time_limit_ms=-1,  # Invalid
        )

    assert exc.value.code == ErrorCode.VALIDATION_ERROR

    # Metrics should record exception/error?
    # The validation decorator runs *before* the metrics decorator if stacked order is:
    # @track_solve_metrics
    # @validate_...
    # So metrics might catch the ValidationError if it propagates up.
    # Let's check the stack order in tools/linear.py:
    # @track_solve_metrics
    # @validate_problem_size
    # @validate_timeout

    # So validate raises exception, track_metrics catches it (or sees it pass through) and records 'exception' status?
    # Our track_solve_metrics re-raises exceptions but counts them as "exception" status.

    # Check "exception" status count
    try:
        current_errors = SOLVE_REQUESTS.labels(
            tool="solve_linear_program", status="exception"
        )._value.get()
    except Exception:
        current_errors = 0

    # Trigger error again
    with pytest.raises(ValidationError):
        await solve_lp(
            variables=variables,
            constraints=constraints,
            objective_coefficients=objective,
            time_limit_ms=-1,
        )

    new_errors = SOLVE_REQUESTS.labels(
        tool="solve_linear_program", status="exception"
    )._value.get()
    assert new_errors > current_errors
