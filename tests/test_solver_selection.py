"""Test solver selection capability."""

import pytest

from vertex.config import SolverType
from vertex.tools.linear import solve_lp
from vertex.tools.mip import solve_mip


@pytest.mark.asyncio
async def test_solve_lp_with_glop():
    """Test LP with explicit GLOP solver."""
    result = await solve_lp(
        variables=[{"name": "x"}, {"name": "y"}],
        constraints=[{"coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}],
        objective_coefficients={"x": 1, "y": 1},
        solver_type=SolverType.GLOP,
    )
    assert result.status == "optimal"
    assert result.objective_value == 10.0


@pytest.mark.asyncio
async def test_solve_mip_with_scip():
    """Test MIP with explicit SCIP solver."""
    result = await solve_mip(
        variables=[
            {"name": "x", "var_type": "integer"},
            {"name": "y", "var_type": "integer"},
        ],
        constraints=[{"coefficients": {"x": 2, "y": 2}, "sense": "<=", "rhs": 9}],
        objective_coefficients={"x": 1, "y": 1},
        solver_type=SolverType.SCIP,
    )
    assert result.status == "optimal"
    # Max integer x+y s.t. 2(x+y) <= 9 => x+y <= 4.5. Integers => x+y <= 4.
    assert result.objective_value == 4.0


@pytest.mark.asyncio
async def test_solve_mip_with_sat():
    """Test MIP with CP-SAT solver."""
    result = await solve_mip(
        variables=[
            {"name": "x", "var_type": "integer"},
            {"name": "y", "var_type": "integer"},
        ],
        constraints=[{"coefficients": {"x": 2, "y": 2}, "sense": "<=", "rhs": 9}],
        objective_coefficients={"x": 1, "y": 1},
        solver_type=SolverType.SAT,
    )
    assert result.status == "optimal"
    assert result.objective_value == 4.0
