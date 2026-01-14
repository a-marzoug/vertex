"""Tests for MIP solver."""

from vertex.config import ConstraintSense, ObjectiveSense, SolverStatus, VariableType
from vertex.models.mip import MIPConstraint, MIPObjective, MIPProblem, MIPVariable
from vertex.solvers.mip import MIPSolver


def test_simple_mip():
    """Test simple MIP: maximize x + 2y subject to x + y <= 5, x,y integer."""
    problem = MIPProblem(
        variables=[
            MIPVariable(name="x", var_type=VariableType.INTEGER, lower_bound=0),
            MIPVariable(name="y", var_type=VariableType.INTEGER, lower_bound=0),
        ],
        constraints=[
            MIPConstraint(
                coefficients={"x": 1, "y": 1},
                sense=ConstraintSense.LEQ,
                rhs=5,
            )
        ],
        objective=MIPObjective(
            coefficients={"x": 1, "y": 2},
            sense=ObjectiveSense.MAXIMIZE,
        ),
    )

    solver = MIPSolver()
    result = solver.solve(problem)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective_value is not None
    assert abs(result.objective_value - 10.0) < 0.01


def test_binary_mip():
    """Test binary MIP."""
    problem = MIPProblem(
        variables=[
            MIPVariable(name="x", var_type=VariableType.BINARY),
            MIPVariable(name="y", var_type=VariableType.BINARY),
        ],
        constraints=[
            MIPConstraint(
                coefficients={"x": 1, "y": 1},
                sense=ConstraintSense.LEQ,
                rhs=1,
            )
        ],
        objective=MIPObjective(
            coefficients={"x": 3, "y": 2},
            sense=ObjectiveSense.MAXIMIZE,
        ),
    )

    solver = MIPSolver()
    result = solver.solve(problem)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective_value == 3.0
