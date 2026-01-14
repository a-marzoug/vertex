"""Tests for linear programming solver."""

from vertex.config import ConstraintSense, ObjectiveSense, SolverStatus
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


def test_simple_lp():
    """Test simple LP: maximize x + y subject to x + y <= 10, x,y >= 0."""
    problem = LPProblem(
        variables=[
            Variable(name="x", lower_bound=0, upper_bound=float("inf")),
            Variable(name="y", lower_bound=0, upper_bound=float("inf")),
        ],
        constraints=[
            Constraint(
                coefficients={"x": 1, "y": 1},
                sense=ConstraintSense.LEQ,
                rhs=10,
            )
        ],
        objective=Objective(
            coefficients={"x": 1, "y": 1},
            sense=ObjectiveSense.MAXIMIZE,
        ),
    )

    solver = LinearSolver()
    result = solver.solve(problem)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective_value is not None
    assert abs(result.objective_value - 10.0) < 0.01


def test_infeasible_lp():
    """Test infeasible LP."""
    problem = LPProblem(
        variables=[Variable(name="x", lower_bound=0, upper_bound=float("inf"))],
        constraints=[
            Constraint(coefficients={"x": 1}, sense=ConstraintSense.LEQ, rhs=-1)
        ],
        objective=Objective(coefficients={"x": 1}, sense=ObjectiveSense.MAXIMIZE),
    )

    solver = LinearSolver()
    result = solver.solve(problem)

    assert result.status == SolverStatus.INFEASIBLE
