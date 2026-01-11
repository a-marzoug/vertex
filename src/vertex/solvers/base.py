"""Abstract solver interface."""

from typing import Protocol

from vertex.models.linear import LPProblem, LPSolution


class Solver(Protocol):
    """Protocol for optimization solvers."""

    def solve(self, problem: LPProblem) -> LPSolution:
        """Solve an optimization problem."""
        ...
