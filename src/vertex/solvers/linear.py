"""Linear Programming solver using Google OR-Tools."""

from ortools.linear_solver import pywraplp

from vertex.config import (
    DEFAULT_TIMEOUT_MS,
    ConstraintSense,
    ObjectiveSense,
    SolverStatus,
    SolverType,
)
from vertex.logging import (
    get_logger,
    log_solve_complete,
    log_solve_error,
    log_solve_start,
)
from vertex.models.linear import LPProblem, LPSolution

logger = get_logger(__name__)


class LinearSolver:
    """LP solver wrapper around OR-Tools GLOP."""

    def __init__(
        self,
        solver_type: SolverType = SolverType.GLOP,
        time_limit_ms: int | None = None,
    ) -> None:
        self.solver_type = solver_type
        self.time_limit_ms = time_limit_ms or DEFAULT_TIMEOUT_MS

    def solve(self, problem: LPProblem) -> LPSolution:
        """Solve a linear programming problem."""
        log_solve_start(
            logger,
            tool_name="LinearSolver",
            num_variables=len(problem.variables),
            num_constraints=len(problem.constraints),
            solver_type=self.solver_type,
        )

        try:
            solver = pywraplp.Solver.CreateSolver(self.solver_type)
            if not solver:
                logger.error("solver_creation_failed", solver_type=self.solver_type)
                return LPSolution(status=SolverStatus.ERROR)

            # Set time limit
            solver.SetTimeLimit(self.time_limit_ms)

            # Create variables
            var_map: dict[str, pywraplp.Variable] = {}
            for var in problem.variables:
                var_map[var.name] = solver.NumVar(
                    var.lower_bound,
                    var.upper_bound,
                    var.name,
                )

            # Add constraints (keep references for dual values)
            constraint_map: dict[str, pywraplp.Constraint] = {}
            for i, constraint in enumerate(problem.constraints):
                expr = sum(
                    coef * var_map[name]
                    for name, coef in constraint.coefficients.items()
                )
                name = constraint.name or f"c{i}"
                match constraint.sense:
                    case ConstraintSense.LEQ:
                        ct = solver.Add(expr <= constraint.rhs)
                    case ConstraintSense.GEQ:
                        ct = solver.Add(expr >= constraint.rhs)
                    case ConstraintSense.EQ:
                        ct = solver.Add(expr == constraint.rhs)
                constraint_map[name] = ct

            # Set objective
            obj_expr = sum(
                coef * var_map[name]
                for name, coef in problem.objective.coefficients.items()
            )
            if problem.objective.sense == ObjectiveSense.MAXIMIZE:
                solver.Maximize(obj_expr)
            else:
                solver.Minimize(obj_expr)

            # Solve
            status = solver.Solve()

            # Map status
            status_map = {
                pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
                pywraplp.Solver.FEASIBLE: SolverStatus.FEASIBLE,
                pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
                pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
            }
            result_status = status_map.get(status, SolverStatus.ERROR)

            # Build solution
            if result_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
                solution = LPSolution(
                    status=result_status,
                    objective_value=round(solver.Objective().Value(), 6),
                    variable_values={
                        name: round(var.solution_value(), 6)
                        for name, var in var_map.items()
                    },
                    shadow_prices={
                        name: round(ct.dual_value(), 6)
                        for name, ct in constraint_map.items()
                    },
                    reduced_costs={
                        name: round(var.reduced_cost(), 6)
                        for name, var in var_map.items()
                    },
                    solve_time_ms=solver.wall_time(),
                    iterations=solver.iterations(),
                )
                log_solve_complete(
                    logger,
                    tool_name="LinearSolver",
                    status=result_status,
                    solve_time_ms=solution.solve_time_ms,
                    objective_value=solution.objective_value,
                )
                return solution

            log_solve_complete(
                logger,
                tool_name="LinearSolver",
                status=result_status,
                solve_time_ms=solver.wall_time(),
            )
            return LPSolution(status=result_status)

        except Exception as e:
            log_solve_error(logger, tool_name="LinearSolver", error=e)
            return LPSolution(status=SolverStatus.ERROR)
