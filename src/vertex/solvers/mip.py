"""Mixed-Integer Programming solver using Google OR-Tools."""

from ortools.linear_solver import pywraplp

from vertex.config import (
    DEFAULT_TIMEOUT_MS,
    ConstraintSense,
    ObjectiveSense,
    SolverStatus,
    SolverType,
    VariableType,
)
from vertex.logging import (
    get_logger,
    log_solve_complete,
    log_solve_error,
    log_solve_start,
)
from vertex.models.mip import MIPProblem, MIPSolution

logger = get_logger(__name__)


class MIPSolver:
    """MIP solver wrapper around OR-Tools SCIP."""

    def __init__(
        self,
        solver_type: SolverType = SolverType.SCIP,
        time_limit_ms: int | None = None,
    ) -> None:
        self.solver_type = solver_type
        self.time_limit_ms = time_limit_ms or DEFAULT_TIMEOUT_MS

    def solve(self, problem: MIPProblem) -> MIPSolution:
        """Solve a mixed-integer programming problem."""
        log_solve_start(
            logger,
            tool_name="MIPSolver",
            num_variables=len(problem.variables),
            num_constraints=len(problem.constraints),
            solver_type=self.solver_type,
        )

        try:
            solver = pywraplp.Solver.CreateSolver(self.solver_type)
            if not solver:
                logger.error("solver_creation_failed", solver_type=self.solver_type)
                return MIPSolution(status=SolverStatus.ERROR)

            # Set time limit
            solver.SetTimeLimit(self.time_limit_ms)

            var_map: dict[str, pywraplp.Variable] = {}
            for var in problem.variables:
                match var.var_type:
                    case VariableType.BINARY:
                        var_map[var.name] = solver.BoolVar(var.name)
                    case VariableType.INTEGER:
                        var_map[var.name] = solver.IntVar(
                            int(var.lower_bound),
                            int(var.upper_bound)
                            if var.upper_bound != float("inf")
                            else solver.infinity(),
                            var.name,
                        )
                    case VariableType.CONTINUOUS:
                        var_map[var.name] = solver.NumVar(
                            var.lower_bound, var.upper_bound, var.name
                        )

            for constraint in problem.constraints:
                expr = sum(
                    coef * var_map[name]
                    for name, coef in constraint.coefficients.items()
                )
                match constraint.sense:
                    case ConstraintSense.LEQ:
                        solver.Add(expr <= constraint.rhs)
                    case ConstraintSense.GEQ:
                        solver.Add(expr >= constraint.rhs)
                    case ConstraintSense.EQ:
                        solver.Add(expr == constraint.rhs)

            obj_expr = sum(
                coef * var_map[name]
                for name, coef in problem.objective.coefficients.items()
            )
            if problem.objective.sense == ObjectiveSense.MAXIMIZE:
                solver.Maximize(obj_expr)
            else:
                solver.Minimize(obj_expr)

            status = solver.Solve()

            status_map = {
                pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
                pywraplp.Solver.FEASIBLE: SolverStatus.FEASIBLE,
                pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
                pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
            }
            result_status = status_map.get(status, SolverStatus.ERROR)

            if result_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
                solution = MIPSolution(
                    status=result_status,
                    objective_value=round(solver.Objective().Value(), 6),
                    variable_values={
                        name: round(var.solution_value(), 6)
                        for name, var in var_map.items()
                    },
                    solve_time_ms=solver.wall_time(),
                )
                log_solve_complete(
                    logger,
                    tool_name="MIPSolver",
                    status=result_status,
                    solve_time_ms=solution.solve_time_ms,
                    objective_value=solution.objective_value,
                )
                return solution

            log_solve_complete(
                logger,
                tool_name="MIPSolver",
                status=result_status,
                solve_time_ms=solver.wall_time(),
            )
            return MIPSolution(status=result_status)

        except Exception as e:
            log_solve_error(logger, tool_name="MIPSolver", error=e)
            return MIPSolution(status=SolverStatus.ERROR)
