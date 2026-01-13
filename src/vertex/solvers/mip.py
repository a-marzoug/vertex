"""Mixed-Integer Programming solver using Google OR-Tools."""

from ortools.linear_solver import pywraplp

from vertex.config import (
    ConstraintSense,
    ObjectiveSense,
    SolverStatus,
    SolverType,
    VariableType,
)
from vertex.models.mip import MIPProblem, MIPSolution


class MIPSolver:
    """MIP solver wrapper around OR-Tools SCIP."""

    def __init__(self, solver_type: SolverType = SolverType.SCIP) -> None:
        self.solver_type = solver_type

    def solve(self, problem: MIPProblem) -> MIPSolution:
        """Solve a mixed-integer programming problem."""
        solver = pywraplp.Solver.CreateSolver(self.solver_type)
        if not solver:
            return MIPSolution(status=SolverStatus.ERROR)

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
                coef * var_map[name] for name, coef in constraint.coefficients.items()
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
            return MIPSolution(
                status=result_status,
                objective_value=round(solver.Objective().Value(), 6),
                variable_values={
                    name: round(var.solution_value(), 6)
                    for name, var in var_map.items()
                },
                solve_time_ms=solver.wall_time(),
            )

        return MIPSolution(status=result_status)
