"""Multi-objective optimization tools."""

from typing import Any

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, SolverStatus
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class ParetoPoint(BaseModel):
    """A point on the Pareto frontier."""

    objectives: dict[str, float]
    variables: dict[str, float]


class MultiObjectiveResult(BaseModel):
    """Result of multi-objective optimization."""

    status: SolverStatus
    pareto_points: list[ParetoPoint] = Field(default_factory=list)
    num_points: int = 0


def solve_multi_objective(
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    objectives: dict[str, dict[str, float]],
    num_points: int = 10,
    objective_senses: dict[str, str] | None = None,
) -> MultiObjectiveResult:
    """
    Solve multi-objective optimization using weighted sum method.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.
        objectives: Dict of objective_name -> {var: coef} mappings.
        num_points: Number of Pareto points to generate.
        objective_senses: Dict of objective_name -> "maximize"/"minimize".

    Returns:
        MultiObjectiveResult with Pareto frontier points.
    """
    obj_names = list(objectives.keys())
    if len(obj_names) != 2:
        return MultiObjectiveResult(status=SolverStatus.ERROR)

    senses = objective_senses or {name: "maximize" for name in obj_names}
    pareto_points: list[ParetoPoint] = []

    # Generate weights for weighted sum
    for i in range(num_points + 1):
        w1 = i / num_points
        w2 = 1 - w1

        # Combine objectives
        combined = {}
        for var in set(
            list(objectives[obj_names[0]].keys())
            + list(objectives[obj_names[1]].keys())
        ):
            c1 = objectives[obj_names[0]].get(var, 0)
            c2 = objectives[obj_names[1]].get(var, 0)
            # Flip sign for minimization objectives
            if senses.get(obj_names[0]) == "minimize":
                c1 = -c1
            if senses.get(obj_names[1]) == "minimize":
                c2 = -c2
            combined[var] = w1 * c1 + w2 * c2

        # Build and solve
        vars_ = [Variable(**v) for v in variables]
        constrs = [
            Constraint(
                coefficients=c["coefficients"],
                sense=ConstraintSense(c["sense"]),
                rhs=c["rhs"],
            )
            for c in constraints
        ]
        obj = Objective(coefficients=combined, sense=ObjectiveSense.MAXIMIZE)
        problem = LPProblem(variables=vars_, constraints=constrs, objective=obj)

        solution = LinearSolver().solve(problem)

        if solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            # Calculate actual objective values
            obj_vals = {}
            for name, coeffs in objectives.items():
                val = sum(
                    coeffs.get(v, 0) * solution.variable_values.get(v, 0)
                    for v in solution.variable_values
                )
                obj_vals[name] = round(val, 4)

            # Check if this is a new Pareto point
            is_new = True
            for existing in pareto_points:
                if all(
                    abs(existing.objectives[n] - obj_vals[n]) < 0.001 for n in obj_names
                ):
                    is_new = False
                    break

            if is_new:
                pareto_points.append(
                    ParetoPoint(
                        objectives=obj_vals,
                        variables={
                            k: round(v, 4) for k, v in solution.variable_values.items()
                        },
                    )
                )

    if not pareto_points:
        return MultiObjectiveResult(status=SolverStatus.INFEASIBLE)

    return MultiObjectiveResult(
        status=SolverStatus.OPTIMAL,
        pareto_points=pareto_points,
        num_points=len(pareto_points),
    )
