"""Analysis tools for optimization problems."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, SolverStatus
from vertex.models.linear import Constraint, LPProblem, Objective, Variable
from vertex.solvers.linear import LinearSolver


class WhatIfResult(BaseModel):
    """Result of what-if analysis."""

    parameter_name: str
    original_value: float
    results: list[dict] = Field(
        default_factory=list, description="List of {value, objective, feasible}"
    )


class InfeasibilityResult(BaseModel):
    """Result of infeasibility diagnosis."""

    status: str
    is_feasible: bool
    conflicting_constraints: list[str] = Field(default_factory=list)
    relaxation_suggestions: dict[str, float] = Field(default_factory=dict)


def analyze_what_if(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    parameter_name: str,
    parameter_values: list[float],
    objective_sense: str = "maximize",
) -> WhatIfResult:
    """
    Perform what-if analysis by varying a constraint RHS value.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions (one must have name=parameter_name).
        objective_coefficients: Objective function coefficients.
        parameter_name: Name of constraint to vary.
        parameter_values: Values to test for the constraint RHS.
        objective_sense: "maximize" or "minimize".

    Returns:
        WhatIfResult with objective value at each parameter value.
    """
    results = []
    original_value = None

    for pval in parameter_values:
        # Build modified constraints
        modified_constraints = []
        for c in constraints:
            if c.get("name") == parameter_name:
                if original_value is None:
                    original_value = c["rhs"]
                modified_constraints.append({**c, "rhs": pval})
            else:
                modified_constraints.append(c)

        # Build and solve problem
        vars_ = [Variable(**v) for v in variables]
        constrs = [
            Constraint(
                coefficients=c["coefficients"],
                sense=ConstraintSense(c["sense"]),
                rhs=c["rhs"],
                name=c.get("name"),
            )
            for c in modified_constraints
        ]
        obj = Objective(
            coefficients=objective_coefficients, sense=ObjectiveSense(objective_sense)
        )
        problem = LPProblem(variables=vars_, constraints=constrs, objective=obj)

        solution = LinearSolver().solve(problem)

        results.append(
            {
                "value": pval,
                "objective": solution.objective_value,
                "feasible": solution.status
                in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE),
            }
        )

    return WhatIfResult(
        parameter_name=parameter_name,
        original_value=original_value or parameter_values[0],
        results=results,
    )


def diagnose_infeasibility(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
) -> InfeasibilityResult:
    """
    Diagnose why a problem is infeasible by finding conflicting constraints.

    Uses constraint relaxation to identify which constraints cause infeasibility.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.
        objective_coefficients: Objective function coefficients.
        objective_sense: "maximize" or "minimize".

    Returns:
        InfeasibilityResult with conflicting constraints and relaxation suggestions.
    """
    # First check if actually infeasible
    vars_ = [Variable(**v) for v in variables]
    constrs = [
        Constraint(
            coefficients=c["coefficients"],
            sense=ConstraintSense(c["sense"]),
            rhs=c["rhs"],
            name=c.get("name"),
        )
        for c in constraints
    ]
    obj = Objective(
        coefficients=objective_coefficients, sense=ObjectiveSense(objective_sense)
    )
    problem = LPProblem(variables=vars_, constraints=constrs, objective=obj)

    solution = LinearSolver().solve(problem)

    if solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        return InfeasibilityResult(status="feasible", is_feasible=True)

    # Find conflicting constraints by removing one at a time
    conflicting = []
    relaxations = {}

    for i, c in enumerate(constraints):
        # Try without this constraint
        reduced_constraints = constraints[:i] + constraints[i + 1 :]
        if not reduced_constraints:
            continue

        reduced_constrs = [
            Constraint(
                coefficients=rc["coefficients"],
                sense=ConstraintSense(rc["sense"]),
                rhs=rc["rhs"],
                name=rc.get("name"),
            )
            for rc in reduced_constraints
        ]
        reduced_problem = LPProblem(
            variables=vars_, constraints=reduced_constrs, objective=obj
        )
        reduced_solution = LinearSolver().solve(reduced_problem)

        if reduced_solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            name = c.get("name", f"constraint_{i}")
            conflicting.append(name)

            # Calculate how much to relax
            if reduced_solution.variable_values:
                lhs = sum(
                    coef * reduced_solution.variable_values.get(var, 0)
                    for var, coef in c["coefficients"].items()
                )
                violation = lhs - c["rhs"] if c["sense"] == "<=" else c["rhs"] - lhs
                if violation > 0:
                    relaxations[name] = round(violation, 4)

    return InfeasibilityResult(
        status="infeasible",
        is_feasible=False,
        conflicting_constraints=conflicting,
        relaxation_suggestions=relaxations,
    )


def solve_rcpsp(
    tasks: list[dict],
    resources: dict[str, int],
    time_limit_seconds: int = 30,
) -> dict:
    """
    Solve Resource-Constrained Project Scheduling Problem.

    Args:
        tasks: List of tasks with 'name', 'duration', 'resources' (dict), 'predecessors' (list).
        resources: Available capacity per resource type.
        time_limit_seconds: Solver time limit.

    Returns:
        Dict with status, makespan, and schedule.
    """
    import time

    from ortools.sat.python import cp_model

    start_time = time.time()
    model = cp_model.CpModel()

    horizon = sum(t["duration"] for t in tasks)
    task_idx = {t["name"]: i for i, t in enumerate(tasks)}

    # Variables
    starts = {}
    ends = {}
    intervals = {}

    for t in tasks:
        name = t["name"]
        starts[name] = model.new_int_var(0, horizon, f"start_{name}")
        ends[name] = model.new_int_var(0, horizon, f"end_{name}")
        intervals[name] = model.new_interval_var(
            starts[name], t["duration"], ends[name], f"interval_{name}"
        )

    # Precedence constraints
    for t in tasks:
        for pred in t.get("predecessors", []):
            model.add(starts[t["name"]] >= ends[pred])

    # Resource constraints using cumulative
    for res_name, capacity in resources.items():
        task_intervals = []
        demands = []
        for t in tasks:
            if t.get("resources", {}).get(res_name, 0) > 0:
                task_intervals.append(intervals[t["name"]])
                demands.append(t["resources"][res_name])
        if task_intervals:
            model.add_cumulative(task_intervals, demands, capacity)

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[t["name"]] for t in tasks])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "solve_time_ms": elapsed}

    schedule = [
        {
            "task": t["name"],
            "start": solver.value(starts[t["name"]]),
            "end": solver.value(ends[t["name"]]),
        }
        for t in tasks
    ]

    return {
        "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
        "makespan": solver.value(makespan),
        "schedule": sorted(schedule, key=lambda x: x["start"]),
        "solve_time_ms": elapsed,
    }


class ModelStats(BaseModel):
    """Statistics about an optimization model."""

    num_variables: int
    num_constraints: int
    num_nonzeros: int
    density: float = Field(description="Fraction of non-zero coefficients")
    variable_types: dict[str, int] = Field(default_factory=dict)
    constraint_types: dict[str, int] = Field(default_factory=dict)


def get_model_stats(
    variables: list[dict],
    constraints: list[dict],
) -> ModelStats:
    """
    Get statistics about an optimization model.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.

    Returns:
        ModelStats with size and sparsity information.
    """
    num_vars = len(variables)
    num_constrs = len(constraints)

    # Count non-zeros
    nonzeros = sum(len(c.get("coefficients", {})) for c in constraints)

    # Density = nonzeros / (vars * constraints)
    max_nonzeros = num_vars * num_constrs if num_constrs > 0 else 1
    density = nonzeros / max_nonzeros if max_nonzeros > 0 else 0

    # Variable types
    var_types = {}
    for v in variables:
        vtype = v.get("var_type", "continuous")
        var_types[vtype] = var_types.get(vtype, 0) + 1

    # Constraint types
    constr_types = {}
    for c in constraints:
        sense = c.get("sense", "<=")
        constr_types[sense] = constr_types.get(sense, 0) + 1

    return ModelStats(
        num_variables=num_vars,
        num_constraints=num_constrs,
        num_nonzeros=nonzeros,
        density=round(density, 4),
        variable_types=var_types,
        constraint_types=constr_types,
    )


def find_alternative_solutions(
    variables: list[dict],
    constraints: list[dict],
    objective_coefficients: dict[str, float],
    objective_sense: str = "maximize",
    max_solutions: int = 5,
    gap_tolerance: float = 0.01,
) -> list[dict]:
    """
    Find multiple near-optimal solutions.

    Args:
        variables: Variable definitions.
        constraints: Constraint definitions.
        objective_coefficients: Objective function coefficients.
        objective_sense: "maximize" or "minimize".
        max_solutions: Maximum solutions to return.
        gap_tolerance: Accept solutions within this fraction of optimal.

    Returns:
        List of solutions with objective values and variable values.
    """
    from ortools.linear_solver import pywraplp

    solutions = []

    # First solve to get optimal
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return solutions

    var_map = {}
    for v in variables:
        lb = v.get("lower_bound", 0)
        ub = v.get("upper_bound", float("inf"))
        vtype = v.get("var_type", "continuous")
        if vtype == "binary":
            var_map[v["name"]] = solver.BoolVar(v["name"])
        elif vtype == "integer":
            var_map[v["name"]] = solver.IntVar(int(lb), int(ub), v["name"])
        else:
            var_map[v["name"]] = solver.NumVar(lb, ub, v["name"])

    for c in constraints:
        expr = sum(coef * var_map[name] for name, coef in c["coefficients"].items())
        sense = c.get("sense", "<=")
        if sense == "<=":
            solver.Add(expr <= c["rhs"])
        elif sense == ">=":
            solver.Add(expr >= c["rhs"])
        else:
            solver.Add(expr == c["rhs"])

    obj_expr = sum(
        coef * var_map[name] for name, coef in objective_coefficients.items()
    )
    if objective_sense == "maximize":
        solver.Maximize(obj_expr)
    else:
        solver.Minimize(obj_expr)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return solutions

    optimal_obj = solver.Objective().Value()
    solutions.append(
        {
            "objective": round(optimal_obj, 6),
            "variables": {
                name: round(var.solution_value(), 6) for name, var in var_map.items()
            },
            "is_optimal": True,
        }
    )

    # Add constraint to exclude current solution and find alternatives
    for _ in range(max_solutions - 1):
        # Add cut to exclude previous solutions (for integer/binary vars)
        int_vars = [name for name, var in var_map.items() if var.integer()]
        if not int_vars:
            break

        # Simple exclusion: at least one variable must differ
        prev_sol = solutions[-1]["variables"]
        cut_expr = sum(
            var_map[name] if prev_sol[name] < 0.5 else (1 - var_map[name])
            for name in int_vars
            if name in prev_sol
        )
        solver.Add(cut_expr >= 1)

        # Add optimality gap constraint
        if objective_sense == "maximize":
            solver.Add(obj_expr >= optimal_obj * (1 - gap_tolerance))
        else:
            solver.Add(obj_expr <= optimal_obj * (1 + gap_tolerance))

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            break

        solutions.append(
            {
                "objective": round(solver.Objective().Value(), 6),
                "variables": {
                    name: round(var.solution_value(), 6)
                    for name, var in var_map.items()
                },
                "is_optimal": False,
            }
        )

    return solutions
