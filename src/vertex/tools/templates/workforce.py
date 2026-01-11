"""Workforce scheduling optimization template."""

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class ShiftAssignment(BaseModel):
    """Worker shift assignment."""

    worker: str
    day: int
    shift: str


class WorkforceResult(BaseModel):
    """Result of workforce scheduling."""

    status: SolverStatus
    assignments: list[ShiftAssignment] = Field(default_factory=list)
    total_cost: float | None = None
    workers_used: int | None = None
    solve_time_ms: float | None = None


def optimize_workforce_schedule(
    workers: list[str],
    days: int,
    shifts: list[str],
    requirements: dict[str, list[int]],
    costs: dict[str, float] | None = None,
    max_shifts_per_worker: int | None = None,
    time_limit_seconds: int = 30,
) -> WorkforceResult:
    """
    Schedule workers to shifts minimizing cost while meeting requirements.

    Args:
        workers: Worker names.
        days: Number of days to schedule.
        shifts: Shift names (e.g., ["morning", "afternoon", "night"]).
        requirements: Shift -> list of required workers per day.
            Example: {"morning": [3, 3, 2, 2, 3, 4, 4]} for 7 days.
        costs: Cost per worker (defaults to 1 for all).
        max_shifts_per_worker: Maximum shifts per worker over the period.
        time_limit_seconds: Solver time limit.

    Returns:
        WorkforceResult with shift assignments.
    """
    import time

    start_time = time.time()
    model = cp_model.CpModel()

    costs = costs or {w: 1 for w in workers}

    # x[w, d, s] = 1 if worker w works day d shift s
    x = {}
    for w in workers:
        for d in range(days):
            for s in shifts:
                x[(w, d, s)] = model.new_bool_var(f"x_{w}_{d}_{s}")

    # Meet requirements for each shift each day
    for s in shifts:
        reqs = requirements.get(s, [0] * days)
        for d in range(days):
            model.add(sum(x[(w, d, s)] for w in workers) >= reqs[d])

    # Each worker works at most one shift per day
    for w in workers:
        for d in range(days):
            model.add(sum(x[(w, d, s)] for s in shifts) <= 1)

    # Max shifts per worker
    if max_shifts_per_worker:
        for w in workers:
            model.add(sum(x[(w, d, s)] for d in range(days) for s in shifts) <= max_shifts_per_worker)

    # Minimize cost
    model.minimize(sum(
        int(costs[w] * 1000) * x[(w, d, s)]
        for w in workers for d in range(days) for s in shifts
    ))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = (time.time() - start_time) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return WorkforceResult(status=SolverStatus.INFEASIBLE, solve_time_ms=elapsed)

    assignments = []
    workers_used = set()
    for w in workers:
        for d in range(days):
            for s in shifts:
                if solver.value(x[(w, d, s)]):
                    assignments.append(ShiftAssignment(worker=w, day=d, shift=s))
                    workers_used.add(w)

    total_cost = sum(costs[a.worker] for a in assignments)

    return WorkforceResult(
        status=SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
        assignments=sorted(assignments, key=lambda a: (a.day, a.shift, a.worker)),
        total_cost=total_cost,
        workers_used=len(workers_used),
        solve_time_ms=elapsed,
    )
