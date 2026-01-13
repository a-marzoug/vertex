"""Healthcare resource allocation optimization."""

from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class ResourceAllocationResult(BaseModel):
    """Result of resource allocation optimization."""

    status: SolverStatus
    allocations: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="resource -> location -> amount",
    )
    total_coverage: float | None = None
    unmet_demand: dict[str, float] = Field(default_factory=dict)


def optimize_resource_allocation(
    resources: list[str],
    locations: list[str],
    availability: dict[str, float],
    demands: dict[str, dict[str, float]],
    effectiveness: dict[str, dict[str, float]] | None = None,
    min_coverage: dict[str, float] | None = None,
) -> ResourceAllocationResult:
    """
    Allocate limited resources across locations to maximize coverage.

    Args:
        resources: Resource types (e.g., ["doctors", "nurses", "beds"]).
        locations: Location names (e.g., ["hospital_A", "clinic_B"]).
        availability: Total available units per resource.
        demands: demands[location][resource] = units needed.
        effectiveness: Optional effectiveness[location][resource] multiplier.
        min_coverage: Optional minimum coverage ratio per location.

    Returns:
        ResourceAllocationResult with optimal allocations.
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return ResourceAllocationResult(status=SolverStatus.ERROR)

    effectiveness = effectiveness or {
        loc: {r: 1.0 for r in resources} for loc in locations
    }

    # Variables: x[r][l] = amount of resource r allocated to location l
    x = {}
    for r in resources:
        for loc in locations:
            x[(r, loc)] = solver.NumVar(0, availability[r], f"x_{r}_{loc}")

    # Resource availability constraints
    for r in resources:
        solver.Add(sum(x[(r, loc)] for loc in locations) <= availability[r])

    # Minimum coverage constraints if specified
    if min_coverage:
        for loc in locations:
            for r in resources:
                if demands.get(loc, {}).get(r, 0) > 0:
                    solver.Add(
                        x[(r, loc)] >= min_coverage.get(loc, 0) * demands[loc][r]
                    )

    # Maximize weighted coverage
    solver.Maximize(
        sum(
            effectiveness.get(loc, {}).get(r, 1.0) * x[(r, loc)]
            for r in resources
            for loc in locations
        )
    )

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return ResourceAllocationResult(status=SolverStatus.INFEASIBLE)

    allocations = {}
    unmet = {}
    for r in resources:
        allocations[r] = {}
        for loc in locations:
            val = x[(r, loc)].solution_value()
            if val > 0.001:
                allocations[r][loc] = round(val, 2)
            demand = demands.get(loc, {}).get(r, 0)
            if demand > val + 0.001:
                unmet[f"{loc}_{r}"] = round(demand - val, 2)

    return ResourceAllocationResult(
        status=SolverStatus.OPTIMAL
        if status == pywraplp.Solver.OPTIMAL
        else SolverStatus.FEASIBLE,
        allocations=allocations,
        total_coverage=round(solver.Objective().Value(), 2),
        unmet_demand=unmet,
    )
