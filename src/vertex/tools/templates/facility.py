"""Facility location optimization template."""

from pydantic import BaseModel, Field

from vertex.config import ConstraintSense, ObjectiveSense, VariableType
from vertex.models.mip import MIPConstraint, MIPObjective, MIPProblem, MIPVariable
from vertex.solvers.mip import MIPSolver


class FacilityResult(BaseModel):
    """Result of facility location optimization."""

    status: str
    total_cost: float | None = None
    open_facilities: list[str] = Field(default_factory=list)
    assignments: dict[str, str] = Field(default_factory=dict)
    solve_time_ms: float | None = None


def optimize_facility_location(
    facilities: list[str],
    customers: list[str],
    fixed_costs: dict[str, float],
    transport_costs: dict[str, dict[str, float]],
) -> FacilityResult:
    """
    Solve facility location: decide which facilities to open and assign customers.

    Args:
        facilities: List of potential facility locations. Example: ["NYC", "LA", "Chicago"]
        customers: List of customer locations. Example: ["Boston", "Seattle", "Miami"]
        fixed_costs: Cost to open each facility. Example: {"NYC": 1000, "LA": 1200, "Chicago": 800}
        transport_costs: Cost to serve customer from facility.
            Example: {"NYC": {"Boston": 50, "Seattle": 200}, ...}

    Returns:
        FacilityResult with open facilities and customer assignments.
    """
    variables = []
    # Binary: open facility or not
    for f in facilities:
        variables.append(MIPVariable(name=f"open_{f}", var_type=VariableType.BINARY))
    # Binary: assign customer to facility
    for f in facilities:
        for c in customers:
            variables.append(
                MIPVariable(name=f"assign_{f}_{c}", var_type=VariableType.BINARY)
            )

    constraints = []
    # Each customer assigned to exactly one facility
    for c in customers:
        constraints.append(
            MIPConstraint(
                coefficients={f"assign_{f}_{c}": 1 for f in facilities},
                sense=ConstraintSense.EQ,
                rhs=1,
            )
        )
    # Can only assign to open facilities
    for f in facilities:
        for c in customers:
            constraints.append(
                MIPConstraint(
                    coefficients={f"assign_{f}_{c}": 1, f"open_{f}": -1},
                    sense=ConstraintSense.LEQ,
                    rhs=0,
                )
            )

    # Objective: minimize fixed + transport costs
    obj_coefficients = {f"open_{f}": fixed_costs[f] for f in facilities}
    for f in facilities:
        for c in customers:
            obj_coefficients[f"assign_{f}_{c}"] = transport_costs[f][c]

    problem = MIPProblem(
        variables=variables,
        constraints=constraints,
        objective=MIPObjective(
            coefficients=obj_coefficients, sense=ObjectiveSense.MINIMIZE
        ),
    )

    solution = MIPSolver().solve(problem)

    open_facilities = []
    assignments = {}
    if solution.variable_values:
        for f in facilities:
            if solution.variable_values.get(f"open_{f}", 0) > 0.5:
                open_facilities.append(f)
        for c in customers:
            for f in facilities:
                if solution.variable_values.get(f"assign_{f}_{c}", 0) > 0.5:
                    assignments[c] = f

    return FacilityResult(
        status=solution.status,
        total_cost=solution.objective_value,
        open_facilities=open_facilities,
        assignments=assignments,
        solve_time_ms=solution.solve_time_ms,
    )
