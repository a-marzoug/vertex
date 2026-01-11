"""Supply chain network design optimization."""

from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field

from vertex.config import SolverStatus


class SupplyChainResult(BaseModel):
    """Result of supply chain network design."""

    status: SolverStatus
    open_facilities: list[str] = Field(default_factory=list)
    flows: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="facility -> customer -> flow",
    )
    total_cost: float | None = None
    fixed_cost: float | None = None
    transport_cost: float | None = None


def optimize_supply_chain(
    facilities: list[str],
    customers: list[str],
    fixed_costs: dict[str, float],
    capacities: dict[str, float],
    demands: dict[str, float],
    transport_costs: dict[str, dict[str, float]],
    max_facilities: int | None = None,
) -> SupplyChainResult:
    """
    Design supply chain network - decide which facilities to open and how to serve customers.

    Args:
        facilities: Potential facility locations.
        customers: Customer locations.
        fixed_costs: Fixed cost to open each facility.
        capacities: Capacity of each facility.
        demands: Demand at each customer.
        transport_costs: transport_costs[facility][customer] = unit cost.
        max_facilities: Optional limit on number of facilities to open.

    Returns:
        SupplyChainResult with open facilities and flows.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return SupplyChainResult(status=SolverStatus.ERROR)

    # Variables
    y = {f: solver.BoolVar(f"open_{f}") for f in facilities}
    x = {(f, c): solver.NumVar(0, demands[c], f"flow_{f}_{c}") for f in facilities for c in customers}

    # Meet all demands
    for c in customers:
        solver.Add(sum(x[(f, c)] for f in facilities) == demands[c])

    # Capacity constraints
    for f in facilities:
        solver.Add(sum(x[(f, c)] for c in customers) <= capacities[f] * y[f])

    # Max facilities constraint
    if max_facilities:
        solver.Add(sum(y[f] for f in facilities) <= max_facilities)

    # Minimize total cost
    fixed = sum(fixed_costs[f] * y[f] for f in facilities)
    transport = sum(transport_costs[f][c] * x[(f, c)] for f in facilities for c in customers)
    solver.Minimize(fixed + transport)

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return SupplyChainResult(status=SolverStatus.INFEASIBLE)

    open_fac = [f for f in facilities if y[f].solution_value() > 0.5]
    flows = {}
    for f in facilities:
        flows[f] = {}
        for c in customers:
            val = x[(f, c)].solution_value()
            if val > 0.001:
                flows[f][c] = round(val, 2)

    fixed_val = sum(fixed_costs[f] for f in open_fac)
    transport_val = sum(
        transport_costs[f][c] * x[(f, c)].solution_value()
        for f in facilities for c in customers
    )

    return SupplyChainResult(
        status=SolverStatus.OPTIMAL if status == pywraplp.Solver.OPTIMAL else SolverStatus.FEASIBLE,
        open_facilities=open_fac,
        flows={f: v for f, v in flows.items() if v},
        total_cost=round(fixed_val + transport_val, 2),
        fixed_cost=round(fixed_val, 2),
        transport_cost=round(transport_val, 2),
    )
