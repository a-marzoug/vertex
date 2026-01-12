"""Solvers for stochastic and dynamic optimization problems."""

import math
from ortools.linear_solver import pywraplp
from vertex.models.stochastic import (
    LotSizingResult,
    NewsvendorResult,
    Scenario,
    TwoStageResult,
)


def solve_two_stage_stochastic(
    products: list[str],
    scenarios: list[Scenario],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    holding_costs: dict[str, float],
    capacity: dict[str, float] | None = None,
) -> TwoStageResult:
    """
    Solve two-stage stochastic program with recourse.
    
    First stage: decide production quantities before demand is known.
    Second stage: after demand realizes, decide shortage/surplus.
    
    Minimize: production cost + E[shortage cost + holding cost]
    """
    import time
    start = time.time()
    
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return TwoStageResult(
            status="ERROR", expected_cost=0, first_stage_decisions={},
            recourse_decisions={}, solve_time=0
        )
    
    # First stage: production variables
    prod = {p: solver.NumVar(0, capacity.get(p, solver.infinity()) if capacity else solver.infinity(), f"prod_{p}") 
            for p in products}
    
    # Second stage: shortage and surplus for each scenario
    shortage = {(s.name, p): solver.NumVar(0, solver.infinity(), f"short_{s.name}_{p}")
                for s in scenarios for p in products}
    surplus = {(s.name, p): solver.NumVar(0, solver.infinity(), f"surp_{s.name}_{p}")
               for s in scenarios for p in products}
    
    # Balance constraints: prod + shortage - surplus = demand
    for s in scenarios:
        for p in products:
            demand = s.demand.get(p, 0)
            solver.Add(prod[p] + shortage[(s.name, p)] - surplus[(s.name, p)] == demand)
    
    # Objective: production cost + expected recourse cost
    obj = sum(production_costs.get(p, 0) * prod[p] for p in products)
    for s in scenarios:
        for p in products:
            obj += s.probability * (
                shortage_costs.get(p, 0) * shortage[(s.name, p)] +
                holding_costs.get(p, 0) * surplus[(s.name, p)]
            )
    solver.Minimize(obj)
    
    status = solver.Solve()
    solve_time = time.time() - start
    
    if status != pywraplp.Solver.OPTIMAL:
        return TwoStageResult(
            status="INFEASIBLE", expected_cost=0, first_stage_decisions={},
            recourse_decisions={}, solve_time=solve_time
        )
    
    return TwoStageResult(
        status="OPTIMAL",
        expected_cost=round(solver.Objective().Value(), 2),
        first_stage_decisions={p: round(prod[p].solution_value(), 2) for p in products},
        recourse_decisions={
            s.name: {
                f"{p}_shortage": round(shortage[(s.name, p)].solution_value(), 2)
                for p in products if shortage[(s.name, p)].solution_value() > 0.01
            } | {
                f"{p}_surplus": round(surplus[(s.name, p)].solution_value(), 2)
                for p in products if surplus[(s.name, p)].solution_value() > 0.01
            }
            for s in scenarios
        },
        solve_time=round(solve_time, 4)
    )


def solve_newsvendor(
    selling_price: float,
    cost: float,
    salvage_value: float,
    mean_demand: float,
    std_demand: float,
) -> NewsvendorResult:
    """
    Solve the newsvendor (single-period stochastic inventory) problem.
    
    Assumes normally distributed demand.
    Critical ratio = (price - cost) / (price - salvage)
    Optimal Q = mean + z * std where z = Phi^{-1}(critical_ratio)
    """
    from scipy import stats
    
    if selling_price <= cost:
        return NewsvendorResult(
            status="INFEASIBLE", optimal_order_quantity=0,
            expected_profit=0, critical_ratio=0, stockout_probability=1
        )
    
    cu = selling_price - cost  # underage cost
    co = cost - salvage_value  # overage cost
    critical_ratio = cu / (cu + co)
    
    # Optimal order quantity
    z = stats.norm.ppf(critical_ratio)
    q_star = mean_demand + z * std_demand
    q_star = max(0, q_star)
    
    # Expected profit calculation
    # E[profit] = p*E[min(Q,D)] - c*Q + s*E[max(Q-D,0)]
    # For normal: E[min(Q,D)] = mean - std*L(z) where L(z) = phi(z) - z*(1-Phi(z))
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    loss_z = phi_z - z * (1 - Phi_z)
    
    expected_sales = mean_demand - std_demand * loss_z
    expected_leftover = q_star - expected_sales
    expected_profit = selling_price * expected_sales - cost * q_star + salvage_value * expected_leftover
    
    return NewsvendorResult(
        status="OPTIMAL",
        optimal_order_quantity=round(q_star, 2),
        expected_profit=round(expected_profit, 2),
        critical_ratio=round(critical_ratio, 4),
        stockout_probability=round(1 - critical_ratio, 4)
    )


def solve_lot_sizing(
    demands: list[float],
    setup_cost: float,
    holding_cost: float,
    production_cost: float = 0,
) -> LotSizingResult:
    """
    Solve dynamic lot sizing using Wagner-Whitin algorithm.
    
    Finds optimal production schedule to meet demands over T periods
    minimizing setup + holding + production costs.
    
    Key insight: optimal policy only produces in period t if inventory = 0.
    """
    T = len(demands)
    if T == 0:
        return LotSizingResult(
            status="OPTIMAL", total_cost=0, production_plan=[],
            inventory_levels=[], setup_periods=[]
        )
    
    # dp[t] = min cost to satisfy demands 0..t-1
    # We use the property that production in period j covers demands j..k for some k >= j
    INF = float('inf')
    dp = [INF] * (T + 1)
    dp[0] = 0
    parent = [-1] * (T + 1)  # tracks which period we produced from
    
    for j in range(T):
        if dp[j] == INF:
            continue
        # Try producing in period j to cover demands j..k
        cumulative_holding = 0
        cumulative_demand = 0
        for k in range(j, T):
            cumulative_demand += demands[k]
            # Holding cost for demand[k] held from period j to k
            cumulative_holding += demands[k] * (k - j) * holding_cost
            cost = dp[j] + setup_cost + production_cost * cumulative_demand + cumulative_holding
            if cost < dp[k + 1]:
                dp[k + 1] = cost
                parent[k + 1] = j
    
    # Backtrack to find production plan
    production_plan = [0.0] * T
    setup_periods = []
    t = T
    while t > 0:
        j = parent[t]
        setup_periods.append(j)
        production_plan[j] = sum(demands[j:t])
        t = j
    setup_periods.reverse()
    
    # Calculate inventory levels
    inventory = [0.0] * T
    current_inv = 0
    for t in range(T):
        current_inv += production_plan[t] - demands[t]
        inventory[t] = round(current_inv, 2)
    
    return LotSizingResult(
        status="OPTIMAL",
        total_cost=round(dp[T], 2),
        production_plan=[round(p, 2) for p in production_plan],
        inventory_levels=inventory,
        setup_periods=setup_periods
    )
