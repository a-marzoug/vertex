"""Solvers for stochastic and dynamic optimization problems."""

from typing import Any

import math

from ortools.linear_solver import pywraplp

from vertex.models.stochastic import (
    BinPacking2DResult,
    ChanceConstrainedResult,
    CrewScheduleResult,
    LotSizingResult,
    MonteCarloResult,
    MultiEchelonResult,
    NetworkDesignResult,
    NewsvendorResult,
    PortfolioQPResult,
    QAPResult,
    QPResult,
    QueueMetrics,
    RobustResult,
    Scenario,
    SteinerTreeResult,
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
            status="ERROR",
            expected_cost=0,
            first_stage_decisions={},
            recourse_decisions={},
            solve_time=0,
        )

    # First stage: production variables
    prod = {
        p: solver.NumVar(
            0,
            capacity.get(p, solver.infinity()) if capacity else solver.infinity(),
            f"prod_{p}",
        )
        for p in products
    }

    # Second stage: shortage and surplus for each scenario
    shortage = {
        (s.name, p): solver.NumVar(0, solver.infinity(), f"short_{s.name}_{p}")
        for s in scenarios
        for p in products
    }
    surplus = {
        (s.name, p): solver.NumVar(0, solver.infinity(), f"surp_{s.name}_{p}")
        for s in scenarios
        for p in products
    }

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
                shortage_costs.get(p, 0) * shortage[(s.name, p)]
                + holding_costs.get(p, 0) * surplus[(s.name, p)]
            )
    solver.Minimize(obj)

    status = solver.Solve()
    solve_time = time.time() - start

    if status != pywraplp.Solver.OPTIMAL:
        return TwoStageResult(
            status="INFEASIBLE",
            expected_cost=0,
            first_stage_decisions={},
            recourse_decisions={},
            solve_time=solve_time,
        )

    return TwoStageResult(
        status="OPTIMAL",
        expected_cost=round(solver.Objective().Value(), 2),
        first_stage_decisions={p: round(prod[p].solution_value(), 2) for p in products},
        recourse_decisions={
            s.name: {
                f"{p}_shortage": round(shortage[(s.name, p)].solution_value(), 2)
                for p in products
                if shortage[(s.name, p)].solution_value() > 0.01
            }
            | {
                f"{p}_surplus": round(surplus[(s.name, p)].solution_value(), 2)
                for p in products
                if surplus[(s.name, p)].solution_value() > 0.01
            }
            for s in scenarios
        },
        solve_time=round(solve_time, 4),
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
            status="INFEASIBLE",
            optimal_order_quantity=0,
            expected_profit=0,
            critical_ratio=0,
            stockout_probability=1,
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
    expected_profit = (
        selling_price * expected_sales
        - cost * q_star
        + salvage_value * expected_leftover
    )

    return NewsvendorResult(
        status="OPTIMAL",
        optimal_order_quantity=round(q_star, 2),
        expected_profit=round(expected_profit, 2),
        critical_ratio=round(critical_ratio, 4),
        stockout_probability=round(1 - critical_ratio, 4),
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
            status="OPTIMAL",
            total_cost=0,
            production_plan=[],
            inventory_levels=[],
            setup_periods=[],
        )

    # dp[t] = min cost to satisfy demands 0..t-1
    # We use the property that production in period j covers demands j..k for some k >= j
    INF = float("inf")
    dp = [INF] * (T + 1)
    dp[0] = 0
    parent = [-1] * (T + 1)  # tracks which period we produced from

    for j in range(T):
        if dp[j] == INF:
            continue
        # Try producing in period j to cover demands j..k
        cumulative_holding = 0.0
        cumulative_demand = 0.0
        for k in range(j, T):
            cumulative_demand += demands[k]
            # Holding cost for demand[k] held from period j to k
            cumulative_holding += demands[k] * (k - j) * holding_cost
            cost = (
                dp[j]
                + setup_cost
                + production_cost * cumulative_demand
                + cumulative_holding
            )
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
    current_inv = 0.0
    for t in range(T):
        current_inv += production_plan[t] - demands[t]
        inventory[t] = round(current_inv, 2)

    return LotSizingResult(
        status="OPTIMAL",
        total_cost=round(dp[T], 2),
        production_plan=[round(p, 2) for p in production_plan],
        inventory_levels=inventory,
        setup_periods=setup_periods,
    )


def solve_robust(
    products: list[str],
    nominal_demand: dict[str, float],
    demand_deviation: dict[str, float],
    uncertainty_budget: float,
    production_costs: dict[str, float],
    selling_prices: dict[str, float],
    capacity: dict[str, float] | None = None,
) -> "RobustResult":
    """
    Solve robust optimization using Bertsimas-Sim formulation.

    Worst-case profit = sum(price * min(prod, demand - z*deviation)) - cost*prod
    where sum(z) <= Gamma and 0 <= z <= 1
    """
    import time

    from vertex.models.stochastic import RobustResult

    start = time.time()
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return RobustResult(
            status="ERROR",
            objective_value=0,
            worst_case_objective=0,
            variable_values={},
            binding_scenarios=[],
            solve_time=0,
        )

    # Production variables
    prod = {
        p: solver.NumVar(
            0,
            capacity.get(p, solver.infinity()) if capacity else solver.infinity(),
            f"prod_{p}",
        )
        for p in products
    }

    # Dual variables for robust counterpart
    # For each product: sales <= demand - z * deviation
    # Robust: sales <= nominal_demand - lambda - mu_p * deviation
    # where lambda + sum(mu) <= Gamma, mu >= 0

    lam = solver.NumVar(0, solver.infinity(), "lambda")
    mu = {p: solver.NumVar(0, solver.infinity(), f"mu_{p}") for p in products}
    sales = {p: solver.NumVar(0, solver.infinity(), f"sales_{p}") for p in products}

    # Budget constraint
    solver.Add(
        lam * uncertainty_budget + sum(mu[p] for p in products)
        <= uncertainty_budget * max(1, len(products))
    )

    # Sales constraints (robust)
    for p in products:
        # sales <= prod
        solver.Add(sales[p] <= prod[p])
        # sales <= nominal - lambda - mu * deviation (worst case)
        solver.Add(sales[p] <= nominal_demand[p] - lam - mu[p] * demand_deviation[p])

    # Maximize worst-case profit
    profit = sum(
        selling_prices[p] * sales[p] - production_costs[p] * prod[p] for p in products
    )
    solver.Maximize(profit)

    status = solver.Solve()
    solve_time = time.time() - start

    if status != pywraplp.Solver.OPTIMAL:
        return RobustResult(
            status="INFEASIBLE",
            objective_value=0,
            worst_case_objective=0,
            variable_values={},
            binding_scenarios=[],
            solve_time=solve_time,
        )

    # Find binding scenarios (where mu > 0)
    binding = [p for p in products if mu[p].solution_value() > 0.01]

    return RobustResult(
        status="OPTIMAL",
        objective_value=round(solver.Objective().Value(), 2),
        worst_case_objective=round(solver.Objective().Value(), 2),
        variable_values={p: round(prod[p].solution_value(), 2) for p in products},
        binding_scenarios=binding,
        solve_time=round(solve_time, 4),
    )


def compute_mm1_metrics(arrival_rate: float, service_rate: float) -> "QueueMetrics":
    """Compute M/M/1 queue metrics."""
    from vertex.models.stochastic import QueueMetrics

    if arrival_rate >= service_rate:
        return QueueMetrics(
            utilization=1.0,
            avg_queue_length=float("inf"),
            avg_system_length=float("inf"),
            avg_wait_time=float("inf"),
            avg_system_time=float("inf"),
            prob_wait=1.0,
            prob_empty=0.0,
        )

    rho = arrival_rate / service_rate
    Lq = rho**2 / (1 - rho)
    L = rho / (1 - rho)
    Wq = Lq / arrival_rate
    W = L / arrival_rate

    return QueueMetrics(
        utilization=round(rho, 4),
        avg_queue_length=round(Lq, 4),
        avg_system_length=round(L, 4),
        avg_wait_time=round(Wq, 4),
        avg_system_time=round(W, 4),
        prob_wait=round(rho, 4),
        prob_empty=round(1 - rho, 4),
    )


def compute_mmc_metrics(
    arrival_rate: float, service_rate: float, num_servers: int
) -> "QueueMetrics":
    """Compute M/M/c queue metrics using Erlang-C formula."""
    from vertex.models.stochastic import QueueMetrics

    c = num_servers
    lam = arrival_rate
    mu = service_rate
    rho = lam / (c * mu)

    if rho >= 1:
        return QueueMetrics(
            utilization=1.0,
            avg_queue_length=float("inf"),
            avg_system_length=float("inf"),
            avg_wait_time=float("inf"),
            avg_system_time=float("inf"),
            prob_wait=1.0,
            prob_empty=0.0,
        )

    # Erlang-C: P(wait) = C(c, a) where a = lam/mu
    a = lam / mu

    # P0 = probability of empty system
    sum_terms = sum((a**n) / math.factorial(n) for n in range(c))
    last_term = (a**c) / (math.factorial(c) * (1 - rho))
    P0 = 1 / (sum_terms + last_term)

    # Erlang-C formula
    Pc = ((a**c) / math.factorial(c)) * (1 / (1 - rho)) * P0

    Lq = Pc * rho / (1 - rho)
    L = Lq + a
    Wq = Lq / lam
    W = Wq + 1 / mu

    return QueueMetrics(
        utilization=round(rho, 4),
        avg_queue_length=round(Lq, 4),
        avg_system_length=round(L, 4),
        avg_wait_time=round(Wq, 4),
        avg_system_time=round(W, 4),
        prob_wait=round(Pc, 4),
        prob_empty=round(P0, 4),
    )


def run_monte_carlo_newsvendor(
    selling_price: float,
    cost: float,
    salvage_value: float,
    order_quantity: float,
    mean_demand: float,
    std_demand: float,
    num_simulations: int = 10000,
) -> "MonteCarloResult":
    """
    Run Monte Carlo simulation for newsvendor profit distribution.
    """
    import numpy as np

    from vertex.models.stochastic import MonteCarloResult

    np.random.seed(42)
    demands = np.random.normal(mean_demand, std_demand, num_simulations)
    demands = np.maximum(demands, 0)  # No negative demand

    sales = np.minimum(order_quantity, demands)
    leftover = np.maximum(order_quantity - demands, 0)
    profits = selling_price * sales - cost * order_quantity + salvage_value * leftover

    return MonteCarloResult(
        status="COMPLETED",
        num_simulations=num_simulations,
        mean_objective=round(float(np.mean(profits)), 2),
        std_objective=round(float(np.std(profits)), 2),
        percentile_5=round(float(np.percentile(profits, 5)), 2),
        percentile_50=round(float(np.percentile(profits, 50)), 2),
        percentile_95=round(float(np.percentile(profits, 95)), 2),
        prob_feasible=1.0,
        var_95=round(float(np.percentile(profits, 5)), 2),  # VaR at 95% confidence
    )


def run_monte_carlo_production(
    products: list[str],
    production_quantities: dict[str, float],
    mean_demands: dict[str, float],
    std_demands: dict[str, float],
    selling_prices: dict[str, float],
    production_costs: dict[str, float],
    shortage_costs: dict[str, float],
    num_simulations: int = 10000,
) -> "MonteCarloResult":
    """
    Run Monte Carlo simulation for production planning profit distribution.
    """
    import numpy as np

    from vertex.models.stochastic import MonteCarloResult

    np.random.seed(42)
    profits = np.zeros(num_simulations)

    for p in products:
        demands = np.random.normal(mean_demands[p], std_demands[p], num_simulations)
        demands = np.maximum(demands, 0)
        q = production_quantities[p]

        sales = np.minimum(q, demands)
        shortage = np.maximum(demands - q, 0)

        profits += (
            selling_prices[p] * sales
            - production_costs[p] * q
            - shortage_costs[p] * shortage
        )

    return MonteCarloResult(
        status="COMPLETED",
        num_simulations=num_simulations,
        mean_objective=round(float(np.mean(profits)), 2),
        std_objective=round(float(np.std(profits)), 2),
        percentile_5=round(float(np.percentile(profits, 5)), 2),
        percentile_50=round(float(np.percentile(profits, 50)), 2),
        percentile_95=round(float(np.percentile(profits, 95)), 2),
        prob_feasible=1.0,
        var_95=round(float(np.percentile(profits, 5)), 2),
    )


def solve_crew_scheduling(
    workers: list[str],
    days: int,
    shifts: list[str],
    requirements: dict[str, list[int]],
    worker_availability: dict[str, list[tuple[int, str]]] | None = None,
    costs: dict[str, float] | None = None,
    max_shifts_per_worker: int | None = None,
    min_rest_between_shifts: int = 0,
    time_limit_seconds: int = 30,
) -> "CrewScheduleResult":
    """
    Solve crew scheduling with availability and rest constraints.
    """
    import time

    from ortools.sat.python import cp_model

    from vertex.models.stochastic import CrewScheduleResult

    start_time = time.time()
    model = cp_model.CpModel()

    n_workers = len(workers)
    n_shifts = len(shifts)
    costs = costs or {w: 1 for w in workers}

    # x[w, d, s] = 1 if worker w works day d shift s
    x = {}
    for w in range(n_workers):
        for d in range(days):
            for s in range(n_shifts):
                x[(w, d, s)] = model.new_bool_var(f"x_{w}_{d}_{s}")

    # Availability constraints
    if worker_availability:
        for w_idx, w in enumerate(workers):
            if w in worker_availability:
                available = set(worker_availability[w])
                for d in range(days):
                    for s_idx, s in enumerate(shifts):
                        if (d, s) not in available:
                            model.add(x[(w_idx, d, s_idx)] == 0)

    # Coverage requirements
    for s_idx, s in enumerate(shifts):
        reqs = requirements.get(s, [0] * days)
        for d in range(days):
            model.add(sum(x[(w, d, s_idx)] for w in range(n_workers)) >= reqs[d])

    # Max one shift per day per worker
    for w in range(n_workers):
        for d in range(days):
            model.add(sum(x[(w, d, s)] for s in range(n_shifts)) <= 1)

    # Max shifts per worker
    if max_shifts_per_worker:
        for w in range(n_workers):
            model.add(
                sum(x[(w, d, s)] for d in range(days) for s in range(n_shifts))
                <= max_shifts_per_worker
            )

    # Minimum rest between shifts (consecutive days)
    if min_rest_between_shifts > 0 and n_shifts > 1:
        for w in range(n_workers):
            for d in range(days - 1):
                # If worked last shift of day d, can't work first shift of day d+1
                model.add(x[(w, d, n_shifts - 1)] + x[(w, d + 1, 0)] <= 1)

    # Minimize cost
    model.minimize(
        sum(
            int(costs[workers[w]] * 100) * x[(w, d, s)]
            for w in range(n_workers)
            for d in range(days)
            for s in range(n_shifts)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = time.time() - start_time

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return CrewScheduleResult(
            status="INFEASIBLE",
            total_cost=0,
            assignments={},
            coverage={},
            solve_time=elapsed,
        )

    assignments = {w: [] for w in workers}
    coverage = {s: {d: 0 for d in range(days)} for s in shifts}
    total_cost = 0.0

    for w in range(n_workers):
        for d in range(days):
            for s in range(n_shifts):
                if solver.value(x[(w, d, s)]):
                    assignments[workers[w]].append(f"day{d}_{shifts[s]}")
                    coverage[shifts[s]][d] += 1
                    total_cost += costs[workers[w]]

    return CrewScheduleResult(
        status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",  # type: ignore[comparison-overlap]
        total_cost=round(total_cost, 2),
        assignments={w: a for w, a in assignments.items() if a},
        coverage=coverage,
        solve_time=round(elapsed, 4),
    )


def solve_chance_constrained(
    products: list[str],
    mean_demands: dict[str, float],
    std_demands: dict[str, float],
    production_costs: dict[str, float],
    selling_prices: dict[str, float],
    service_level: float = 0.95,
    capacity: dict[str, float] | None = None,
) -> "ChanceConstrainedResult":
    """
    Solve chance-constrained production planning.

    Ensures P(production >= demand) >= service_level for each product.
    Uses deterministic equivalent with safety stock.
    """
    import time

    from scipy import stats

    from vertex.models.stochastic import ChanceConstrainedResult

    start = time.time()
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return ChanceConstrainedResult(
            status="ERROR",
            objective_value=0,
            variable_values={},
            constraint_satisfaction_probs={},
            solve_time=0,
        )

    # z-score for service level
    z = stats.norm.ppf(service_level)

    # Production variables
    prod = {
        p: solver.NumVar(
            0,
            capacity.get(p, solver.infinity()) if capacity else solver.infinity(),
            f"prod_{p}",
        )
        for p in products
    }

    # Chance constraint: prod >= mean + z * std (deterministic equivalent)
    for p in products:
        min_prod = mean_demands[p] + z * std_demands[p]
        solver.Add(prod[p] >= min_prod)

    # Maximize expected profit (assuming we sell min(prod, demand))
    # Simplified: profit = price * mean_demand - cost * prod (conservative)
    profit = sum(
        selling_prices[p] * mean_demands[p] - production_costs[p] * prod[p]
        for p in products
    )
    solver.Maximize(profit)

    status = solver.Solve()
    solve_time = time.time() - start

    if status != pywraplp.Solver.OPTIMAL:
        return ChanceConstrainedResult(
            status="INFEASIBLE",
            objective_value=0,
            variable_values={},
            constraint_satisfaction_probs={},
            solve_time=solve_time,
        )

    # Calculate actual satisfaction probabilities
    probs = {}
    for p in products:
        q = prod[p].solution_value()
        # P(demand <= q) = Phi((q - mean) / std)
        prob = stats.norm.cdf((q - mean_demands[p]) / std_demands[p])
        probs[p] = round(prob, 4)

    return ChanceConstrainedResult(
        status="OPTIMAL",
        objective_value=round(solver.Objective().Value(), 2),
        variable_values={p: round(prod[p].solution_value(), 2) for p in products},
        constraint_satisfaction_probs=probs,
        solve_time=round(solve_time, 4),
    )


def solve_2d_bin_packing(
    rectangles: list[dict[str, Any]],
    bin_width: int,
    bin_height: int,
    max_bins: int | None = None,
    allow_rotation: bool = True,
    time_limit_seconds: int = 30,
) -> "BinPacking2DResult":
    """
    Solve 2D bin packing using CP-SAT.
    """
    import time

    from ortools.sat.python import cp_model

    from vertex.models.stochastic import BinPacking2DResult, RectanglePlacement

    start = time.time()
    model = cp_model.CpModel()

    n_rects = len(rectangles)
    n_bins = max_bins or n_rects

    # Variables
    x = {}  # x position
    y = {}  # y position
    b = {}  # bin assignment
    r = {}  # rotation (if allowed)
    w = {}  # effective width
    h = {}  # effective height

    for i, rect in enumerate(rectangles):
        x[i] = model.new_int_var(0, bin_width - 1, f"x_{i}")
        y[i] = model.new_int_var(0, bin_height - 1, f"y_{i}")
        b[i] = model.new_int_var(0, n_bins - 1, f"b_{i}")

        if allow_rotation:
            r[i] = model.new_bool_var(f"r_{i}")
            w[i] = model.new_int_var(1, max(rect["width"], rect["height"]), f"w_{i}")
            h[i] = model.new_int_var(1, max(rect["width"], rect["height"]), f"h_{i}")
            # If rotated, swap width/height
            model.add(w[i] == rect["width"]).only_enforce_if(r[i].negated())
            model.add(h[i] == rect["height"]).only_enforce_if(r[i].negated())
            model.add(w[i] == rect["height"]).only_enforce_if(r[i])
            model.add(h[i] == rect["width"]).only_enforce_if(r[i])
        else:
            w[i] = rect["width"]
            h[i] = rect["height"]

    # Fit within bin
    for i, rect in enumerate(rectangles):
        if allow_rotation:
            model.add(x[i] + w[i] <= bin_width)
            model.add(y[i] + h[i] <= bin_height)
        else:
            model.add(x[i] + rect["width"] <= bin_width)
            model.add(y[i] + rect["height"] <= bin_height)

    # No overlap within same bin
    for i in range(n_rects):
        for j in range(i + 1, n_rects):
            # Either different bins or no overlap
            same_bin = model.new_bool_var(f"same_{i}_{j}")
            model.add(b[i] == b[j]).only_enforce_if(same_bin)
            model.add(b[i] != b[j]).only_enforce_if(same_bin.negated())

            # If same bin, no overlap (one of 4 conditions)
            left = model.new_bool_var(f"left_{i}_{j}")
            right = model.new_bool_var(f"right_{i}_{j}")
            below = model.new_bool_var(f"below_{i}_{j}")
            above = model.new_bool_var(f"above_{i}_{j}")

            wi = w[i] if allow_rotation else rectangles[i]["width"]
            hi = h[i] if allow_rotation else rectangles[i]["height"]
            wj = w[j] if allow_rotation else rectangles[j]["width"]
            hj = h[j] if allow_rotation else rectangles[j]["height"]

            model.add(x[i] + wi <= x[j]).only_enforce_if(left)
            model.add(x[j] + wj <= x[i]).only_enforce_if(right)
            model.add(y[i] + hi <= y[j]).only_enforce_if(below)
            model.add(y[j] + hj <= y[i]).only_enforce_if(above)

            # If same bin, at least one separation must hold
            model.add_bool_or([left, right, below, above, same_bin.negated()])

    # Minimize max bin used
    max_bin = model.new_int_var(0, n_bins - 1, "max_bin")
    model.add_max_equality(max_bin, [b[i] for i in range(n_rects)])
    model.minimize(max_bin)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = time.time() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return BinPacking2DResult(
            status="INFEASIBLE",
            num_bins_used=0,
            placements=[],
            bin_utilization={},
            solve_time=elapsed,
        )

    placements = []
    bin_areas = {}
    for i, rect in enumerate(rectangles):
        bin_id = solver.value(b[i])
        rotated = solver.value(r[i]) if allow_rotation else False
        pw = solver.value(w[i]) if allow_rotation else rect["width"]
        ph = solver.value(h[i]) if allow_rotation else rect["height"]

        placements.append(
            RectanglePlacement(
                name=rect["name"],
                bin_id=bin_id,
                x=solver.value(x[i]),
                y=solver.value(y[i]),
                width=pw,
                height=ph,
                rotated=bool(rotated),
            )
        )
        bin_areas[bin_id] = bin_areas.get(bin_id, 0) + pw * ph

    num_bins = solver.value(max_bin) + 1
    utilization = {
        bid: round(area / (bin_width * bin_height), 4)
        for bid, area in bin_areas.items()
    }

    return BinPacking2DResult(
        status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",  # type: ignore[comparison-overlap]
        num_bins_used=num_bins,
        placements=placements,
        bin_utilization=utilization,
        solve_time=round(elapsed, 4),
    )


def solve_network_design(
    nodes: list[str],
    potential_arcs: list[dict[str, Any]],
    commodities: list[dict[str, Any]],
    arc_fixed_costs: dict[tuple[str, str], float],
    arc_capacities: dict[tuple[str, str], float],
    arc_variable_costs: dict[tuple[str, str], float],
    time_limit_seconds: int = 30,
) -> "NetworkDesignResult":
    """
    Solve capacitated network design - decide which arcs to open.
    """
    import time

    from ortools.linear_solver import pywraplp

    from vertex.models.stochastic import NetworkDesignResult

    start = time.time()
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return NetworkDesignResult(
            status="ERROR",
            total_cost=0,
            opened_facilities=[],
            opened_arcs=[],
            flows={},
            solve_time=0,
        )

    # Arc opening variables (binary)
    y = {}
    for arc in potential_arcs:
        key = (arc["source"], arc["target"])
        y[key] = solver.BoolVar(f"y_{key}")

    # Flow variables per commodity
    x = {}
    for k, comm in enumerate(commodities):
        for arc in potential_arcs:
            key = (arc["source"], arc["target"])
            x[(k, key)] = solver.NumVar(0, solver.infinity(), f"x_{k}_{key}")

    # Flow conservation
    for k, comm in enumerate(commodities):
        for node in nodes:
            inflow = sum(
                x[(k, (arc["source"], arc["target"]))]
                for arc in potential_arcs
                if arc["target"] == node
            )
            outflow = sum(
                x[(k, (arc["source"], arc["target"]))]
                for arc in potential_arcs
                if arc["source"] == node
            )

            if node == comm["source"]:
                solver.Add(outflow - inflow == comm["demand"])
            elif node == comm["sink"]:
                solver.Add(inflow - outflow == comm["demand"])
            else:
                solver.Add(inflow == outflow)

    # Capacity constraints (only if arc is open)
    for arc in potential_arcs:
        key = (arc["source"], arc["target"])
        cap = arc_capacities.get(key, 1e6)
        total_flow = sum(x[(k, key)] for k in range(len(commodities)))
        solver.Add(total_flow <= cap * y[key])

    # Objective: fixed costs + variable costs
    fixed_cost = sum(arc_fixed_costs.get(key, 0) * y[key] for key in y)
    var_cost = sum(
        arc_variable_costs.get(key, 0) * x[(k, key)]
        for k in range(len(commodities))
        for key in y
    )
    solver.Minimize(fixed_cost + var_cost)

    solver.set_time_limit(time_limit_seconds * 1000)
    status = solver.Solve()
    elapsed = time.time() - start

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return NetworkDesignResult(
            status="INFEASIBLE",
            total_cost=0,
            opened_facilities=[],
            opened_arcs=[],
            flows={},
            solve_time=elapsed,
        )

    opened = [(s, t) for (s, t), var in y.items() if var.solution_value() > 0.5]
    flows = {}
    for k, comm in enumerate(commodities):
        flows[comm["name"]] = {
            f"{s}->{t}": round(x[(k, (s, t))].solution_value(), 2)
            for (s, t) in y
            if x[(k, (s, t))].solution_value() > 0.01
        }

    return NetworkDesignResult(
        status="OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
        total_cost=round(solver.Objective().Value(), 2),
        opened_facilities=[],
        opened_arcs=opened,
        flows=flows,
        solve_time=round(elapsed, 4),
    )


def solve_qap(
    facilities: list[str],
    locations: list[str],
    flow_matrix: dict[str, dict[str, float]],
    distance_matrix: dict[str, dict[str, float]],
    time_limit_seconds: int = 30,
) -> "QAPResult":
    """
    Solve Quadratic Assignment Problem - assign facilities to locations
    minimizing total flow * distance.
    """
    import time

    from ortools.sat.python import cp_model

    from vertex.models.stochastic import QAPResult

    start = time.time()
    model = cp_model.CpModel()

    n = len(facilities)
    if n != len(locations):
        return QAPResult(status="ERROR", total_cost=0, assignment={}, solve_time=0)

    # x[i][j] = 1 if facility i assigned to location j
    x = {}
    for i, f in enumerate(facilities):
        for j, l in enumerate(locations):
            x[(i, j)] = model.new_bool_var(f"x_{i}_{j}")

    # Each facility to exactly one location
    for i in range(n):
        model.add_exactly_one(x[(i, j)] for j in range(n))

    # Each location gets exactly one facility
    for j in range(n):
        model.add_exactly_one(x[(i, j)] for i in range(n))

    # Linearize quadratic objective using auxiliary variables
    # cost = sum_{i,k,j,l} flow[i][k] * dist[j][l] * x[i][j] * x[k][l]
    obj_terms = []
    for i, fi in enumerate(facilities):
        for k, fk in enumerate(facilities):
            if fi == fk:
                continue
            flow = flow_matrix.get(fi, {}).get(fk, 0)
            if flow == 0:
                continue
            for j, lj in enumerate(locations):
                for l, ll in enumerate(locations):
                    if lj == ll:
                        continue
                    dist = distance_matrix.get(lj, {}).get(ll, 0)
                    if dist == 0:
                        continue
                    # x[i][j] * x[k][l] linearization
                    y = model.new_bool_var(f"y_{i}_{j}_{k}_{l}")
                    model.add_implication(y, x[(i, j)])
                    model.add_implication(y, x[(k, l)])
                    model.add_bool_or([y, x[(i, j)].negated(), x[(k, l)].negated()])
                    obj_terms.append(int(flow * dist) * y)

    model.minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = time.time() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return QAPResult(
            status="INFEASIBLE", total_cost=0, assignment={}, solve_time=elapsed
        )

    assignment = {}
    for i, f in enumerate(facilities):
        for j, l in enumerate(locations):
            if solver.value(x[(i, j)]):
                assignment[f] = l

    return QAPResult(
        status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",  # type: ignore[comparison-overlap]
        total_cost=solver.objective_value,
        assignment=assignment,
        solve_time=round(elapsed, 4),
    )


def solve_steiner_tree(
    nodes: list[str],
    edges: list[dict[str, Any]],
    terminals: list[str],
    time_limit_seconds: int = 30,
) -> "SteinerTreeResult":
    """
    Solve Steiner Tree - connect terminal nodes with minimum cost,
    optionally using non-terminal (Steiner) nodes.
    """
    import time

    from ortools.sat.python import cp_model

    from vertex.models.stochastic import SteinerTreeResult

    start = time.time()
    model = cp_model.CpModel()

    # Create edge variables
    edge_vars = {}
    edge_weights = {}
    for e in edges:
        key = (e["source"], e["target"])
        edge_vars[key] = model.new_bool_var(f"e_{key}")
        edge_weights[key] = e["weight"]
        # Undirected - add reverse
        rev_key = (e["target"], e["source"])
        edge_vars[rev_key] = edge_vars[key]
        edge_weights[rev_key] = e["weight"]

    # Node usage variables
    node_vars = {n: model.new_bool_var(f"n_{n}") for n in nodes}

    # Terminals must be used
    for t in terminals:
        model.add(node_vars[t] == 1)

    # If edge used, both endpoints must be used
    for (u, v), var in edge_vars.items():
        model.add_implication(var, node_vars[u])
        model.add_implication(var, node_vars[v])

    # Connectivity: use flow-based formulation
    # Pick first terminal as root, flow from root to all other terminals
    if len(terminals) >= 2:
        root = terminals[0]
        for t in terminals[1:]:
            # Flow variable for path to terminal t
            flow = {}
            for u, v in edge_vars:
                flow[(u, v, t)] = model.new_bool_var(f"flow_{u}_{v}_{t}")
                # Flow only if edge is used
                model.add_implication(flow[(u, v, t)], edge_vars[(u, v)])

            # Flow conservation
            for n in nodes:
                inflow = sum(
                    flow.get((u, n, t), model.new_constant(0))
                    for u in nodes
                    if (u, n) in edge_vars
                )
                outflow = sum(
                    flow.get((n, v, t), model.new_constant(0))
                    for v in nodes
                    if (n, v) in edge_vars
                )
                if n == root:
                    model.add(outflow - inflow == 1)
                elif n == t:
                    model.add(inflow - outflow == 1)
                else:
                    model.add(inflow == outflow)

    # Minimize total edge weight (count each undirected edge once)
    seen = set()
    obj_terms = []
    for (u, v), var in edge_vars.items():
        if (v, u) not in seen:
            obj_terms.append(edge_weights[(u, v)] * var)
            seen.add((u, v))
    model.minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.solve(model)
    elapsed = time.time() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):  # type: ignore[comparison-overlap]
        return SteinerTreeResult(
            status="INFEASIBLE",
            total_weight=0,
            edges=[],
            steiner_nodes=[],
            solve_time=elapsed,
        )

    result_edges = []
    seen = set()
    for (u, v), var in edge_vars.items():
        if solver.value(var) and (v, u) not in seen:
            result_edges.append((u, v, edge_weights[(u, v)]))
            seen.add((u, v))

    steiner = [n for n in nodes if solver.value(node_vars[n]) and n not in terminals]

    return SteinerTreeResult(
        status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",  # type: ignore[comparison-overlap]
        total_weight=solver.objective_value,
        edges=result_edges,
        steiner_nodes=steiner,
        solve_time=round(elapsed, 4),
    )


def solve_multi_echelon_inventory(
    locations: list[str],
    parent: dict[str, str | None],
    demands: dict[str, float],
    lead_times: dict[str, float],
    holding_costs: dict[str, float],
    service_levels: dict[str, float],
) -> "MultiEchelonResult":
    """
    Solve multi-echelon inventory - compute base-stock levels.
    Uses guaranteed service model approximation.
    """
    import time

    from scipy import stats

    from vertex.models.stochastic import MultiEchelonResult

    start = time.time()

    # Compute echelon stock levels using safety stock formula
    # S = mean_demand * lead_time + z * std_demand * sqrt(lead_time)
    base_stock = {}
    fill_rates = {}

    # Assume demand std = 0.3 * mean (coefficient of variation)
    cv = 0.3

    for loc in locations:
        demand = demands.get(loc, 0)
        lt = lead_times.get(loc, 1)
        sl = service_levels.get(loc, 0.95)

        if demand == 0:
            base_stock[loc] = 0
            fill_rates[loc] = 1.0
            continue

        z = stats.norm.ppf(sl)
        std_demand = cv * demand

        # Base stock = expected demand during lead time + safety stock
        mean_lt_demand = demand * lt
        safety_stock = z * std_demand * (lt**0.5)
        base_stock[loc] = round(mean_lt_demand + safety_stock, 2)
        fill_rates[loc] = round(sl, 4)

    elapsed = time.time() - start

    return MultiEchelonResult(
        status="OPTIMAL",
        total_cost=round(
            sum(base_stock[l] * holding_costs.get(l, 1) for l in locations), 2
        ),
        base_stock_levels=base_stock,
        expected_fill_rates=fill_rates,
        solve_time=round(elapsed, 4),
    )


def solve_qp(
    variables: list[str],
    Q: list[list[float]],  # Quadratic term (n x n)
    c: list[float],  # Linear term
    A_eq: list[list[float]] | None = None,
    b_eq: list[float] | None = None,
    A_ineq: list[list[float]] | None = None,
    b_ineq: list[float] | None = None,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
) -> "QPResult":
    """
    Solve Quadratic Programming: min 0.5 * x'Qx + c'x
    subject to A_eq @ x = b_eq, A_ineq @ x <= b_ineq, lb <= x <= ub
    """
    import time

    import cvxpy as cp
    import numpy as np

    from vertex.models.stochastic import QPResult

    start = time.time()
    n = len(variables)

    x = cp.Variable(n)
    Q_np = np.array(Q)
    c_np = np.array(c)

    # Objective: 0.5 * x'Qx + c'x
    objective = 0.5 * cp.quad_form(x, Q_np) + c_np @ x  # type: ignore[attr-defined]

    constraints = []

    if A_eq is not None and b_eq is not None:
        constraints.append(np.array(A_eq) @ x == np.array(b_eq))

    if A_ineq is not None and b_ineq is not None:
        constraints.append(np.array(A_ineq) @ x <= np.array(b_ineq))

    if lower_bounds is not None:
        constraints.append(x >= np.array(lower_bounds))

    if upper_bounds is not None:
        constraints.append(x <= np.array(upper_bounds))

    prob = cp.Problem(cp.Minimize(objective), constraints)

    try:
        prob.solve()  # type: ignore[no-untyped-call]
        elapsed = time.time() - start

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return QPResult(
                status="INFEASIBLE",
                objective_value=0,
                variable_values={},
                solve_time=elapsed,
            )

        return QPResult(
            status="OPTIMAL",
            objective_value=round(prob.value, 6),
            variable_values={
                variables[i]: round(float(x.value[i]), 6)
                for i in range(n)  # type: ignore[index]
            },
            solve_time=round(elapsed, 4),
        )
    except Exception as e:
        return QPResult(
            status=f"ERROR: {e}",
            objective_value=0,
            variable_values={},
            solve_time=time.time() - start,
        )


def solve_portfolio_qp(
    assets: list[str],
    expected_returns: list[float],
    covariance_matrix: list[list[float]],
    target_return: float | None = None,
    risk_aversion: float | None = None,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> "PortfolioQPResult":
    """
    Solve Markowitz portfolio optimization with covariance.

    Either minimize variance for target return, or maximize utility = return - risk_aversion * variance.
    """
    import time

    import cvxpy as cp
    import numpy as np

    from vertex.models.stochastic import PortfolioQPResult

    start = time.time()
    n = len(assets)

    w = cp.Variable(n)
    ret = np.array(expected_returns)
    cov = np.array(covariance_matrix)

    portfolio_return = ret @ w
    portfolio_variance = cp.quad_form(w, cov)  # type: ignore[attr-defined]

    constraints = [
        cp.sum(w) == 1,  # type: ignore[attr-defined]  # Fully invested
        w >= min_weight,
        w <= max_weight,
    ]

    if target_return is not None:
        # Minimize variance subject to target return
        constraints.append(portfolio_return >= target_return)
        objective = cp.Minimize(portfolio_variance)
    elif risk_aversion is not None:
        # Maximize return - risk_aversion * variance
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    else:
        # Default: minimize variance (minimum variance portfolio)
        objective = cp.Minimize(portfolio_variance)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()  # type: ignore[no-untyped-call]
        elapsed = time.time() - start

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return PortfolioQPResult(
                status="INFEASIBLE",
                expected_return=0,
                portfolio_variance=0,
                portfolio_std=0,
                sharpe_ratio=None,
                weights={},
                solve_time=elapsed,
            )

        weights = {assets[i]: round(float(w.value[i]), 6) for i in range(n)}  # type: ignore[index]
        exp_ret = float(ret @ w.value)
        var = float(w.value @ cov @ w.value)
        std = var**0.5
        sharpe = (exp_ret - risk_free_rate) / std if std > 0 else None

        return PortfolioQPResult(
            status="OPTIMAL",
            expected_return=round(exp_ret, 6),
            portfolio_variance=round(var, 6),
            portfolio_std=round(std, 6),
            sharpe_ratio=round(sharpe, 4) if sharpe else None,
            weights=weights,
            solve_time=round(elapsed, 4),
        )
    except Exception as e:
        return PortfolioQPResult(
            status=f"ERROR: {e}",
            expected_return=0,
            portfolio_variance=0,
            portfolio_std=0,
            sharpe_ratio=None,
            weights={},
            solve_time=time.time() - start,
        )
