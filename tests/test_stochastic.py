"""Tests for stochastic optimization."""

from vertex.models.stochastic import Scenario
from vertex.solvers.stochastic import solve_newsvendor, solve_two_stage_stochastic


def test_two_stage_stochastic():
    """Test two-stage stochastic programming."""
    scenarios = [
        Scenario(name="low", probability=0.3, demand={"P1": 50}),
        Scenario(name="high", probability=0.7, demand={"P1": 100}),
    ]

    result = solve_two_stage_stochastic(
        products=["P1"],
        scenarios=scenarios,
        production_costs={"P1": 10},
        shortage_costs={"P1": 50},
        holding_costs={"P1": 2},
    )

    assert result.status == "OPTIMAL"
    assert result.first_stage_decisions is not None


def test_newsvendor():
    """Test newsvendor problem."""
    result = solve_newsvendor(
        selling_price=20,
        cost=10,
        salvage_value=5,
        mean_demand=100,
        std_demand=20,
    )

    assert result.status == "OPTIMAL"
    assert result.optimal_order_quantity is not None
    assert result.optimal_order_quantity > 0


def test_robust_optimization():
    """Test robust optimization tool."""
    from vertex.tools.stochastic import solve_robust_optimization

    products = ["P1", "P2"]
    nominal_demand = {"P1": 100, "P2": 150}
    demand_deviation = {"P1": 20, "P2": 30}
    uncertainty_budget = 1.5
    production_costs = {"P1": 10, "P2": 12}
    selling_prices = {"P1": 20, "P2": 25}
    capacity = {"P1": 200, "P2": 200}

    result = solve_robust_optimization(
        products=products,
        nominal_demand=nominal_demand,
        demand_deviation=demand_deviation,
        uncertainty_budget=uncertainty_budget,
        production_costs=production_costs,
        selling_prices=selling_prices,
        capacity=capacity,
    )

    assert result.status == "OPTIMAL"
    assert result.objective_value > 0
    assert len(result.variable_values) == 2
    # Worst case objective is what we maximize, so it should be equal to objective_value
    # (The solver returns the same value for both fields in the implementation)
    assert result.worst_case_objective == result.objective_value


def test_lot_sizing():
    """Test dynamic lot sizing."""
    from vertex.tools.stochastic import compute_lot_sizing

    result = compute_lot_sizing(
        demands=[20, 50, 10, 50, 50],
        setup_cost=100,
        holding_cost=1,
        production_cost=10,
    )

    assert result.status == "OPTIMAL"
    assert result.total_cost > 0
    assert len(result.production_plan) == 5
    assert len(result.inventory_levels) == 5


def test_queuing():
    """Test queuing analysis tools."""
    from vertex.tools.stochastic import analyze_queue_mm1, analyze_queue_mmc

    # M/M/1
    mm1 = analyze_queue_mm1(arrival_rate=5, service_rate=10)
    assert mm1.utilization == 0.5
    assert mm1.avg_system_length > 0

    # M/M/c
    mmc = analyze_queue_mmc(arrival_rate=15, service_rate=10, num_servers=2)
    assert mmc.utilization == 0.75
    assert mmc.avg_wait_time > 0


def test_simulations():
    """Test Monte Carlo simulations."""
    from vertex.tools.stochastic import (
        simulate_newsvendor_monte_carlo,
        simulate_production_monte_carlo,
    )

    # Newsvendor simulation
    nv_sim = simulate_newsvendor_monte_carlo(
        selling_price=20,
        cost=10,
        salvage_value=5,
        order_quantity=100,
        mean_demand=100,
        std_demand=20,
        num_simulations=100,
    )
    assert nv_sim.status == "COMPLETED"
    assert nv_sim.mean_objective != 0

    # Production simulation
    prod_sim = simulate_production_monte_carlo(
        products=["P1"],
        production_quantities={"P1": 100},
        mean_demands={"P1": 100},
        std_demands={"P1": 20},
        selling_prices={"P1": 20},
        production_costs={"P1": 10},
        shortage_costs={"P1": 5},
        num_simulations=100,
    )
    assert prod_sim.status == "COMPLETED"
    assert prod_sim.mean_objective != 0


def test_crew_scheduling():
    """Test crew scheduling."""
    from vertex.tools.stochastic import schedule_crew

    result = schedule_crew(
        workers=["Alice", "Bob"],
        days=2,
        shifts=["Morning", "Evening"],
        requirements={"Morning": [1, 1], "Evening": [1, 0]},
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert len(result.assignments) > 0


def test_chance_constrained():
    """Test chance constrained production."""
    from vertex.tools.stochastic import solve_chance_constrained_production

    result = solve_chance_constrained_production(
        products=["P1"],
        mean_demands={"P1": 100},
        std_demands={"P1": 10},
        production_costs={"P1": 10},
        selling_prices={"P1": 20},
        service_level=0.95,
    )

    assert result.status == "OPTIMAL"
    assert result.variable_values["P1"] > 100  # Should be > mean due to service level


def test_bin_packing_2d():
    """Test 2D bin packing."""
    from vertex.tools.stochastic import pack_rectangles_2d

    rects = [
        {"name": "r1", "width": 2, "height": 3},
        {"name": "r2", "width": 3, "height": 2},
    ]

    result = pack_rectangles_2d(
        rectangles=rects,
        bin_width=5,
        bin_height=5,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert result.num_bins_used >= 1
    assert len(result.placements) == 2


def test_network_design():
    """Test network design."""
    from vertex.tools.stochastic import design_network

    nodes = ["S", "T", "U"]
    arcs = [
        {"source": "S", "target": "T"},
        {"source": "S", "target": "U"},
        {"source": "U", "target": "T"},
    ]
    commodities = [
        {"name": "c1", "source": "S", "sink": "T", "demand": 10},
    ]
    
    # Map arcs to strings for dict keys as per the tool interface
    arc_keys = [f"{a['source']}->{a['target']}" for a in arcs]
    
    fixed_costs = {k: 100 for k in arc_keys}
    capacities = {k: 100 for k in arc_keys}
    variable_costs = {k: 1 for k in arc_keys}

    result = design_network(
        nodes=nodes,
        potential_arcs=arcs,
        commodities=commodities,
        arc_fixed_costs=fixed_costs,
        arc_capacities=capacities,
        arc_variable_costs=variable_costs,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert result.total_cost > 0


def test_qap():
    """Test Quadratic Assignment Problem."""
    from vertex.tools.stochastic import solve_quadratic_assignment

    facilities = ["F1", "F2"]
    locations = ["L1", "L2"]
    flow = {"F1": {"F2": 10}, "F2": {"F1": 10}}
    dist = {"L1": {"L2": 5}, "L2": {"L1": 5}}

    result = solve_quadratic_assignment(
        facilities=facilities,
        locations=locations,
        flow_matrix=flow,
        distance_matrix=dist,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert len(result.assignment) == 2


def test_steiner_tree():
    """Test Steiner Tree."""
    from vertex.tools.stochastic import find_steiner_tree

    nodes = ["A", "B", "C", "D"]
    edges = [
        {"source": "A", "target": "B", "weight": 1},
        {"source": "B", "target": "C", "weight": 1},
        {"source": "C", "target": "D", "weight": 1},
        {"source": "A", "target": "D", "weight": 5},
    ]
    terminals = ["A", "D"]

    result = find_steiner_tree(
        nodes=nodes,
        edges=edges,
        terminals=terminals,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    # Path A-B-C-D (cost 3) is better than A-D (cost 5)
    assert result.total_weight == 3


def test_multi_echelon():
    """Test multi-echelon inventory."""
    from vertex.tools.stochastic import optimize_multi_echelon_inventory

    result = optimize_multi_echelon_inventory(
        locations=["DC", "Store"],
        parent={"Store": "DC", "DC": None},
        demands={"Store": 100, "DC": 0},
        lead_times={"Store": 1, "DC": 2},
        holding_costs={"Store": 2, "DC": 1},
        service_levels={"Store": 0.95, "DC": 0.95},
    )

    assert result.status == "OPTIMAL"
    assert result.total_cost > 0
    assert result.base_stock_levels["Store"] > 100


def test_qp():
    """Test Quadratic Programming."""
    from vertex.tools.stochastic import solve_quadratic_program

    # Minimize x^2 + y^2 subject to x + y = 1
    # Optimal: x = 0.5, y = 0.5, obj = 0.5
    result = solve_quadratic_program(
        variables=["x", "y"],
        Q=[[2, 0], [0, 2]],  # 2x^2 + 2y^2 in standard form is 0.5 * [x y] Q [x y]^T -> so Q should be diag(2) for 0.5*2*x^2 = x^2
        c=[0, 0],
        A_eq=[[1, 1]],
        b_eq=[1],
    )

    assert result.status == "OPTIMAL"
    assert abs(result.objective_value - 0.5) < 0.01
    assert abs(result.variable_values["x"] - 0.5) < 0.01


def test_portfolio_qp():
    """Test Portfolio QP."""
    from vertex.tools.stochastic import optimize_portfolio_qp

    assets = ["A", "B"]
    returns = [0.1, 0.2]
    cov = [[0.01, 0], [0, 0.04]]

    # Maximize return - 1 * variance
    result = optimize_portfolio_qp(
        assets=assets,
        expected_returns=returns,
        covariance_matrix=cov,
        risk_aversion=1.0,
    )

    assert result.status == "OPTIMAL"
    assert sum(result.weights.values()) > 0.99  # Weights sum to 1


def test_robust_supply_chain():
    """Test robust optimization for supply chain disruption (Scenario 4)."""
    from vertex.tools.stochastic import solve_robust_optimization

    # Scenario 4: Plan component sourcing from Supplier A and B robust to disruptions.
    # Nominal Supply: A=1000, B=1000.
    # Disruption: Supplier A might drop to 500, B to 600.
    # Budget: Max 1 supplier fails.
    # Costs: A=$10, B=$12.
    # Demand: 1500 units.

    suppliers = ["A", "B"]
    nominal_supply = {"A": 1000, "B": 1000}
    # Deviation = Nominal - Disrupted
    supply_deviation = {"A": 500, "B": 400}  # A drops to 500 (1000-500), B to 600 (1000-400)
    costs = {"A": 10, "B": 12}
    
    result = solve_robust_optimization(
        products=suppliers,
        uncertainty_budget=1.0, # Max 1 failure
        production_costs=costs, # Sourcing costs
        capacity=nominal_supply,
        capacity_deviation=supply_deviation,
        min_total_demand=1500,
    )

    assert result.status == "OPTIMAL"
    # To be robust against A failing (capacity 500), we need B to provide at least 1000?
    # If A fails: A=500. We need 1500 total. B can provide 1000. Total 1500. Feasible.
    # If B fails: B=600. A can provide 1000. Total 1600. Feasible.
    # Cost should be minimized.
    # Ideally, source max from A (cheaper).
    # If we source A=1000, B=500. Total 1500.
    # Solver optimizes to balance "robustness tax".
    # If A=900, Loss A = max(0, 900-500) = 400.
    # If B=1000, Loss B = max(0, 1000-600) = 400.
    # MaxLoss = 400.
    # Delivered = 1900 - 400 = 1500. OK.
    # Cost = 10*900 + 12*1000 = 21000.
    
    # If A=1000, Loss A = 500.
    # If B=1000, Loss B = 400.
    # MaxLoss = 500.
    # Delivered = 2000 - 500 = 1500. OK.
    # Cost = 10*1000 + 12*1000 = 22000.
    
    # 21000 < 22000. So A=900 is optimal.
    
    assert result.variable_values["A"] == 900
    assert result.variable_values["B"] == 1000
    assert result.objective_value == 21000


def test_yield_simulation():
    """Test Monte Carlo simulation with yield uncertainty (Scenario 6)."""
    from vertex.tools.stochastic import simulate_production_monte_carlo

    # Scenario 6: Semiconductor Yield
    # Wafer Cost: $5000.
    # Yield Rate: N(80%, 5%).
    # Chips per Wafer: 500.
    # Sale Price: $20/chip.
    
    # We simulate 1 batch (1 wafer).
    # Production Quantity = 1 (wafer).
    # Actually, the tool takes `production_quantities` which usually maps to `q` in cost equation.
    # If `production_costs` is per unit of `q`.
    # Let's map: q = 1 wafer.
    # Cost = 5000.
    # Yield Mean = 0.8 * 500 = 400 chips?
    # No, Yield Mean in tool is a multiplier.
    # So if q=1, yield should be chips count?
    # Or, q = 500 (potential chips). Cost = $10/potential chip? (5000/500).
    # Yield Rate N(0.8, 0.05).
    # Then actual chips = 500 * Yield.
    # This fits the tool model: q_actual = q_target * yield_factor.
    
    result = simulate_production_monte_carlo(
        products=["Chip"],
        production_quantities={"Chip": 500}, # 500 potential chips (1 wafer)
        mean_demands={"Chip": 10000}, # Infinite demand assumption
        std_demands={"Chip": 0},
        selling_prices={"Chip": 20},
        production_costs={"Chip": 10}, # $5000 / 500 = $10 per potential chip
        shortage_costs={"Chip": 0},
        yield_mean={"Chip": 0.8}, # 80% yield
        yield_std={"Chip": 0.05},
        num_simulations=100,
    )
    
    assert result.status == "COMPLETED"
    assert result.mean_objective > 0
