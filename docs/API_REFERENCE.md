# API Reference

Comprehensive reference for all 65 tools and 7 prompts available in the Vertex MCP Server.

## Linear Programming (LP)

### `solve_linear_program`

Solve a Linear Programming problem with continuous variables.

- **variables** (`list[dict]`): List of variable definitions (`name`, `lower_bound`, `upper_bound`).
- **constraints** (`list[dict]`): List of constraints (`name`, `coefficients`, `sense`, `rhs`).
- **objective_coefficients** (`dict[str, float]`): Dictionary mapping variable names to coefficients.
- **objective_sense** (`str`): "maximize" or "minimize".

### `analyze_lp_sensitivity`

Analyze LP solution sensitivity to parameter changes. Returns shadow prices and reduced costs.

- **variables**, **constraints**, **objective_coefficients**, **objective_sense**: Same as `solve_linear_program`.

### `analyze_what_if_scenario`

Perform what-if analysis by varying a constraint RHS value.

- **variables**, **constraints**, **objective_coefficients**, **objective_sense**: Standard LP inputs.
- **parameter_name** (`str`): Name of the constraint to vary.
- **parameter_values** (`list[float]`): List of values to test for the constraint RHS.

### `diagnose_infeasibility`

Diagnose why a problem is infeasible by finding conflicting constraints.

- **variables**, **constraints**, **objective_coefficients**, **objective_sense**: Standard LP inputs.

### `solve_pareto_frontier`

Solve multi-objective optimization and find Pareto frontier.

- **variables**, **constraints**: Standard definitions.
- **objectives** (`dict[str, dict[str, float]]`): Dict of objective_name -> {var: coef} mappings.
- **num_points** (`int`): Number of Pareto points to generate (default: 10).
- **objective_senses** (`dict[str, str]`): Dict of objective_name -> "maximize"/"minimize".

## Mixed-Integer Programming (MIP)

### `solve_mixed_integer_program`

Solve a MIP problem where variables can be integer or binary.

- **variables** (`list[dict]`): Definitions include `type` ("continuous", "integer", "binary").
- **constraints**, **objective_coefficients**, **objective_sense**: Standard inputs.

### `find_alternative_optimal_solutions`

Find multiple near-optimal solutions for MIP problems.

- **variables**, **constraints**, **objective_coefficients**, **objective_sense**: Standard MIP inputs.
- **max_solutions** (`int`): Maximum solutions to return (default: 5).
- **gap_tolerance** (`float`): Accept solutions within this fraction of optimal (default: 0.01).

## Network Optimization

### `find_max_flow`

Find maximum flow from source to sink in a network.

- **nodes** (`list[str]`): Node names.
- **arcs** (`list[dict]`): Arcs with `source`, `target`, `capacity`.
- **source** (`str`): Source node.
- **sink** (`str`): Sink node.

### `find_min_cost_flow`

Find minimum cost flow satisfying supplies and demands.

- **nodes** (`list[str]`): Node names.
- **arcs** (`list[dict]`): Arcs with `source`, `target`, `capacity`, `cost`.
- **supplies** (`dict[str, int]`): Node supplies (positive) and demands (negative).

### `find_shortest_path`

Find shortest path between two nodes.

- **nodes** (`list[str]`): Node names.
- **arcs** (`list[dict]`): Arcs with `source`, `target`, `cost`.
- **source** (`str`): Start node.
- **target** (`str`): End node.

### `find_minimum_spanning_tree`

Find Minimum Spanning Tree connecting all nodes with minimum total weight.

- **nodes** (`list[str]`): Node names.
- **edges** (`list[dict]`): Edges with `source`, `target`, `weight`.

### `find_multi_commodity_flow`

Route multiple commodities through shared network.

- **nodes** (`list[str]`): Node names.
- **arcs** (`list[dict]`): Arcs with `source`, `target`, `capacity`, `cost`.
- **commodities** (`list[dict]`): List with `name`, `source`, `sink`, `demand`.
- **time_limit_seconds** (`int`): Solver time limit.

### `solve_transshipment`

Solve Transshipment Problem - ship goods through intermediate nodes.

- **sources** (`list[str]`): Source nodes.
- **transshipment_nodes** (`list[str]`): Intermediate nodes.
- **destinations** (`list[str]`): Destination nodes.
- **supplies** (`dict[str, int]`): Supply at each source.
- **demands** (`dict[str, int]`): Demand at each destination.
- **costs** (`dict[str, dict[str, float]]`): Unit shipping costs.
- **capacities** (`dict[str, dict[str, float]]`): Optional max flow capacities.

## Scheduling & Routing

### `solve_tsp`

Solve Traveling Salesman Problem - find shortest tour visiting all locations.

- **locations** (`list[str]`): Location names.
- **distance_matrix** (`list[list[float]]`): 2D distance matrix.
- **time_limit_seconds** (`int`): Solver time limit.

### `solve_vrp`

Solve Capacitated Vehicle Routing Problem.

- **locations** (`list[str]`): Location names.
- **distance_matrix** (`list[list[float]]`): 2D distance matrix.
- **demands** (`list[int]`): Demand at each location.
- **vehicle_capacities** (`list[int]`): Capacity of each vehicle.
- **depot** (`int`): Index of depot location (default: 0).
- **time_limit_seconds** (`int`): Time limit.

### `solve_vrp_time_windows`

Solve VRP with Time Windows.

- **locations**, **distance_matrix**, **demands**, **vehicle_capacities**: Same as VRP.
- **time_matrix** (`list[list[int]]`): Travel time matrix.
- **time_windows** (`list[tuple[int, int]]`): (earliest, latest) arrival time per location.
- **depot** (`int`): Depot index (default: 0).
- **time_limit_seconds** (`int`): Time limit.

### `solve_pickup_delivery`

Solve VRP with pickup and delivery constraints.

- **locations** (`list[str]`): Location names.
- **distance_matrix** (`list[list[float]]`): Distance matrix.
- **pickups_deliveries** (`list[tuple[int, int]]`): Pairs of (pickup_location, delivery_location) indices.
- **vehicle_capacities** (`list[int]`): Vehicle capacities.
- **depot** (`int`): Depot index (default: 0).
- **time_limit_seconds** (`int`): Time limit.

### `solve_multi_depot_vrp`

Solve VRP with multiple depots.

- **locations** (`list[str]`): Location names.
- **distance_matrix** (`list[list[float]]`): Distance matrix.
- **demands** (`list[int]`): Demand at each location.
- **depots** (`list[int]`): Indices of depot locations.
- **vehicle_capacities** (`list[int]`): Vehicle capacities.
- **vehicles_per_depot** (`list[int]`): Number of vehicles at each depot.
- **time_limit_seconds** (`int`): Time limit.

### `solve_job_shop`

Schedule multi-step jobs on machines.

- **jobs** (`list[list[dict]]`): List of jobs, where each job is a sequence of `{machine, duration}` tasks.
- **time_limit_seconds** (`int`): Time limit.

### `solve_flexible_job_shop`

Solve Flexible Job Shop - tasks can run on alternative machines.

- **jobs** (`list[list[dict]]`): Each task has list of alternatives: `{"machines": [(machine_id, duration), ...]}`.
- **time_limit_seconds** (`int`): Time limit.

### `solve_flow_shop`

Solve Flow Shop Scheduling - all jobs follow same machine sequence.

- **processing_times** (`list[list[int]]`): `processing_times[job][machine]` = duration.
- **time_limit_seconds** (`int`): Time limit (default: 60).

### `solve_parallel_machines`

Assign jobs to identical parallel machines minimizing makespan.

- **job_durations** (`list[int]`): Duration of each job.
- **num_machines** (`int`): Number of identical machines.

### `solve_rcpsp`

Resource-Constrained Project Scheduling.

- **tasks** (`list[dict]`): Tasks with `name`, `duration`, `resources` (dict), `predecessors` (list).
- **resources** (`dict[str, int]`): Available global capacity per resource type.
- **time_limit_seconds** (`int`): Time limit.

## Combinatorial & Discrete

### `solve_bin_packing`

Pack items into minimum number of bins.

- **items** (`list[str]`): Item names.
- **weights** (`dict[str, float]`): Weight of each item.
- **bin_capacity** (`float`): Capacity of each bin.
- **max_bins** (`int`): Maximum bins available.

### `solve_set_cover`

Select minimum cost sets to cover all elements.

- **universe** (`list[str]`): Elements to cover.
- **sets** (`dict[str, list[str]]`): Map of set name to covered elements.
- **costs** (`dict[str, float]`): Cost of each set.

### `solve_graph_coloring`

Assign colors to nodes such that adjacent nodes differ.

- **nodes** (`list[str]`): Node names.
- **edges** (`list[tuple[str, str]]`): Edges.
- **max_colors** (`int`): Max colors.

### `solve_cutting_stock`

Cut items from stock minimizing waste.

- **items** (`list[str]`): Item names.
- **lengths** (`dict[str, int]`): Item lengths.
- **demands** (`dict[str, int]`): Item demands.
- **stock_length** (`int`): Length of stock material.

## Stochastic & Robust

### `solve_two_stage_stochastic`

Solve two-stage stochastic program for production.

- **products** (`list[str]`): Product names.
- **scenarios** (`list[dict]`): Scenarios with `probability` and `demand`.
- **production_costs**, **shortage_costs**, **holding_costs**: Cost parameters.

### `solve_newsvendor`

Solve single-period stochastic inventory.

- **selling_price**, **cost**, **salvage_value**, **mean_demand**, **std_demand**.

### `solve_lot_sizing`

Solve dynamic lot sizing (Wagner-Whitin algorithm).

- **demands** (`list[float]`): Demand for each period.
- **setup_cost** (`float`): Fixed cost to place an order.
- **holding_cost** (`float`): Cost to hold one unit for one period.
- **production_cost** (`float`): Variable cost per unit produced.

### `solve_robust_production`

Robust optimization under demand uncertainty (Bertsimas-Sim approach).

- **products** (`list[str]`): Product names.
- **nominal_demand** (`dict[str, float]`): Expected demand per product.
- **demand_deviation** (`dict[str, float]`): Maximum deviation from nominal.
- **uncertainty_budget** (`float`): Gamma parameter - max number of deviations.
- **production_costs** (`dict[str, float]`): Cost per unit.
- **selling_prices** (`dict[str, float]`): Revenue per unit.
- **capacity** (`float`): Optional production capacity limit.

## Queueing Analysis

### `analyze_mm1_queue`

Analyze M/M/1 queue (single server) performance metrics.

- **arrival_rate** (`float`): Lambda - average arrival rate.
- **service_rate** (`float`): Mu - average service rate.

Returns: utilization, average queue length, average wait time, probability of empty system.

### `analyze_mmc_queue`

Analyze M/M/c queue (multiple servers) performance metrics.

- **arrival_rate** (`float`): Lambda - average arrival rate.
- **service_rate** (`float`): Mu - average service rate per server.
- **num_servers** (`int`): Number of parallel servers.

Returns: utilization, average queue length, average wait time, probability of waiting.

## Monte Carlo Simulation

### `simulate_newsvendor`

Monte Carlo simulation for newsvendor profit distribution.

- **selling_price**, **cost**, **salvage_value**: Cost parameters.
- **order_quantity**: Fixed order quantity to evaluate.
- **mean_demand**, **std_demand**: Demand distribution.
- **num_simulations**: Number of runs (default: 10000).

### `simulate_production`

Monte Carlo simulation for multi-product production planning.

- **products**: Product names.
- **production_quantities**: Quantities to produce.
- **mean_demands**, **std_demands**: Demand distributions.
- **selling_prices**, **production_costs**, **shortage_costs**: Cost parameters.
- **num_simulations**: Number of runs.

## Crew Scheduling

### `solve_crew_schedule`

Solve crew/shift scheduling with constraints.

- **workers** (`list[str]`): Worker names.
- **days** (`int`): Number of days to schedule.
- **shifts** (`list[str]`): Shift names (e.g., ["morning", "afternoon", "night"]).
- **requirements** (`dict[str, list[int]]`): Required workers per shift per day - `{shift: [day0_count, day1_count, ...]}`.
- **worker_availability** (`dict[str, list[tuple[int, str]]]`): Optional - `{worker: [(day, shift), ...]}` of available slots.
- **costs** (`dict[str, float]`): Cost per worker per shift.
- **max_shifts_per_worker** (`int`): Maximum shifts per worker over the period.
- **min_rest_between_shifts** (`int`): Minimum days of rest between shifts.
- **time_limit_seconds** (`int`): Solver time limit.

## Chance-Constrained Programming

### `solve_chance_constrained`

Solve chance-constrained production planning with service level guarantees.

- **products** (`list[str]`): Product names.
- **mean_demands**, **std_demands**: Demand distribution parameters.
- **production_costs**, **selling_prices**: Cost parameters.
- **service_level** (`float`): Required probability of meeting demand (default: 0.95).
- **capacity**: Optional production limits.

## 2D Bin Packing

### `solve_2d_bin_packing`

Pack rectangles into 2D bins minimizing bins used.

- **rectangles** (`list[dict]`): List of `{name, width, height}`.
- **bin_width**, **bin_height**: Bin dimensions.
- **max_bins**: Maximum bins available.
- **allow_rotation**: Allow 90-degree rotation (default: True).

## Network Design

### `solve_network_design`

Capacitated network design - decide which arcs to build.

- **nodes** (`list[str]`): Node names.
- **potential_arcs** (`list[dict]`): List of `{source, target}`.
- **commodities** (`list[dict]`): List of `{name, source, sink, demand}`.
- **arc_fixed_costs** (`dict[str, float]`): Fixed cost to open arc `{"A->B": cost}`.
- **arc_capacities**: Capacity per arc.
- **arc_variable_costs**: Cost per unit flow.

### `solve_steiner_tree`

Connect terminal nodes with minimum total edge weight.

- **nodes** (`list[str]`): All node names.
- **edges** (`list[dict]`): List of `{source, target, weight}`.
- **terminals** (`list[str]`): Nodes that must be connected.

### `solve_quadratic_assignment_problem`

Assign facilities to locations minimizing flow * distance.

- **facilities**, **locations**: Names (same count).
- **flow_matrix**: `flow_matrix[f1][f2]` = flow between facilities.
- **distance_matrix**: `distance_matrix[l1][l2]` = distance between locations.

### `optimize_multi_echelon`

Multi-echelon inventory optimization with base-stock levels.

- **locations**: Location names (warehouses, DCs, stores).
- **parent**: `parent[loc]` = upstream location (None for top).
- **demands**: Mean demand per period.
- **lead_times**: Replenishment lead time.
- **holding_costs**: Holding cost per unit.
- **service_levels**: Target service level per location.

## Quadratic Programming

### `solve_qp`

Solve convex Quadratic Programming: min 0.5 * x'Qx + c'x.

- **variables** (`list[str]`): Variable names.
- **Q** (`list[list[float]]`): Quadratic coefficient matrix (must be positive semi-definite).
- **c** (`list[float]`): Linear coefficient vector.
- **A_eq**, **b_eq**: Equality constraints A_eq @ x = b_eq.
- **A_ineq**, **b_ineq**: Inequality constraints A_ineq @ x <= b_ineq.
- **lower_bounds**, **upper_bounds**: Variable bounds.

### `optimize_portfolio_variance`

Markowitz mean-variance portfolio optimization.

- **assets** (`list[str]`): Asset names.
- **expected_returns** (`list[float]`): Expected return for each asset.
- **covariance_matrix** (`list[list[float]]`): Covariance matrix of returns.
- **target_return** (`float`): If set, minimize variance for this target return.
- **risk_aversion** (`float`): If set, maximize return - risk_aversion * variance.
- **risk_free_rate** (`float`): Risk-free rate for Sharpe ratio calculation.
- **max_weight**, **min_weight**: Weight bounds per asset.

## Domain Templates

### `optimize_production_plan`

Maximize profit given resource constraints.

- **products**, **resources**, **profits**, **requirements**, **availability**.

### `optimize_investment_portfolio`

Maximize portfolio return.

- **assets**, **expected_returns**, **budget**, **min/max_allocation**.

### `optimize_worker_assignment`

Assign workers to tasks (1:1).

- **workers**, **tasks**, **costs**.

### `optimize_knapsack_selection`

Select items to maximize value (0/1 knapsack).

- **items**, **values**, **weights**, **capacity**.

### `optimize_facility_locations`

Facility location problem.

- **facilities**, **customers**, **fixed_costs**, **transport_costs**.

### `optimize_inventory_eoq`

Economic Order Quantity.

- **annual_demand**, **ordering_cost**, **holding_cost**, **lead_time**, **safety_stock**.

### `optimize_workforce`

Schedule workers to shifts.

- **workers**, **days**, **shifts**, **requirements**, **costs**.

### `optimize_healthcare_resources`

Allocate medical resources.

- **resources**, **locations**, **availability**, **demands**, **effectiveness**.

### `optimize_supply_chain_network`

Supply chain network design.

- **facilities**, **customers**, **fixed_costs**, **capacities**, **demands**, **transport_costs**.

## Nonlinear Programming

### `solve_nonlinear_program`

Solve nonlinear programming problems using SciPy optimizers.

- **variables** (`list[dict]`): Variable definitions with `name`, `initial_value`, `lower_bound`, `upper_bound`.
- **objective** (`str`): Python expression for objective function (e.g., `"x**2 + y**2"`).
- **constraints** (`list[dict]`): Constraints with `expression` (Python string) and `type` ("eq" or "ineq").
- **objective_sense** (`str`): "minimize" or "maximize".
- **method** (`str`): Solver method - "SLSQP", "trust-constr", "differential_evolution" (default: "SLSQP").

### `solve_minlp`

Solve Mixed-Integer Nonlinear Programming problems.

- **variables** (`list[dict]`): Variables with `name`, `type` ("continuous", "integer", "binary"), `initial_value`, bounds.
- **objective** (`str`): Python expression for objective.
- **constraints** (`list[dict]`): Nonlinear constraints.
- **objective_sense** (`str`): "minimize" or "maximize".

## Markov Decision Processes

### `solve_discrete_mdp`

Solve discrete Markov Decision Processes using value iteration or policy iteration.

- **states** (`list[str]`): State names.
- **actions** (`list[str]`): Action names.
- **transitions** (`dict`): Transition probabilities - `{(state, action): [(next_state, probability), ...]}`.
- **rewards** (`dict`): Rewards - `{(state, action, next_state): reward}`.
- **discount_factor** (`float`): Discount factor gamma (default: 0.9).
- **algorithm** (`str`): "value_iteration" or "policy_iteration" (default: "value_iteration").
- **max_iterations** (`int`): Maximum iterations (default: 1000).
- **tolerance** (`float`): Convergence tolerance (default: 1e-6).

### `optimize_equipment_replacement`

Optimize equipment replacement policy using MDP.

- **equipment_name** (`str`): Equipment identifier.
- **max_age** (`int`): Maximum equipment age.
- **purchase_cost** (`float`): Cost to purchase new equipment.
- **maintenance_costs** (`list[float]`): Maintenance cost per age.
- **salvage_values** (`list[float]`): Salvage value per age.
- **failure_probabilities** (`list[float]`): Failure probability per age.
- **discount_factor** (`float`): Discount factor (default: 0.9).

## Simulation & Optimization

### `optimize_simulation_parameters`

Optimize parameters using simulation-based black-box optimization.

- **parameter_names** (`list[str]`): Names of parameters to optimize.
- **parameter_bounds** (`list[tuple[float, float]]`): (min, max) bounds for each parameter.
- **simulation_function** (`str`): Python code defining simulation (must return float objective value).
- **objective_sense** (`str`): "minimize" or "maximize".
- **num_iterations** (`int`): Number of optimization iterations (default: 100).
- **num_simulations_per_eval** (`int`): Simulations per parameter evaluation (default: 100).

## Solver Selection

### `select_solver`

Automatically recommend the best solver for given problem characteristics.

- **problem_type** (`str`): "lp", "mip", "qp", "nlp", "minlp", "network", "routing", "scheduling", "stochastic".
- **num_variables** (`int`): Number of variables.
- **num_constraints** (`int`): Number of constraints.
- **has_integer_vars** (`bool`): Whether problem has integer variables.
- **has_binary_vars** (`bool`): Whether problem has binary variables.
- **is_quadratic** (`bool`): Whether objective/constraints are quadratic.
- **is_nonlinear** (`bool`): Whether problem is nonlinear.
- **sparsity** (`float`): Constraint matrix sparsity (0-1).

Returns recommended solver with rationale.

## System Metrics

### `get_system_metrics`

Get server performance metrics in Prometheus format.

Returns:

- Tool call counts and durations
- Error rates
- Active solves
- System resource usage

## Analysis Tools

### `get_model_statistics`

Get model size, sparsity, and type breakdown.

- **variables** (`list[dict]`): Variable definitions.
- **constraints** (`list[dict]`): Constraint definitions.

Returns statistics about problem structure.

### `find_alternative_solutions`

Find multiple near-optimal solutions for MIP problems.

- **variables**, **constraints**, **objective_coefficients**, **objective_sense**: Standard MIP inputs.
- **max_solutions** (`int`): Maximum solutions to return (default: 5).
- **gap_tolerance** (`float`): Accept solutions within this fraction of optimal (default: 0.01).

## Prompts

### `select_optimization_approach`

**Argument**: `problem_description` (str)

Help select the right tool/algorithm for a problem based on natural language description.

### `formulate_lp_problem`

**Argument**: `problem_description` (str)

Guide for extracting LP components (variables, constraints, objective) from natural language.

### `formulate_mip_problem`

**Argument**: `problem_description` (str)

Guide for formulating Mixed-Integer Programming problems from natural language.

### `formulate_network_problem`

**Argument**: `problem_description` (str)

Guide for formulating network flow problems from natural language.

### `formulate_scheduling_problem`

**Argument**: `problem_description` (str)

Guide for formulating scheduling problems from natural language.

### `interpret_lp_solution`

**Arguments**: `status`, `objective_value`, `variable_values`, `problem_context`

Interpret optimization results for decision makers in plain language.

### `interpret_sensitivity_analysis`

**Arguments**: `shadow_prices`, `reduced_costs`, `problem_context`

Explain shadow prices and reduced costs to decision makers.
