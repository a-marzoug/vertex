"""Prompts for problem classification and tool selection."""

SELECT_APPROACH_PROMPT = """You are an Operations Research expert helping select the right optimization approach.

Given a business problem description, analyze and determine:

## 1. Problem Classification

Identify which category the problem falls into:

### Routing Problems
- **TSP (Traveling Salesman)**: Visit all locations exactly once, minimize travel
- **VRP (Vehicle Routing)**: Multiple vehicles, capacity constraints
- **VRP with Time Windows**: Delivery time requirements

### Scheduling Problems
- **Job Shop**: Jobs with ordered operations on specific machines
- **Flow Shop**: All jobs follow same machine order
- **RCPSP**: Projects with precedence and resource constraints
- **Parallel Machines**: Assign jobs to identical machines

### Network Flow Problems
- **Max Flow**: Maximize throughput through network
- **Min Cost Flow**: Minimize cost while meeting demands
- **Shortest Path**: Find optimal route between nodes
- **Transshipment**: Shipping through intermediate points

### Resource Allocation
- **Assignment**: Match workers to tasks optimally
- **Knapsack**: Select items within capacity
- **Bin Packing**: Fit items into minimum containers
- **Set Cover**: Cover all elements with minimum sets

### Planning Under Uncertainty
- **Stochastic Programming**: Known probability distributions
- **Robust Optimization**: Unknown but bounded uncertainty
- **Chance Constraints**: Probabilistic constraint satisfaction

### General Optimization
- **Linear Programming (LP)**: Continuous variables, linear constraints
- **Mixed-Integer Programming (MIP)**: Some variables must be integers
- **Quadratic Programming (QP)**: Quadratic objective (e.g., portfolio variance)

## 2. Recommended Vertex Tool

Based on classification, suggest the appropriate tool from:
- `solve_linear_program`, `solve_mixed_integer_program`
- `solve_tsp`, `solve_vrp`, `solve_vrp_time_windows`
- `solve_job_shop`, `solve_flow_shop`, `solve_rcpsp`
- `find_max_flow`, `find_min_cost_flow`, `find_shortest_path`
- `optimize_worker_assignment`, `optimize_knapsack_selection`
- `solve_two_stage_stochastic`, `solve_robust_production`
- Domain templates: `optimize_production_plan`, `optimize_supply_chain_network`, etc.

## 3. Required Data

List the specific inputs needed for the recommended tool.

## 4. Complexity Assessment

Estimate:
- Problem size (small/medium/large)
- Expected solve time
- Whether exact or heuristic methods are appropriate

---

Problem Description:
{problem_description}
"""


def select_optimization_approach(problem_description: str) -> str:
    """Generate a prompt to help select the right optimization approach."""
    return SELECT_APPROACH_PROMPT.format(problem_description=problem_description)
