# Advanced Optimization Test Problems

This file contains test problems for stochastic, dynamic, and advanced optimization tools.

## Two-Stage Stochastic Programming

### Problem 1: Production Under Demand Uncertainty
A manufacturer must decide production quantities before knowing actual demand.

**Input:**
```python
products = ["widget", "gadget"]
scenarios = [
    {"name": "low", "probability": 0.25, "demand": {"widget": 80, "gadget": 40}},
    {"name": "medium", "probability": 0.50, "demand": {"widget": 100, "gadget": 60}},
    {"name": "high", "probability": 0.25, "demand": {"widget": 120, "gadget": 80}},
]
production_costs = {"widget": 5, "gadget": 8}
shortage_costs = {"widget": 15, "gadget": 20}  # Lost sales penalty
holding_costs = {"widget": 1, "gadget": 2}     # Excess inventory cost
```

**Expected:** Optimal production balances expected shortage vs holding costs.

---

## Newsvendor Model

### Problem 2: Newspaper Vendor
Classic newsvendor: how many newspapers to order?

**Input:**
```python
selling_price = 2.50
cost = 1.00
salvage_value = 0.10  # Recycling value
mean_demand = 500
std_demand = 100
```

**Expected:**
- Critical ratio = (2.50 - 1.00) / (2.50 - 0.10) = 0.625
- Order quantity > mean (since underage cost > overage cost)

---

## Dynamic Lot Sizing (Wagner-Whitin)

### Problem 3: Basic Lot Sizing
10-period production planning.

**Input:**
```python
demands = [20, 50, 10, 50, 50, 10, 20, 40, 20, 30]
setup_cost = 100
holding_cost = 1
```

**Expected:**
- Batch production in select periods
- Total cost < sum of individual setups

---

## Robust Optimization

### Problem 4: Robust Production Planning
Protect against worst-case demand scenarios.

**Input:**
```python
products = ["A", "B"]
nominal_demand = {"A": 100, "B": 80}
demand_deviation = {"A": 20, "B": 15}
uncertainty_budget = 1.5  # At most 1.5 products deviate fully
production_costs = {"A": 10, "B": 12}
selling_prices = {"A": 25, "B": 30}
```

**Expected:** Conservative production protecting against demand drops.

### Problem 5: High Uncertainty Budget
Full protection (Gamma = number of products).

**Input:**
```python
uncertainty_budget = 2.0  # Both products can deviate
```

**Expected:** Most conservative solution.

---

## Queueing Analysis

### Problem 6: M/M/1 Call Center
Single agent handling calls.

**Input:**
```python
arrival_rate = 8  # calls per hour
service_rate = 10  # calls per hour capacity
```

**Expected:**
- Utilization = 0.8
- Avg queue length = 3.2
- Avg wait time = 0.4 hours

### Problem 7: M/M/c Hospital ER
Multiple doctors in emergency room.

**Input:**
```python
arrival_rate = 20  # patients per hour
service_rate = 10  # patients per hour per doctor
num_servers = 3    # doctors
```

**Expected:**
- Utilization = 0.667
- Lower wait times than single server

### Problem 8: Capacity Planning
Find minimum servers for target wait time.

**Test:** Vary num_servers from 2 to 5, find where avg_wait_time < 0.1

---

## Flow Shop Scheduling

### Problem 9: 3-Job 3-Machine Flow Shop
All jobs visit machines in same order.

**Input:**
```python
processing_times = [
    [3, 2, 2],  # Job 0: 3 on M0, 2 on M1, 2 on M2
    [2, 1, 4],  # Job 1
    [4, 3, 1],  # Job 2
]
```

**Expected:** Optimal sequence minimizes makespan.

### Problem 10: Johnson's Rule (2 machines)
Classic 2-machine flow shop.

**Input:**
```python
processing_times = [
    [5, 2],
    [1, 6],
    [9, 7],
    [3, 8],
    [10, 4],
]
```

**Expected:** Johnson's rule gives optimal sequence.

---

## Parallel Machine Scheduling

### Problem 11: Load Balancing
Distribute jobs across identical machines.

**Input:**
```python
job_durations = [5, 3, 7, 2, 4, 6, 1]
num_machines = 3
```

**Expected:** 
- Makespan ≈ ceil(sum/machines) = ceil(28/3) = 10
- Balanced load across machines

### Problem 12: Many Small Jobs
Test with many jobs.

**Input:**
```python
job_durations = [1, 2, 1, 3, 2, 1, 4, 2, 1, 3, 2, 1]
num_machines = 4
```

**Expected:** Near-optimal load balancing.

---

## Monte Carlo Simulation

### Problem 13: Newsvendor Risk Analysis
Evaluate profit distribution for a given order quantity.

**Input:**
```python
selling_price = 10
cost = 6
salvage_value = 2
order_quantity = 100
mean_demand = 100
std_demand = 20
num_simulations = 10000
```

**Expected:**
- Mean profit ≈ 336 (matches analytical)
- 5th percentile shows downside risk
- 95th percentile shows upside potential

### Problem 14: Production Risk Analysis
Multi-product profit distribution.

**Input:**
```python
products = ["A", "B"]
production_quantities = {"A": 100, "B": 80}
mean_demands = {"A": 100, "B": 80}
std_demands = {"A": 20, "B": 15}
selling_prices = {"A": 25, "B": 30}
production_costs = {"A": 10, "B": 12}
shortage_costs = {"A": 5, "B": 8}
```

**Expected:** Profit distribution with VaR metrics.

---

## Crew Scheduling

### Problem 15: Basic Shift Coverage
Schedule workers to cover shifts.

**Input:**
```python
workers = ["Alice", "Bob", "Carol", "Dave", "Eve"]
days = 7
shifts = ["morning", "evening"]
requirements = {
    "morning": [2, 2, 2, 2, 2, 1, 1],
    "evening": [2, 2, 2, 2, 2, 1, 1],
}
max_shifts_per_worker = 5
```

**Expected:** All shifts covered, no worker exceeds 5 shifts.

### Problem 16: Crew with Rest Constraints
Ensure minimum rest between shifts.

**Input:**
```python
workers = ["W1", "W2", "W3", "W4"]
days = 5
shifts = ["day", "night"]
requirements = {"day": [2, 2, 2, 2, 2], "night": [1, 1, 1, 1, 1]}
min_rest_between_shifts = 1  # Can't work night then day
```

**Expected:** No worker works night shift followed by day shift.



---

## Chance-Constrained Programming

### Problem 17: Service Level Guarantee
Production with 95% service level.

**Input:**
```python
products = ["A", "B"]
mean_demands = {"A": 100, "B": 80}
std_demands = {"A": 20, "B": 15}
production_costs = {"A": 10, "B": 12}
selling_prices = {"A": 25, "B": 30}
service_level = 0.95
```

**Expected:** Production > mean + 1.645 * std for each product.

### Problem 18: High Service Level
99% service level requirement.

**Input:**
```python
service_level = 0.99
```

**Expected:** Much higher production (safety stock).

---

## 2D Bin Packing

### Problem 19: Small Rectangles
Pack small rectangles into bins.

**Input:**
```python
rectangles = [
    {"name": "A", "width": 4, "height": 3},
    {"name": "B", "width": 2, "height": 5},
    {"name": "C", "width": 3, "height": 2},
    {"name": "D", "width": 2, "height": 2},
]
bin_width = 6
bin_height = 6
```

**Expected:** All fit in 1 bin with rotation.

### Problem 20: Manufacturing Cutting
Cut parts from sheet metal.

**Input:**
```python
rectangles = [
    {"name": "part1", "width": 10, "height": 5},
    {"name": "part2", "width": 8, "height": 4},
    {"name": "part3", "width": 6, "height": 6},
    {"name": "part4", "width": 4, "height": 8},
]
bin_width = 20
bin_height = 15
```

**Expected:** Minimize sheets used.

---

## Network Design

### Problem 21: Simple Network
Build minimum cost network.

**Input:**
```python
nodes = ["S", "A", "B", "T"]
potential_arcs = [
    {"source": "S", "target": "A"},
    {"source": "S", "target": "B"},
    {"source": "A", "target": "T"},
    {"source": "B", "target": "T"},
]
commodities = [{"name": "flow1", "source": "S", "sink": "T", "demand": 10}]
arc_fixed_costs = {"S->A": 100, "S->B": 80, "A->T": 90, "B->T": 70}
```

**Expected:** Open cheapest path S->B->T.

### Problem 22: Multi-Commodity Network
Multiple flows sharing capacity.

**Input:**
```python
commodities = [
    {"name": "flow1", "source": "S", "sink": "T", "demand": 10},
    {"name": "flow2", "source": "A", "sink": "B", "demand": 5},
]
```

**Expected:** May need to open more arcs.


---

## Quadratic Assignment Problem (QAP)

### Problem 23: Facility Layout
Assign 3 facilities to 3 locations.

**Input:**
```python
facilities = ["F1", "F2", "F3"]
locations = ["L1", "L2", "L3"]
flow_matrix = {
    "F1": {"F2": 10, "F3": 5},
    "F2": {"F1": 10, "F3": 8},
    "F3": {"F1": 5, "F2": 8},
}
distance_matrix = {
    "L1": {"L2": 1, "L3": 3},
    "L2": {"L1": 1, "L3": 2},
    "L3": {"L1": 3, "L2": 2},
}
```

**Expected:** High-flow pairs assigned to close locations.

---

## Steiner Tree

### Problem 24: Network with Steiner Node
Connect 4 terminals using optional hub.

**Input:**
```python
nodes = ["A", "B", "C", "D", "S"]  # S is Steiner node
edges = [
    {"source": "A", "target": "S", "weight": 1},
    {"source": "B", "target": "S", "weight": 1},
    {"source": "C", "target": "S", "weight": 1},
    {"source": "D", "target": "S", "weight": 1},
    {"source": "A", "target": "B", "weight": 3},
]
terminals = ["A", "B", "C", "D"]
```

**Expected:** Use Steiner node S, total weight = 4.

---

## Multi-Echelon Inventory

### Problem 25: Two-Tier Supply Chain
DC supplying two stores.

**Input:**
```python
locations = ["DC", "Store1", "Store2"]
parent = {"DC": None, "Store1": "DC", "Store2": "DC"}
demands = {"DC": 0, "Store1": 100, "Store2": 80}
lead_times = {"DC": 5, "Store1": 2, "Store2": 2}
holding_costs = {"DC": 1, "Store1": 2, "Store2": 2}
service_levels = {"DC": 0.99, "Store1": 0.95, "Store2": 0.95}
```

**Expected:** Base stock levels with safety stock for service levels.


---

## Quadratic Programming

### Problem 26: Simple QP
Minimize quadratic function with equality constraint.

**Input:**
```python
variables = ["x", "y"]
Q = [[2, 0], [0, 2]]  # min x^2 + y^2
c = [0, 0]
A_eq = [[1, 1]]
b_eq = [1]  # x + y = 1
```

**Expected:** x = y = 0.5, objective = 0.5

### Problem 27: QP with Bounds
Minimize with variable bounds.

**Input:**
```python
variables = ["x", "y", "z"]
Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
c = [-2, -3, -1]
lower_bounds = [0, 0, 0]
upper_bounds = [1, 1, 1]
```

---

## Portfolio Optimization (Mean-Variance)

### Problem 28: Minimum Variance Portfolio
Find portfolio with lowest risk.

**Input:**
```python
assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
expected_returns = [0.12, 0.10, 0.11, 0.14]
covariance_matrix = [
    [0.04, 0.01, 0.015, 0.02],
    [0.01, 0.03, 0.01, 0.015],
    [0.015, 0.01, 0.025, 0.012],
    [0.02, 0.015, 0.012, 0.05],
]
```

**Expected:** Diversified portfolio minimizing variance.

### Problem 29: Target Return Portfolio
Minimize variance for 11% target return.

**Input:**
```python
target_return = 0.11
```

**Expected:** Portfolio achieving 11% return with minimum variance.

### Problem 30: Risk-Aversion Portfolio
Maximize utility with risk aversion.

**Input:**
```python
risk_aversion = 2.0
```

**Expected:** Balance between return and risk based on aversion.
