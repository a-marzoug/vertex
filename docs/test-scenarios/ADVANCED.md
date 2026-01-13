# Advanced Optimization Test Scenarios

This document contains test scenarios for stochastic, dynamic, and advanced optimization tools. Feed these prompts to an LLM to test the tools.

## Two-Stage Stochastic Programming

### Scenario 1: Production Under Demand Uncertainty

**Prompt:**
"I need to decide production quantities for 'widgets' and 'gadgets' *before* knowing the actual demand.
**Scenarios:**

- **Low Demand (25% prob)**: Widget=80, Gadget=40
- **Medium Demand (50% prob)**: Widget=100, Gadget=60
- **High Demand (25% prob)**: Widget=120, Gadget=80

**Costs:**

- **Production**: Widget $5, Gadget $8
- **Shortage (Lost Sales)**: Widget $15, Gadget $20
- **Holding (Excess)**: Widget $1, Gadget $2

What are the optimal production quantities to minimize expected total cost?"

*(Expected Tool: `compute_two_stage_stochastic`)*

---

## Newsvendor Model

### Scenario 2: Newspaper Vendor

**Prompt:**
"I sell newspapers.

- **Selling Price**: $2.50
- **Cost**: $1.00
- **Salvage Value** (Unsold): $0.10
- **Daily Demand**: Normally distributed (Mean=500, Std=100)

How many papers should I order to maximize expected profit?"

*(Expected Tool: `compute_newsvendor`)*

---

## Dynamic Lot Sizing

### Scenario 3: Wagner-Whitin Lot Sizing

**Prompt:**
"Plan production for the next 10 periods to minimize setup + holding costs.
**Demands**: [20, 50, 10, 50, 50, 10, 20, 40, 20, 30]
**Setup Cost**: $100 per batch.
**Holding Cost**: $1 per unit per period.

What is the optimal production schedule?"

*(Expected Tool: `compute_lot_sizing`)*

---

## Robust Optimization

### Scenario 4: Robust Production

**Prompt:**
"Plan production for products A and B under uncertainty.
**Nominal Demand**: A=100, B=80.
**Deviation**: Demand might drop by 20 for A, and 15 for B.
**Uncertainty Budget (Gamma)**: 1.5 (Protect against up to 1.5 products seeing worst-case demand).

**Financials**:

- Cost: A=$10, B=$12
- Price: A=$25, B=$30

Find the robust production quantities."

*(Expected Tool: `solve_robust_optimization`)*

---

## Queueing Analysis

### Scenario 5: M/M/1 Call Center

**Prompt:**
"Analyze a single-agent call center.

- **Arrival Rate**: 8 calls/hour
- **Service Rate**: 10 calls/hour

Calculate utilization, average queue length, and average wait time."

*(Expected Tool: `analyze_queue_mm1`)*

---

### Scenario 6: M/M/c Hospital ER

**Prompt:**
"Analyze an Emergency Room with **3 doctors**.

- **Arrival Rate**: 20 patients/hour
- **Service Rate**: 10 patients/hour per doctor

Calculate the Key Performance Indicators (Utilization, Wait Time)."

*(Expected Tool: `analyze_queue_mmc`)*

---

## Flow Shop Scheduling

### Scenario 7: 3-Job 3-Machine Flow Shop

**Prompt:**
"Schedule 3 jobs through 3 machines (M0 -> M1 -> M2).
**Processing Times (Job `i` on M0, M1, M2):**

- Job 0: [3, 2, 2]
- Job 1: [2, 1, 4]
- Job 2: [4, 3, 1]

Find the sequence that minimizes the makespan."

*(Expected Tool: `compute_flow_shop`)*

---

## Parallel Machine Scheduling

### Scenario 8: Load Balancing

**Prompt:**
"Distribute these jobs across **3 identical machines** to minimize the makespan (finish time of the last machine).
**Job Durations**: [5, 3, 7, 2, 4, 6, 1].

What is the assignment?"

*(Expected Tool: `compute_parallel_machines`)*

---

## Monte Carlo Simulation

### Scenario 9: Newsvendor Risk Simulation

**Prompt:**
"Simulate 10,000 days for a newsvendor scenario to estimate risk.

- Price: $10, Cost: $6, Salvage: $2
- Order Quantity: 100
- Demand: Normal(Mean=100, Std=20)

Report the Mean Profit and Value at Risk (5th percentile)."

*(Expected Tool: `simulate_newsvendor_monte_carlo`)*

---

### Scenario 10: Production Risk Analysis

**Prompt:**
"Simulate profit distribution for 2 products (A, B).
**Plan**: Produce 100 of A, 80 of B.
**Demand**:

- A: Normal(100, 20)
- B: Normal(80, 15)
**Financials**:
- Costs: A=10, B=12
- Prices: A=25, B=30
- Shortage Penalty: A=5, B=8

Run a Monte Carlo simulation to analyze profit variability."

*(Expected Tool: `simulate_production_monte_carlo`)*

---

## Crew Scheduling

### Scenario 11: Shift Coverage

**Prompt:**
"Schedule 5 workers (Alice, Bob, Carol, Dave, Eve) for a 7-day week.
**Requirements:**

- Morning Shift: [2, 2, 2, 2, 2, 1, 1] workers needed (Mon-Sun)
- Evening Shift: [2, 2, 2, 2, 2, 1, 1] workers needed
**Constraint**: Max 5 shifts per worker per week.

Generate a valid schedule."

*(Expected Tool: `schedule_crew`)*

---

## Chance-Constrained Programming

### Scenario 12: Service Level Guarantee

**Prompt:**
"Determine production quantities to guarantee a **95% Service Level** (95% probability of meeting demand).

- **Product A**: Demand ~ N(100, 20)
- **Product B**: Demand ~ N(80, 15)

Ensure production satisfies this probabilistic constraint at minimum cost."

*(Expected Tool: `solve_chance_constrained_production`)*

---

## 2D Bin Packing

### Scenario 13: Rectangle Packing

**Prompt:**
"Pack these rectangles into a **6x6 bin**.

- A: 4x3
- B: 2x5
- C: 3x2
- D: 2x2

Can they all fit? Show coordinates."

*(Expected Tool: `pack_rectangles_2d`)*

---

## Network Design

### Scenario 14: Network Synthesis

**Prompt:**
"Design a minimum cost network to route 10 units of flow from S to T.
**Candidate Arcs (Fixed Cost to build):**

- S->A ($100)
- S->B ($80)
- A->T ($90)
- B->T ($70)

Which arcs should be built?"

*(Expected Tool: `design_network`)*

---

## Quadratic Assignment Problem (QAP)

### Scenario 15: Facility Layout

**Prompt:**
"Assign 3 Facilities (F1, F2, F3) to 3 Locations (L1, L2, L3) to minimize Flow * Distance.

**Flow Matrix (Between Facilities):**

- F1-F2: 10, F1-F3: 5
- F2-F3: 8

**Distance Matrix (Between Locations):**

- L1-L2: 1, L1-L3: 3
- L2-L3: 2

Find the optimal assignment."

*(Expected Tool: `solve_quadratic_assignment`)*

---

## Steiner Tree

### Scenario 16: Optimal Connectivity

**Prompt:**
"Connect Terminals {A, B, C, D} using minimum edge weight. You may use the optional Steiner node 'S'.
**Edges:**

- A-S (1), B-S (1), C-S (1), D-S (1)
- A-B (3)

Find the Minimum Steiner Tree."

*(Expected Tool: `find_steiner_tree`)*

---

## Multi-Echelon Inventory

### Scenario 17: Supply Chain Stock

**Prompt:**
"Optimize inventory for a 2-tier chain: DC -> Store1, Store2.
**Data:**

- **Demands**: Store1=100, Store2=80
- **Lead Times**: DC=5 days, Stores=2 days
- **Service Level**: DC=99%, Stores=95%

Calculate safety stocks and reorder points."

*(Expected Tool: `optimize_multi_echelon_inventory`)*

---

## Quadratic Programming

### Scenario 18: Simple QP

**Prompt:**
"Minimize the function: `x^2 + y^2`
**Subject to**: `x + y = 1`

Find x and y."

*(Expected Tool: `solve_quadratic_program`)*

---

## Portfolio Optimization (Mean-Variance)

### Scenario 19: Minimum Variance

**Prompt:**
"Find the portfolio allocation that minimizes risk (variance) for these assets:
Assets: [AAPL, GOOGL, MSFT, AMZN]
Returns: [0.12, 0.10, 0.11, 0.14]
**Covariance Matrix**:
[[0.04, 0.01, 0.015, 0.02],
 [0.01, 0.03, 0.01, 0.015],
 [0.015, 0.01, 0.025, 0.012],
 [0.02, 0.015, 0.012, 0.05]]

Target Return: None (Global Min Variance)."

*(Expected Tool: `optimize_portfolio_qp`)*
