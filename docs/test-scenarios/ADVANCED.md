# Advanced Optimization Test Scenarios

This document contains test scenarios for stochastic, dynamic, and advanced optimization tools in heavy industry and finance.

## Two-Stage Stochastic Programming

### Scenario 1: Fashion Retail Seasonal Planning

**Prompt:**
"Determine order quantities for Winter Coats under uncertain weather.
**Scenarios**:

- **Mild Winter (30%)**: Demand 500
- **Cold Winter (50%)**: Demand 800
- **Harsh Winter (20%)**: Demand 1200
**Costs**:
- Production: $50/coat
- Retail Price: $150/coat
- Clearance Price (Unsold): $30/coat
- Lost Sale Penalty: $20/coat
Optimize the initial order quantity."
*(Expected Tool: `compute_two_stage_stochastic`)*

## Newsvendor Model

### Scenario 2: Perishable Food Inventory

**Prompt:**
"Optimize daily inventory for a supermarket's fresh seafood section.

- **Item**: Salmon Fillet
- **Cost**: $10/kg
- **Price**: $25/kg
- **Spoilage**: Zero value if unsold.
- **Demand**: N(Mean=50kg, Std=10kg)
Calculate optimal daily order."
*(Expected Tool: `compute_newsvendor`)*

## Dynamic Lot Sizing

### Scenario 3: Pharmaceutical Batch Production

**Prompt:**
"Plan production batches for a drug with shelf-life constraints.
**Forecast (Weeks)**: [10k, 50k, 20k, 60k, 40k] units.
**Setup Cost**: $5,000 (Sterilization & Setup).
**Holding Cost**: $0.10/unit/week (Storage).
**Production Cost**: $2/unit.
Find the optimal production schedule (Wagner-Whitin)."
*(Expected Tool: `compute_lot_sizing`)*

## Robust Optimization

### Scenario 4: Supply Chain Disruption Mitigation

**Prompt:**
"Plan component sourcing from Supplier A and B robust to disruptions.
**Nominal Supply**: A=1000, B=1000.
**Disruption**: Supplier A might drop to 500, B to 600.
**Budget**: Max 1 supplier fails.
**Costs**: A=$10, B=$12.
**Demand**: 1500 units.
Find robust order quantities."
*(Expected Tool: `solve_robust_optimization`)*

## Queueing Analysis

### Scenario 5: Cloud Load Balancer Sizing

**Prompt:**
"Size a server cluster.
**Requests**: 1000 req/sec.
**Server Capacity**: 120 req/sec.
**Target**: Avg wait time < 0.05 sec.
How many servers (M/M/c) are needed?"
*(Expected Tool: `analyze_queue_mmc`)*

## Monte Carlo Simulation

### Scenario 6: Semiconductor Yield Analysis

**Prompt:**
"Simulate chip yield revenue.
**Wafer Cost**: $5,000.
**Yield Rate**: N(80%, 5%).
**Chips per Wafer**: 500.
**Sale Price**: $20/chip.
Simulate 1,000 batches to find the probability of losing money."
*(Expected Tool: `simulate_production_monte_carlo`)*

## Network Design

### Scenario 7: Gas Pipeline Expansion

**Prompt:**
"Design a pipeline network to connect Gas Fields to Power Plants.
**Nodes**: Field1, Field2, Plant1, Plant2.
**Candidate Pipes**:

- F1->P1 ($10M)
- F1->P2 ($15M)
- F2->P1 ($12M)
- F2->P2 ($10M)
**Flow Requirement**: 100 units total.
Minimize construction cost."
*(Expected Tool: `design_network`)*

## Quadratic Assignment Problem

### Scenario 8: Motherboard Chip Placement

**Prompt:**
"Place 4 Chips (CPU, RAM, GPU, IO) on 4 Slots to minimize wire length * data rate.
**Data Rates**: CPU-RAM (High), CPU-GPU (High), IO-RAM (Low).
**Distances**: Grid 2x2.
Find optimal placement."
*(Expected Tool: `solve_quadratic_assignment`)*

## Mixed-Integer Nonlinear Programming

### Scenario 9: Process Engineering Design

**Prompt:**
"Design a heat exchanger network.
**Variables**:

- N: Number of exchangers (Integer)
- Area: Surface area (Continuous)
**Objective**: Minimize Cost = 1000*N + 50*Area^0.6
**Constraint**: N * Area >= 500 (Heat transfer req).
Find optimal N and Area."
*(Expected Tool: `solve_minlp`)*
