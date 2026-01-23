# Combinatorial Optimization Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's combinatorial optimization capabilities (packing, covering, coloring) using realistic industrial contexts.

## Bin Packing

### Scenario 1: Cloud Container Allocation

**Prompt:**
"Optimize the placement of microservices onto physical servers (nodes) to minimize the number of servers used.
**Node Capacity**: 128 GB RAM.
**Services (RAM Requirement):**

- Service A: 64 GB
- Service B: 32 GB
- Service C: 16 GB
- Service D: 8 GB
- Service E: 48 GB
How many nodes are needed and what is the placement configuration?"
*(Expected Tool: `solve_bin_packing`)*

### Scenario 2: Logistics Container Loading

**Prompt:**
"Pack shipping pallets into standard 40ft containers.
**Container Weight Limit**: 25,000 kg.
**Pallet Weights**: [2000, 5000, 5000, 12000, 8000, 4000, 2000, 3000].
Minimize the number of containers required."
*(Expected Tool: `solve_bin_packing`)*

## Set Covering

### Scenario 3: Airline Crew Pairing

**Prompt:**
"Select the minimum number of crew pairings to cover all flight legs.
**Flight Legs**: {L1, L2, L3, L4, L5}
**Candidate Pairings (Cost 1 each):**

- P1: {L1, L2}
- P2: {L2, L3}
- P3: {L3, L4, L5}
- P4: {L1, L5}
- P5: {L4}
Which pairings should be selected?"
*(Expected Tool: `compute_set_cover`)*

### Scenario 4: Emergency Siren Placement

**Prompt:**
"Determine the optimal locations for emergency sirens to ensure 100% coverage of all city districts.
**Districts**: {D1, D2, D3, D4, D5, D6}
**Potential Locations:**

- Loc A: Covers {D1, D2, D3} (Cost $10k)
- Loc B: Covers {D2, D4, D6} (Cost $12k)
- Loc C: Covers {D3, D5} (Cost $8k)
- Loc D: Covers {D4, D5, D6} (Cost $10k)
- Loc E: Covers {D1, D6} (Cost $9k)
Minimize total cost."
*(Expected Tool: `compute_set_cover`)*

## Graph Coloring

### Scenario 5: Telecom Frequency Assignment

**Prompt:**
"Assign frequencies to cell towers such that adjacent towers (which interfere) do not share the same frequency. Minimize the number of unique frequencies used.
**Towers (Nodes)**: {T1, T2, T3, T4, T5}
**Interference (Edges)**:

- T1-T2, T1-T3
- T2-T3, T2-T4
- T3-T5
- T4-T5
What is the minimum spectrum required?"
*(Expected Tool: `compute_graph_coloring`)*

### Scenario 6: Compiler Register Allocation

**Prompt:**
"Assign physical CPU registers to variables based on their liveness interference.
**Variables**: {v1, v2, v3, v4, v5, v6}
**Interference Graph**:

- v1 interferes with v2, v3
- v2 interferes with v1, v4, v5
- v3 interferes with v1, v6
- v4 interferes with v2, v5
- v5 interferes with v2, v4, v6
Minimize the number of registers."
*(Expected Tool: `compute_graph_coloring`)*

## Cutting Stock

### Scenario 7: Steel Coil Slitting

**Prompt:**
"Optimize the cutting of master steel coils into narrower strips for customer orders.
**Master Coil Width**: 1500 mm.
**Customer Orders**:

- 15 coils of 600 mm
- 20 coils of 450 mm
- 10 coils of 300 mm
Minimize the total number of master coils used and waste."
*(Expected Tool: `compute_cutting_stock`)*

### Scenario 8: Paper Roll Trim Loss

**Prompt:**
"A paper mill produces 200-inch wide rolls.
Orders:

- 50 rolls of 80 inches
- 30 rolls of 60 inches
- 40 rolls of 50 inches
Find the cutting patterns that minimize trim loss."
*(Expected Tool: `compute_cutting_stock`)*

## Knapsack Variants

### Scenario 9: Capital Budgeting

**Prompt:**
"Select capital projects to fund within a $10M budget to maximize ROI.
**Projects:**

- Project Alpha: Cost $3M, NPV $5M
- Project Beta: Cost $4M, NPV $6M
- Project Gamma: Cost $2M, NPV $3.5M
- Project Delta: Cost $5M, NPV $8M
- Project Epsilon: Cost $1M, NPV $1.2M
Maximize total Net Present Value (NPV)."
*(Expected Tool: `optimize_knapsack_selection`)*
