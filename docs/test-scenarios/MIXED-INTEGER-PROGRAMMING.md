# Mixed-Integer Programming Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's MIP capabilities using complex industrial problems.

## Assignment Problems

### Scenario 1: Vehicle-Route Assignment

**Prompt:**
"Assign 5 delivery trucks to 5 regions based on operating costs.
**Trucks**: T1, T2, T3, T4, T5 (Different fuel efficiencies).
**Regions**: R1, R2, R3, R4, R5 (Different terrains).
**Cost Matrix** (Fuel Cost):

- T1: R1(100), R2(120), R3(140), R4(110), R5(130)
- T2: R1(110), R2(115), R3(125), R4(105), R5(120)
- T3: R1(130), R2(140), R3(150), R4(135), R5(145)
- T4: R1(90), R2(100), R3(110), R4(95), R5(105)
- T5: R1(115), R2(125), R3(130), R4(120), R5(125)
Find the assignment that minimizes total fuel cost."
*(Expected Tool: `optimize_worker_assignment`)*

## Knapsack Problems

### Scenario 2: IT Project Selection

**Prompt:**
"Select projects for the annual IT portfolio given a budget of $5M and 10 engineers.
**Projects:**

- P1: Cost $1M, Eng 2, ROI $2.5M
- P2: Cost $2M, Eng 4, ROI $4.5M
- P3: Cost $1.5M, Eng 3, ROI $3.0M
- P4: Cost $0.5M, Eng 1, ROI $1.2M
- P5: Cost $2.5M, Eng 5, ROI $6.0M
Maximize ROI subject to Budget and Engineer constraints."
*(Expected Tool: `solve_mixed_integer_program`)*

## Facility Location Problems

### Scenario 3: Edge Server Placement

**Prompt:**
"Determine where to place Edge Servers to minimize latency for 5 smart cities.
**Candidates**: Site A, Site B, Site C.
**Fixed Costs**: A($50k), B($60k), C($45k).
**Latency Cost to City ($/ms)**:

- Site A: C1(5), C2(10), C3(15), C4(20), C5(25)
- Site B: C1(20), C2(15), C3(10), C4(5), C5(10)
- Site C: C1(15), C2(20), C3(25), C4(10), C5(5)
Constraint: Must serve all cities.
Minimize Fixed + Latency Costs."
*(Expected Tool: `optimize_facility_locations`)*

## Generic MIP Problems

### Scenario 4: Power Generation Unit Commitment

**Prompt:**
"Schedule 3 power plants to meet a load of 500 MW.
**Plants:**

- P1: Min 50 MW, Max 300 MW. Fixed Start Cost $1000. Variable $20/MW.
- P2: Min 40 MW, Max 250 MW. Fixed Start Cost $800. Variable $25/MW.
- P3: Min 30 MW, Max 150 MW. Fixed Start Cost $500. Variable $30/MW.
**Constraint**: At least 2 plants must be active for reliability.
Find the active plants and their output levels to minimize cost."
*(Expected Tool: `solve_mixed_integer_program`)*

### Scenario 5: Airline Pilot Bidding System

**Prompt:**
"Assign pilots to monthly schedules based on seniority bids.
**Pilots**: 4 Captains.
**Schedules**: 4 Lines of Flying (L1, L2, L3, L4).
**Preferences (Penalty Points, 1=Top Choice)**:

- C1: L1(1), L2(5), L3(10), L4(100)
- C2: L1(2), L2(1), L3(5), L4(10)
- C3: L1(100), L2(5), L3(1), L4(2)
- C4: L1(10), L2(100), L3(2), L4(1)
**Constraint**: C1 is senior, must get their 1st or 2nd choice.
Minimize total penalty."
*(Expected Tool: `solve_mixed_integer_program`)*
