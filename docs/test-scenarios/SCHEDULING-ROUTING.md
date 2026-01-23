# Scheduling & Routing Test Scenarios

This document contains test scenarios for validating the Vertex MCP server's scheduling and routing capabilities in manufacturing and logistics.

## Traveling Salesman Problem (TSP)

### Scenario 1: PCB Manufacturing

**Prompt:**
"Optimize the movement of a robotic drill on a circuit board.
**Holes (Coordinates)**:

- A: (10, 10)
- B: (50, 20)
- C: (30, 40)
- D: (10, 50)
The robot must start at (0,0), visit all holes, and return. Minimize travel distance."
*(Expected Tool: `solve_tsp`)*

## Vehicle Routing Problem (VRP)

### Scenario 2: Grocery Home Delivery

**Prompt:**
"Plan the routes for 3 delivery vans (Capacity 20 crates each).
**Customer Orders**:

- C1: 5 crates
- C2: 8 crates
- C3: 12 crates
- C4: 15 crates
- C5: 10 crates
**Depot**: (0,0).
**Locations**: (Ask LLM to generate or assume grid).
Minimize total driving time."
*(Expected Tool: `solve_vrp`)*

## VRP with Time Windows

### Scenario 3: Technician Service Calls

**Prompt:**
"Schedule field technicians to repair equipment at 3 sites.
**Techs**: 2 available.
**Service Window**: 08:00 to 17:00 (0-540 mins).
**Jobs**:

- Site A: Duration 60m, Window [60, 180]
- Site B: Duration 90m, Window [120, 300]
- Site C: Duration 45m, Window [240, 480]
**Travel**: 30 mins between any site.
Create a valid schedule."
*(Expected Tool: `solve_vrp_time_windows`)*

## Job Shop Scheduling

### Scenario 4: Semiconductor Wafer Fabrication

**Prompt:**
"Schedule wafer lots through a fab.
**Machines**: Lithography (M1), Etching (M2), Deposition (M3).
**Jobs (Lots)**:

- **Lot A**: M1(4h) -> M2(2h) -> M3(3h)
- **Lot B**: M1(3h) -> M3(4h)
- **Lot C**: M2(5h) -> M1(2h) -> M3(2h)
Minimize the makespan (total time to finish all lots)."
*(Expected Tool: `solve_job_shop`)*

## Resource-Constrained Project Scheduling (RCPSP)

### Scenario 5: Construction Project

**Prompt:**
"Schedule the construction of a small building.
**Resources**: 1 Crane, 4 Laborers.
**Tasks**:

1. **Foundation**: 5 days, 1 Crane, 2 Laborers.
2. **Framework**: 4 days, 1 Crane, 3 Laborers. (After Foundation)
3. **Walls**: 6 days, 0 Crane, 4 Laborers. (After Framework)
4. **Roofing**: 3 days, 1 Crane, 2 Laborers. (After Framework)
5. **Plumbing**: 4 days, 0 Crane, 2 Laborers. (After Foundation)
Minimize project duration."
*(Expected Tool: `solve_rcpsp`)*

## Flexible Job Shop

### Scenario 6: Automated Manufacturing Cell

**Prompt:**
"Schedule production in a cell with redundant machines.
**Job 1**:

- Op 1: CNC1 (5m) OR CNC2 (6m)
- Op 2: Drill1 (3m)
**Job 2**:
- Op 1: CNC1 (4m) OR CNC2 (4m)
- Op 2: Drill1 (2m) OR Drill2 (3m)
Minimize makespan."
*(Expected Tool: `solve_flexible_job_shop`)*

## Bin Packing

### Scenario 7: Cargo Loading

**Prompt:**
"Load cargo crates into Air Freight containers (Max weight 1000kg).
**Crates**:

- 5x 450kg
- 3x 200kg
- 4x 300kg
- 2x 150kg
Minimize the number of containers used."
*(Expected Tool: `solve_bin_packing`)*
