# Network Optimization Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's network optimization capabilities (flows, paths, trees). Feed these prompts to an LLM to test the tools.

## Maximum Flow

### Scenario 1: Simple Pipeline Max Flow

**Prompt:**
"Calculate the maximum flow rate possible from Source (S) to Sink (T) in this pipeline network.

**Network Connections (Capacity):**

- **S -> A**: Cap 10
- **S -> B**: Cap 8
- **A -> B**: Cap 5
- **A -> T**: Cap 7
- **B -> T**: Cap 10

What is the max flow amount?"

*(Expected Tool: `find_max_flow`)*

---

### Scenario 2: Bipartite Matching

**Prompt:**
"Find the maximum number of job matches possible. We have 3 workers (W1, W2, W3) and 3 jobs (J1, J2, J3).
Each worker can only do specific jobs (Capacity = 1).

**Eligible Matches:**

- **W1** can do: J1, J2
- **W2** can do: J2, J3
- **W3** can do: J1, J3

Construct a max flow problem to find the maximum matching."

*(Expected Tool: `find_max_flow`)*

---

## Minimum Cost Flow

### Scenario 3: Transportation Network

**Prompt:**
"I need to ship 100 units from a Factory to a Warehouse at minimum cost.
The network has two intermediate hubs.

**Routes (Capacity, Cost/Unit):**

- **Factory -> Hub1**: Cap 60, Cost $2
- **Factory -> Hub2**: Cap 80, Cost $4
- **Hub1 -> Warehouse**: Cap 70, Cost $3
- **Hub2 -> Warehouse**: Cap 50, Cost $1

How should I route the 100 units?"

*(Expected Tool: `find_min_cost_flow`)*

---

### Scenario 4: Task Assignment as Flow

**Prompt:**
"Assign 3 workers to 3 tasks to minimize total cost.
Model this as a min-cost flow problem (Supply=1 per worker, Demand=-1 per task).

**Assignment Costs:**

- **W1**: T1($5), T2($8), T3($6)
- **W2**: T1($9), T2($4), T3($7)
- **W3**: T1($6), T2($7), T3($3)

Find the optimal flow."

*(Expected Tool: `find_min_cost_flow`)*

---

## Shortest Path

### Scenario 5: Simple Graph Path

**Prompt:**
"Find the shortest path from Node A to Node D.

**Links & Costs:**

- **A -> B**: Cost 1
- **A -> C**: Cost 4
- **B -> C**: Cost 2
- **B -> D**: Cost 5
- **C -> D**: Cost 1

What is the path and total cost?"

*(Expected Tool: `find_shortest_path`)*

---

### Scenario 6: Road Network Commute

**Prompt:**
"I need the fastest route from 'Home' to 'Work'.

**Road Segment Times (minutes):**

- **Home -> A**: 10
- **Home -> B**: 15
- **A -> B**: 5
- **A -> C**: 20
- **B -> C**: 10
- **C -> Work**: 5

Which sequence of roads takes the least time?"

*(Expected Tool: `find_shortest_path`)*

---

## Minimum Spanning Tree

### Scenario 7: Connecting Cities

**Prompt:**
"We need to connect 4 cities (A, B, C, D) with fiber optic cables.
We want to minimize the total length of cable used to ensure everyone is connected.

**Possible Connections (Length):**

- **A-B**: 1
- **A-C**: 4
- **B-C**: 2
- **B-D**: 5
- **C-D**: 3

Which connections should we build?"

*(Expected Tool: `find_minimum_spanning_tree`)*

---

### Scenario 8: Office Infrastructure

**Prompt:**
"Connect 5 offices (O1..O5) to the customized local network with minimum cabling cost.

**Cabling Costs:**

- **O1-O2**: 10
- **O1-O3**: 15
- **O2-O3**: 8
- **O2-O4**: 12
- **O3-O4**: 6
- **O3-O5**: 9
- **O4-O5**: 7

Tree must span all nodes."

*(Expected Tool: `find_minimum_spanning_tree`)*

---

## Multi-Commodity Flow

### Scenario 9: Shared Network Routing

**Prompt:**
"Route two different products (A and B) through the same network without exceeding capacities.

**Network:**

- **S1 -> Hub**: Cap 50
- **S2 -> Hub**: Cap 50
- **Hub -> D1**: Cap 40
- **Hub -> D2**: Cap 40

**Demands:**

- **Product A**: From S1 to D1, needs 20 units.
- **Product B**: From S2 to D2, needs 30 units.

Find a feasible flow for both."

*(Expected Tool: `find_multi_commodity_flow`)*

---

### Scenario 10: Bottleneck Competition

**Prompt:**
"Three commodities must pass through a single shared pipe (Capacity 100).

**Demands:**

- **C1**: 40 units
- **C2**: 35 units
- **C3**: 30 units
*(Total Demand = 105)*

Since demand > capacity, find the flow that maximizes total throughput (or satisfies as much as possible)."

*(Expected Tool: `find_multi_commodity_flow`)*

---

## Transshipment

### Scenario 11: Multi-Echelon Supply Chain

**Prompt:**
"Optimize the shipment plan.
**Structure:** Factories -> Warehouses -> Stores.

**Supplies:**

- **Factory1**: 100 capacity
- **Factory2**: 80 capacity

**Demands:**

- **Store1**: 60
- **Store2**: 70
- **Store3**: 50

**Transshipment Nodes:** Warehouse1, Warehouse2.

**(Implicitly assume costs/arcs would be provided or LLM should ask for them. For testing, ask LLM to generate random valid costs if needed, or provide simple ones:)**
Assume unit cost = 1 for all Factory->Warehouse links, and cost = 2 for all Warehouse->Store links. Find the flow."

*(Expected Tool: `solve_transshipment`)*

---

### Scenario 12: 4-Tier Distribution

**Prompt:**
"Solve a complex transshipment problem with 4 tiers:
Tier 1 (2 Plants) -> Tier 2 (3 Regional WH) -> Tier 3 (5 Local DCs) -> Tier 4 (10 Stores).
Objective: Minimize total shipping cost while meeting store demand.
(Ask the LLM to set up a mock instance of this problem)."

*(Expected Tool: `solve_transshipment`)*
