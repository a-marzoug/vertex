# Combinatorial Optimization Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's combinatorial optimization capabilities (packing, covering, coloring). Feed these prompts to an LLM to test the tools.

## Bin Packing

### Scenario 1: Simple Packing

**Prompt:**
"Pack these items into the fewest number of bins possible.
**Bin Capacity**: 60 units.

**Items (Weight):**

- A: 40
- B: 30
- C: 35
- D: 25
- E: 20

How many bins are needed and what is the packing configuration?"

*(Expected Tool: `solve_bin_packing`)*

---

### Scenario 2: Tight Fit

**Prompt:**
"I have items with weights: [50, 50, 50, 50, 25, 25, 25, 25].
My bin capacity is **100**.
Can you pack these into exactly 3 bins?"

*(Expected Tool: `solve_bin_packing`)*

---

### Scenario 3: First Fit Challenge

**Prompt:**
"Optimize the packing of these items: [7, 5, 3, 8, 6, 4, 2, 9].
**Bin Capacity**: 10.
Compare a simple strategy vs the optimal one. Minimize the bin count."

*(Expected Tool: `solve_bin_packing`)*

---

## Set Covering

### Scenario 4: Simple Set Cover

**Prompt:**
"I need to cover all elements in the Universe {1, 2, 3, 4, 5} using the minimum cost selection of sets.

**Available Sets:**

- **S1**: {1, 2, 3} (Cost 5)
- **S2**: {2, 4} (Cost 3)
- **S3**: {3, 4} (Cost 3)
- **S4**: {4, 5} (Cost 4)

Which sets should I pick?"

*(Expected Tool: `compute_set_cover`)*

---

### Scenario 5: Fire Station Coverage

**Prompt:**
"Select the best locations for fire stations to ensure all 6 neighborhoods (N1-N6) are covered at minimum cost.

**Station Options:**

- **Station A**: Covers N1, N2, N3. (Cost 100)
- **Station B**: Covers N2, N4, N5. (Cost 80)
- **Station C**: Covers N3, N5, N6. (Cost 90)
- **Station D**: Covers N1, N6. (Cost 70)

What is the optimal selection?"

*(Expected Tool: `compute_set_cover`)*

---

### Scenario 6: Airline Crew Scheduling

**Prompt:**
"Assign crew pairings to cover all Flights {101, 102, 103, 104, 105, 106} with minimum pairings used (assume cost=1 per pairing).

**Pairing Options:**

- **P1**: Flights {101, 102, 103}
- **P2**: Flights {102, 104}
- **P3**: Flights {103, 105, 106}
- **P4**: Flights {104, 106}

Find the cover."

*(Expected Tool: `compute_set_cover`)*

---

## Graph Coloring

### Scenario 7: Topology Coloring

**Prompt:**
"Determine the minimum number of colors needed to color this graph such that no connected nodes share a color.

**Graph Topology:**

- a 'triangle' formed by A-B, B-C, C-A.
- C is also connected to D.

What is the chromatic number?"

*(Expected Tool: `compute_graph_coloring`)*

---

### Scenario 8: Bipartite Check

**Prompt:**
"Color the graph with Nodes {1..6} and Edges:
1-4, 1-5, 2-4, 2-6, 3-5, 3-6.
Minimize the number of colors."

*(Expected Tool: `compute_graph_coloring`)*

---

### Scenario 9: Exam Scheduling

**Prompt:**
"Schedule exams into the minimum number of time slots so that no conflicting exams occur at the same time.

**Conflicts (Students share these classes):**

- **Math**: Physics, CS
- **Physics**: Math, Chemistry
- **CS**: Math, English
- **Chemistry**: Physics
- **English**: CS

How many slots are required?"

*(Expected Tool: `compute_graph_coloring`)*

---

### Scenario 10: Register Allocation

**Prompt:**
"In a compiler, live variables interfering with each other need different physical registers.
**Interference Graph:**

- **a**: intersects b, c
- **b**: intersects a, d
- **c**: intersects a, d
- **d**: intersects b, c

Find the minimum registers needed (chromatic number)."

*(Expected Tool: `compute_graph_coloring`)*

---

## Cutting Stock

### Scenario 11: Lumber Cutting

**Prompt:**
"We have stock lumber of length **100cm**.
**Orders:**

- 3 pieces of 45cm
- 4 pieces of 30cm
- 2 pieces of 25cm

How should we cut the stock to use the fewest total boards?"

*(Expected Tool: `compute_cutting_stock`)*

---

### Scenario 12: Steel Bar Optimization

**Prompt:**
"Minimize waste when cutting standard **12m** steel bars into:

- 10 bars of 3m
- 8 bars of 4m
- 6 bars of 5m

What is the cutting pattern?"

*(Expected Tool: `compute_cutting_stock`)*

---

### Scenario 13: Paper Roll Trim Loss

**Prompt:**
"A paper mill produces master rolls of width **200cm**.
Customers ordered:

- 20 rolls of 50cm
- 15 rolls of 60cm
- 10 rolls of 80cm

Find the combination of cutting patterns that minimizes the number of master rolls used."

*(Expected Tool: `compute_cutting_stock`)*

---

## Knapsack Variants

### Scenario 14: Classic 0/1 Knapsack

**Prompt:**
"Solve the 0/1 Knapsack problem. **Capacity = 50**.

**Items:**

- **A**: Val 60, Wgt 10
- **B**: Val 100, Wgt 20
- **C**: Val 120, Wgt 30

Maximize total value."

*(Expected Tool: `optimize_knapsack_selection`)*

---

### Scenario 15: Multiple Knapsacks

**Prompt:**
"We have **2 knapsacks**, each with capacity **15**.
**Items to pack (Value, Weight):**

- A: (10, 5)
- B: (15, 8)
- C: (12, 6)
- D: (8, 4)
- E: (20, 10)

Allocate items to knapsacks to maximize total value."

*(Expected Tool: `optimize_knapsack_selection` or `solve_bin_packing` if interpreted as packing, but bin packing minimizes bins. For value max with multiple bins, use `solve_mixed_integer_program` or heuristic.)*

*(Correction: The standard knapsack tool is single-container. This prompts a check if the server supports `multiple_knapsack` or if it requires MIP)*

---

### Scenario 16: Bounded Knapsack

**Prompt:**
"Select items to maximize value. Capacity: **100**.
**Items (Multiple copies available):**

- **Type A**: Val 10, Wgt 5 (Max 5 items)
- **Type B**: Val 15, Wgt 8 (Max 3 items)
- **Type C**: Val 20, Wgt 12 (Max 4 items)

What is the optimal mix?"

*(Expected Tool: `optimize_knapsack_selection` or MIP)*
