# Analysis Tools Test Scenarios

This document contains test scenarios for sensitivity analysis, infeasibility diagnosis, and constraint programming. Feed these prompts to an LLM to test the tools.

## Sensitivity Analysis

### Scenario 1: Shadow Prices

**Prompt:**
"Maximize the objective function `3x + 4y` subject to:

1. `x + 2y <= 14` (Resource A)
2. `3x - y >= 0` (Resource B)
3. `x - y <= 2` (Resource C)

After solving, analyze the sensitivity. What are the shadow prices for the constraints? Which resources are binding?"

*(Expected Tool: `analyze_lp_sensitivity`)*

---

### Scenario 2: Reduced Costs

**Prompt:**
"Using the same problem:
Maximize `3x + 4y` subject to `x + 2y <= 14`, `3x - y >= 0`, `x - y <= 2`.

Report the reduced costs for the variables x and y."

*(Expected Tool: `analyze_lp_sensitivity`)*

---

## What-If Analysis

### Scenario 3: Resource Variation

**Prompt:**
"For the problem: Maximize `3x + 4y` subject to `x + 2y <= 10` (and non-negativity).
Perform a What-If analysis on the constraint `x + 2y <= 10`.
Vary the Right-Hand Side (RHS) from 8 to 14 in increments of 2.
How does the maximum profit change?"

*(Expected Tool: `analyze_what_if_scenario`)*

---

## Infeasibility Diagnosis

### Scenario 4: Simple Conflict

**Prompt:**
"Diagnose why this linear program is infeasible:

- `x >= 10`
- `x <= 5`
- Objective: Maximize `x`

Find the conflicting constraints."

*(Expected Tool: `diagnose_infeasibility`)*

---

### Scenario 5: Hidden Conflict

**Prompt:**
"Explain the infeasibility in this system:

1. `x + y <= 10`
2. `x >= 8`
3. `y >= 5`

Objective can be anything (e.g., Max `x+y`). Identify the minimal set of conflicting constraints."

*(Expected Tool: `diagnose_infeasibility`)*

---

## Constraint Programming

### Scenario 6: Sudoku Solver

**Prompt:**
"Solve this Sudoku puzzle:
(Input as a grid where 0 is blank)
Row 1: 5 3 0 | 0 7 0 | 0 0 0
Row 2: 6 0 0 | 1 9 5 | 0 0 0
Row 3: 0 9 8 | 0 0 0 | 0 6 0
Row 4: 8 0 0 | 0 6 0 | 0 0 3
Row 5: 4 0 0 | 8 0 3 | 0 0 1
Row 6: 7 0 0 | 0 2 0 | 0 0 6
Row 7: 0 6 0 | 0 0 0 | 2 8 0
Row 8: 0 0 0 | 4 1 9 | 0 0 5
Row 9: 0 0 0 | 0 8 0 | 0 7 9

Please fill in the blanks."

*(Expected Tool: `solve_sudoku_puzzle`)*

---

### Scenario 7: N-Queens

**Prompt:**
"Find a valid placement for 8 Queens on an 8x8 chess board such that no two queens attack each other."

*(Expected Tool: `solve_n_queens_puzzle`)*

---

## Multi-Objective Optimization

### Scenario 8: Profit vs Quality

**Prompt:**
"Find the Pareto Frontier for this bi-objective problem:

- **Obj 1 (Profit)**: Maximize `3x + 2y`
- **Obj 2 (Quality)**: Maximize `x + 4y`
- **Constraint**: `x + y <= 10`
- bounds: x>=0, y>=0

Generate 5 points on the frontier."

*(Expected Tool: `solve_pareto_frontier`)*

---

## Solution Enumeration

### Scenario 9: Multiple Optimal Solutions

**Prompt:**
"Find multiple optimal solutions for this knapsack problem:
Capacity = 5.
Items:

- A: Value 3, Weight 2
- B: Value 3, Weight 2
- C: Value 3, Weight 2

Maximize Value. Return up to 3 alternative solutions."

*(Expected Tool: `find_alternative_optimal_solutions`)*

---

### Scenario 10: Near-Optimal Pool

**Prompt:**
"Find the optimal solution and 4 other solutions within 10% of the optimal value for:
Maximize `x + y`
Subject to:
`2x + 2y <= 10`
`x, y` are integers >= 0."

*(Expected Tool: `find_alternative_optimal_solutions`)*
