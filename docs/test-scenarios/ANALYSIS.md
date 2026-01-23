# Analysis Tools Test Scenarios

This document contains test scenarios for sensitivity analysis, infeasibility diagnosis, and solution analysis in industrial contexts.

## Sensitivity Analysis

### Scenario 1: Refinery Shadow Prices

**Prompt:**
"Maximize profit for a refinery blending Gas ($50/bbl) and Diesel ($40/bbl).
**Constraints**:

- Crude Oil: <= 5000 bbl
- Distillation Capacity: <= 4000 hours
- Crack Spread Constraint: Gas >= 0.4 * Total Output
After solving, analyze the sensitivity. What is the marginal value (shadow price) of one additional barrel of Crude Oil? Is Distillation Capacity binding?"
*(Expected Tool: `analyze_lp_sensitivity`)*

### Scenario 2: Reduced Costs in Logistics

**Prompt:**
"Optimize shipping from 2 Warehouses to 2 Cities.
Route W1->C1 is currently unused in the optimal solution.
Report the reduced cost for the variable `flow_W1_C1`. How much would the shipping cost on this route need to drop for it to become viable?"
*(Expected Tool: `analyze_lp_sensitivity`)*

## What-If Analysis

### Scenario 3: Raw Material Price Shock

**Prompt:**
"For a furniture factory maximizing profit:
Constraint `Wood <= 1000`.
Perform a What-If analysis on the Wood availability constraint.
Vary the RHS from 800 to 1200 in increments of 50.
How does the maximum profit change? Plot the impact."
*(Expected Tool: `analyze_what_if_scenario`)*

## Infeasibility Diagnosis

### Scenario 4: Conflicting Shift Requirements

**Prompt:**
"Diagnose why this nurse schedule is infeasible:

- `Total_Nurses <= 5` (Budget cut)
- `Day_Shift >= 3` (Minimum staffing)
- `Night_Shift >= 3` (Minimum staffing)
- `Total_Nurses = Day_Shift + Night_Shift`
Find the conflicting constraints."
*(Expected Tool: `diagnose_infeasibility`)*

## Multi-Objective Optimization

### Scenario 5: Profit vs Environmental Impact

**Prompt:**
"Find the Pareto Frontier for a manufacturing plant:

- **Obj 1 (Profit)**: Maximize Revenue - Cost
- **Obj 2 (Sustainability)**: Minimize Carbon Emissions
- **Variables**: Product A (High profit, high carbon), Product B (Low profit, low carbon).
- **Constraint**: Total Production <= 1000.
Generate 5 points on the frontier to show the trade-off."
*(Expected Tool: `solve_pareto_frontier`)*

## Solution Enumeration

### Scenario 6: Alternative Supply Chain Routes

**Prompt:**
"Find multiple optimal supply chain configurations for a network design problem.
We want to see 3 different sets of active warehouses that achieve within 1% of the minimum cost.
This helps us consider qualitative factors like political stability not in the model."
*(Expected Tool: `find_alternative_optimal_solutions`)*
