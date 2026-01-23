# Linear Programming Test Scenarios

This document contains test scenarios designed to be fed as prompts to an LLM to verify the Vertex MCP server's Linear Programming capabilities in industrial settings.

## Production Planning

### Scenario 1: Multi-Period Production Smoothing

**Prompt:**
"Optimize the production plan for a factory over the next 4 weeks to minimize costs.
**Demands**: [100, 150, 200, 120] units.
**Costs**:

- Production: $10/unit (Regular time), $15/unit (Overtime).
- Inventory Holding: $2/unit/week.
**Capacities**:
- Regular Time: 120 units/week.
- Overtime: 40 units/week.
**Constraint**: No ending inventory at Week 4.
What is the optimal production schedule?"
*(Expected Tool: `optimize_production_plan` or `solve_linear_program`)*

### Scenario 2: Chemical Blending

**Prompt:**
"Determine the mix of 3 raw chemicals (C1, C2, C3) to produce 1000kg of a product.
**Raw Material Costs**: C1=$20/kg, C2=$15/kg, C3=$10/kg.
**Composition Constraints**:

- Ingredient X: Product must contain >= 20% X. (C1 has 40%, C2 has 10%, C3 has 5%).
- Ingredient Y: Product must contain <= 10% Y. (C1 has 5%, C2 has 15%, C3 has 8%).
Minimize total cost."
*(Expected Tool: `solve_linear_program`)*

### Scenario 3: Animal Feed Mixing

**Prompt:**
"Find the least-cost ration for cattle feed.
**Ingredients**: Corn ($0.20/kg), Soymeal ($0.50/kg), Limestone ($0.05/kg).
**Nutrient Requirements**:

- Energy: >= 3000 kcal/kg
- Protein: >= 16%
- Calcium: >= 1%
**Nutrient Content**:
- Corn: 3400 kcal, 9% Prot, 0.02% Ca
- Soymeal: 2200 kcal, 45% Prot, 0.3% Ca
- Limestone: 0 kcal, 0% Prot, 38% Ca
Calculate the mix."
*(Expected Tool: `solve_linear_program`)*

## Investment Portfolio

### Scenario 4: Bond Portfolio Immunization

**Prompt:**
"Construct a bond portfolio to match a liability of $1M due in 5 years.
**Available Bonds**:

- Bond A: Duration 3 years, Yield 4%
- Bond B: Duration 7 years, Yield 6%
**Objective**: Maximize Yield.
**Constraints**:
- Total Value = $1M
- Weighted Average Duration = 5 years (to immunize against interest rate risk)
- Max 60% in any single bond.
Find the allocation."
*(Expected Tool: `optimize_investment_portfolio` or `solve_linear_program`)*

## Transportation Problem

### Scenario 5: Coal Supply to Power Plants

**Prompt:**
"Minimize the cost of transporting coal from 3 mines to 3 power plants.
**Supplies (Tons)**: Mine1 (500), Mine2 (700), Mine3 (800).
**Demands (Tons)**: PlantA (400), PlantB (900), PlantC (700).
**Shipping Costs ($/Ton)**:

- M1 -> A: 12, B: 15, C: 20
- M2 -> A: 18, B: 10, C: 14
- M3 -> A: 22, B: 16, C: 8
Find the optimal shipping schedule."
*(Expected Tool: `solve_linear_program`)*

## Workforce Scheduling

### Scenario 6: Call Center Staffing with Shift Overlaps

**Prompt:**
"Minimize daily labor cost for a call center.
**Staffing Needs**:

- 08:00-12:00: 20 agents
- 12:00-16:00: 35 agents
- 16:00-20:00: 30 agents
**Shift Types**:
- Shift 1 (08:00-16:00): Cost $150
- Shift 2 (12:00-20:00): Cost $160
How many agents should be hired for Shift 1 and Shift 2?"
*(Expected Tool: `solve_linear_program`)*

## Marketing Mix

### Scenario 7: Media Mix Optimization

**Prompt:**
"Maximize total audience reach with a $100k budget.
**Channels**:

- Digital Ads: Cost $2k/unit, Reach 50k, Max 20 units.
- TV Spots: Cost $15k/unit, Reach 300k, Min 2 units.
- Print: Cost $5k/unit, Reach 80k.
**Constraints**:
- At least 50% of budget must be on Digital + TV.
- Reach frequency balance: TV spots <= 5.
Find the optimal spend."
*(Expected Tool: `solve_linear_program`)*
