# Domain Template Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's high-level domain templates (Production, Diet, Portfolio, etc.). Feed these prompts to an LLM to test the tools.

## Production Planning

### Scenario 1: Furniture Production

**Prompt:**
"I run a factory making Chairs and Tables.
**Profit**: Chairs ($45), Tables ($80).
**Resources**:

- **Wood**: Chairs need 5, Tables need 20. (Total 400 available).
- **Labor**: Chairs need 2 hrs, Tables need 5 hrs. (Total 100 available).

What is the optimal product mix to maximize profit?"

*(Expected Tool: `optimize_production_plan`)*

---

### Scenario 2: Multi-Product Manufacturing

**Prompt:**
"Maximize profit for products A, B, and C.
**Resources Available**:

- Metal: 1000 kg
- Energy: 500 kWh
- Man-hours: 200

**Requirements per Unit**:

- **A**: 10 Metal, 5 Energy, 2 Man-hours. Profit $50.
- **B**: 20 Metal, 3 Energy, 3 Man-hours. Profit $80.
- **C**: 15 Metal, 8 Energy, 1 Man-hour. Profit $60.

Find the plan."

*(Expected Tool: `optimize_production_plan`)*

---

## Diet Optimization

### Scenario 3: Minimum Cost Diet

**Prompt:**
"Plan a daily diet using only Bread, Milk, and Eggs.
**Goal**: Minimize Cost.
**Prices**: Bread $1, Milk $2, Eggs $3.

**Nutritional Needs**:

- Calories: >= 2000
- Protein: >= 50g
- Calcium: >= 800mg

**Nutrition per Unit**:

- **Bread**: 250 Cal, 5g Prot, 10mg Calc
- **Milk**: 150 Cal, 8g Prot, 300mg Calc
- **Eggs**: 70 Cal, 6g Prot, 30mg Calc

What should I eat?"

*(Expected Tool: `optimize_diet_plan`)*

---

## Portfolio Optimization

### Scenario 4: Simple Portfolio

**Prompt:**
"Allocate a budget of $100,000 into Stocks, Bonds, and Cash.
**Expected Returns**:

- Stocks: 12%
- Bonds: 6%
- Cash: 2%

**Constraints**:

- Max 60% in Stocks.
- Min 10% in Cash.

Maximize expected return."

*(Expected Tool: `optimize_investment_portfolio`)*

---

### Scenario 5: Sector Diversification

**Prompt:**
"Invest $1M to maximize return across 5 sectors (Tech, Healthcare, Finance, Energy, Utilities).
**Returns**: Tech 15%, Health 10%, Finance 12%, Energy 8%, Utilities 6%.

**Diversification Rules**:

- Tech <= 30%
- Healthcare <= 25%
- Finance <= 20%
- Energy <= 15%
- Utilities <= 20%

What is the allocation?"

*(Expected Tool: `optimize_investment_portfolio`)*

---

## Inventory (EOQ)

### Scenario 6: Single Item EOQ

**Prompt:**
"Calculate the Economic Order Quantity (EOQ) for a product.

- **Annual Demand**: 10,000 units
- **Ordering Cost**: $50 per order
- **Holding Cost**: $2 per unit per year
- **Lead Time**: 7 days (for Reorder Point)

What is the optimal order size?"

*(Expected Tool: `optimize_inventory_eoq`)*

---

### Scenario 7: EOQ with Safety Stock

**Prompt:**
"Calculate EOQ and Safety Stock.

- **Annual Demand**: 10,000
- **Order Cost**: $50
- **Holding**: $2
- **Lead Time**: 7 days
- **Safety Stock**: 100 units (manually specified)

Return the optimal policy."

*(Expected Tool: `optimize_inventory_eoq`)*

---

## Workforce Scheduling

### Scenario 8: Weekly Staffing

**Prompt:**
"Schedule workers (Alice, Bob, Carol, Dave, Eve) for a week.
**Shifts**: 'Morning' and 'Evening'.
**Daily Requirements**:

- Morning: [3, 3, 2, 2, 3, 4, 4] (Mon-Sun)
- Evening: [2, 2, 2, 2, 2, 3, 3]

**Constraints**: Max 5 shifts per worker per week.
Minimize cost (assume equal cost)."

*(Expected Tool: `optimize_workforce`)*

---

## Healthcare Resource Allocation

### Scenario 9: Hospital Resource Allocation

**Prompt:**
"Allocate healthcare resources (Doctors, Nurses, Beds) to 3 locations (Hospital A, Clinic B, Clinic C) to maximize demand coverage.

**Availabilities**: 10 Doctors, 20 Nurses, 50 Beds.

**Needs (Demand)**:

- **Hospital A**: 6 Docs, 12 Nurses, 30 Beds
- **Clinic B**: 3 Docs, 6 Nurses, 15 Beds
- **Clinic C**: 2 Docs, 4 Nurses, 10 Beds

How should resources be distributed?"

*(Expected Tool: `optimize_healthcare_resources`)*

---

## Supply Chain Network Design

### Scenario 10: Facility Location & Sourcing

**Prompt:**
"Design a supply chain.
**Facilities** (Potential): F1 (Cap 100, Cost $100k), F2 (Cap 150, Cost $150k), F3 (Cap 120, Cost $120k).
**Customers** (Demands): C1(40), C2(50), C3(30), C4(45), C5(35).

**Transport Costs** (per unit):

- From F1: All $5
- From F2: All $4
- From F3: All $6

Which facilities should open to minimize Fixed + Transport costs?"

*(Expected Tool: `optimize_supply_chain_network`)*

---

## Assignment Problems

### Scenario 11: Worker-Task Matrix

**Prompt:**
"Assign 3 workers to 3 tasks to minimize cost.
**Cost Matrix**:

- W1: [5, 8, 6]
- W2: [9, 4, 7]
- W3: [6, 7, 3]

Each worker does exactly one task."

*(Expected Tool: `optimize_worker_assignment`)*

---

## Facility Location

### Scenario 12: General Facility Location

**Prompt:**
"Choose the best locations from candidates [L1, L2, L3, L4] to serve customers [C1, C2, C3, C4, C5, C6].
**Fixed Costs**: [1000, 1100, 1200, 1300]
**Transport Costs**: Assume simple distance-based costs.
(Ask LLM to mock up costs if needed).

Minimize total cost."

*(Expected Tool: `optimize_facility_locations`)*
