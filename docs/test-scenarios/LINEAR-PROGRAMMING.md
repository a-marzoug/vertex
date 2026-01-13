# Linear Programming Test Scenarios

This document contains test scenarios designed to be fed as prompts to an LLM to verify the Vertex MCP server's capabilities.

## Production Planning

### Scenario 1: Furniture Manufacturing

**Prompt:**
"I need to optimize the weekly production plan for a furniture manufacturing company that makes chairs, tables, and desks.

Here is the data:

- **Chairs**: Sell for $25 profit. Require 3 units of wood, 2 hours of labor, and 1 hour of finishing.
- **Tables**: Sell for $40 profit. Require 5 units of wood, 3 hours of labor, and 2 hours of finishing.
- **Desks**: Sell for $55 profit. Require 7 units of wood, 4 hours of labor, and 3 hours of finishing.

**Constraints & Resources:**

- Wood available: 200 units
- Labor available: 150 hours
- Finishing available: 100 hours
- **Market Demand**: To maintain our brand presence, we must produce at least 10 chairs and 5 tables per week.
- **Storage Limit**: We cannot store more than 50 total items in our warehouse.

What is the optimal production mix to maximize our total profit?"

*(Expected Tool: `optimize_production_plan`)*

---

### Scenario 2: Electronics Assembly

**Prompt:**
"You are an operations manager for an electronics firm producing Smartphones, Tablets, and Laptops. Calculate the optimal product mix for the next shift.

**Profit & Requirements:**

- **Smartphone**: $80 profit/unit. Needs 1.5h assembly, 0.5h testing, 30 components.
- **Tablet**: $120 profit/unit. Needs 2.0h assembly, 0.8h testing, 45 components.
- **Laptop**: $180 profit/unit. Needs 3.0h assembly, 1.2h testing, 60 components.

**Available Resources:**

- Assembly Time: 300 hours
- Testing Time: 120 hours
- Electronic Components: 6000 units

**Operational Constraints:**

- We have a contract to deliver at least 20 Smartphones.
- Due to supply chain limits, we can produce at most 50 Laptops.
- The testing facility cannot handle more than 200 total devices per shift.

Please find the production quantities that maximize profit."

*(Expected Tool: `optimize_production_plan`)*

---

### Scenario 3: Bakery Production

**Prompt:**
"I run a bakery and need to decide what to bake tomorrow.
I make Croissants, Baguettes, and Cakes.

**Resource Usage & Profit:**

- **Croissant**: Profit $2. Needs 0.1kg flour, 0.05kg butter, 15 min oven time.
- **Baguette**: Profit $3. Needs 0.3kg flour, 0.01kg butter, 20 min oven time.
- **Cake**: Profit $12. Needs 0.5kg flour, 0.2kg butter, 45 min oven time.

**Daily Supply:**

- Flour: 50 kg
- Butter: 10 kg
- Oven Time: 480 minutes (8 hours)

**Business Rules:**

- Variety is key: We must make at least 20 Croissants and 10 Baguettes.
- We don't have box space for more than 5 Cakes.

What production plan maximizes my manufacturing profit?"

*(Expected Tool: `optimize_production_plan`)*

---

## Diet Optimization

### Scenario 4: Athlete Meal Planning

**Prompt:**
"I'm training for a marathon and need a daily meal plan that minimizes cost while hitting my macro goals.
Can you optimize my diet using these foods?

**Foods & Costs:**

- **Chicken Breast**: $8.00/kg. (Per 100g: 300 cal, 30g protein, 5g fat)
- **Brown Rice**: $2.00/kg. (Per 100g: 350 cal, 7g protein, 2g fat)
- **Broccoli**: $3.00/kg. (Per 100g: 35 cal, 3g protein, 0.5g fat)
- **Almonds**: $15.00/kg. (Per 100g: 580 cal, 21g protein, 50g fat)

**Nutritional Goals:**

- Calories: Exactly 2500 kcal (+/- 10% is fine, but aim for minimum cost meeting exactly 2500 if possible, or treated as a minimum requirement of 2500) -> *Correction: Treating as minimum 2500 kcal.*
- Protein: Minimum 150g
- Fat: Maximum 70g

Please tell me exactly how many grams of each food I should eat."

*(Expected Tool: `optimize_diet_plan`)*

---

### Scenario 5: Budget Meal Prep

**Prompt:**
"Help me plan a grocery list for the week. I want to spend as little money as possible but stay healthy.

**Options:**

- **Pasta**: $1.50/lb. (Per 100g: 350 cal, 12g protein, 2g carbs)
- **Eggs**: $3.00/dozen. (Per egg: 70 cal, 6g protein, 5g fat)
- **Beans**: $2.00/can. (Per 100g: 120 cal, 8g protein, 15g carbs)
- **Spinach**: $2.00/bag. (Per 100g: 20 cal, 3g protein, 3g carbs)

**Daily Requirements:**

- At least 2000 Calories
- At least 60g Protein
- At least 200g Carbs
- **Variety Constraint**: I want to eat at least 100g of Spinach for iron.

What is the cheapest combination?"

*(Expected Tool: `optimize_diet_plan`)*

---

## Investment Portfolio

### Scenario 6: Retirement Portfolio

**Prompt:**
"I have $100,000 to invest for retirement. I want to maximize my expected yearly return.

**Assets & Returns:**

- **Stocks**: 12% return
- **Bonds**: 5% return
- **Real Estate**: 8% return
- **Cash**: 2% return

**Risk Management Constraints:**

- Safety: Invest at least $10,000 in Bonds.
- Liquidity: Keep at least $5,000 in Cash.
- Diversity: Real Estate should not exceed 30% of the total portfolio.
- Risk Limit: Stocks cannot exceed $40,000.

How should I allocate the $100,000?"

*(Expected Tool: `optimize_investment_portfolio`)*

---

### Scenario 7: Risk-Averse Allocation

**Prompt:**
"I am managing a $500,000 fund. The goal is maximum return, but we have strict policy guidelines.

**Assets:**

1. Tech Stocks (15% return)
2. Blue Chip Stocks (10% return)
3. Gov Bonds (4% return)
4. Corp Bonds (6% return)
5. Commodities (8% return)

**Guidelines:**

- **Minimum Safety**: At least $100,000 must be in Government Bonds.
- **Maximum Volatility**: No more than $150,000 in Tech Stocks.
- **Stability**: Blue Chip Stocks must be at least $50,000.
- **Commodities Limit**: Max $100,000.
- **Bond Balance**: Total in bonds (Gov + Corp) must be at least 40% of the total portfolio ($200,000).

Calculate the optimal allocation."

*(Expected Tool: `optimize_investment_portfolio`)*

---

## General Linear Programming

### Scenario 8: The Transportation Problem

**Prompt:**
"Solve this transportation optimization problem to minimize shipping costs.

**Warehouses (Supply):**

- W1: 100 units
- W2: 150 units
- W3: 120 units

**Stores (Demand):**

- S1: 80 units
- S2: 90 units
- S3: 70 units
- S4: 60 units

**Shipping Costs (per unit):**

- W1 -> S1: $5, S2: $7, S3: $3, S4: $10
- W2 -> S1: $8, S2: $6, S3: $9, S4: $12
- W3 -> S1: $4, S2: $5, S3: $2, S4: $7

**Constraint**: Every store's demand must be fully met.
Provide the optimal shipping schedule from each warehouse to each store."

*(Expected Tool: `solve_linear_program`)*

---

### Scenario 9: Workforce Scheduling

**Prompt:**
"I manage a call center with three shifts. I need to minimize total labor cost while ensuring coverage.

**Shifts & Coverage Needed:**

1. Morning (8am-4pm): Minimum 15 workers
2. Afternoon (4pm-12am): Minimum 20 workers
3. Night (12am-8am): Minimum 10 workers

**Worker Types:**

- **Full-Time** ($20/hr, 8hr shift): Can work Morning OR Afternoon OR Night.
- **Part-Time AM** ($15/hr, 4hr shift): Works 8am-12pm (covers half of Morning). *Note: Simplify to 'Partial coverage' or treated as full coverage for simplicity if needed, but let's assume they contribute to the Morning headcount.*
- Let's simplify:
  - **Type A (Morning)**: $160/shift. Covers Morning.
  - **Type B (Afternoon)**: $160/shift. Covers Afternoon.
  - **Type C (Night)**: $170/shift. Covers Night.
  - **Type D (Split)**: $140/shift. Covers half of Morning and half of Afternoon? (Maybe too complex for basic LP prompt).

*Let's retry the prompt with clearer LP structure:*

**Prompt:**
'Optimize the workforce schedule to minimize daily cost.

**Shifts:**

- Shift 1 (08:00-16:00): Needs 15 staff.
- Shift 2 (16:00-24:00): Needs 20 staff.
- Shift 3 (00:00-08:00): Needs 10 staff.

**Employee Types:**

- **Regular**: Works one full shift (1, 2, or 3). Cost: $200/shift.
- **Overtime**: Can work double shifts (1+2, or 2+3). Cost: $350/shift.
- **Part-Time**: Works only 4 hours of a shift (Assume 2 Part-Time = 1 Full Staff). Cost: $90 per 4hr block. Available for all shifts.

**Constraints:**

- Union rule: At least 50% of staff in any shift must be Regular.

Find the optimal number of Regular, Overtime, and Part-time staff for each shift.'"

*(Expected Tool: `solve_linear_program`)*

---

### Scenario 10: Advertising Mix

**Prompt:**
"Allocate our marketing budget of $50,000 to maximize total audience reach.

**Channels:**

1. **Social Media**: Cost $1,000 per block. Reach: 10,000 people/block. Max Blocks: 20.
2. **TV Spots**: Cost $5,000 per spot. Reach: 50,000 people/spot. Min Spots: 2 (Brand visibility).
3. **Radio**: Cost $2,000 per slot. Reach: 15,000 people/slot.
4. **Newspaper**: Cost $1,500 per ad. Reach: 8,000 people/ad.

**Strategic Constraints:**

- We must have a presence on at least 3 distinct channels.
- Spending on Social Media cannot exceed spending on TV.
- Total Newspaper ads must not exceed 5.

What is the best allocation?"

*(Expected Tool: `solve_linear_program`)*

---

## How to Use These Problems

1. **With Claude Desktop**: Simply describe the problem to Claude, and it will use the appropriate Vertex tool to solve it.

2. **Direct API calls**: Use the tool parameters shown in each problem description.

3. **Learning**: Start with simpler problems (furniture, basic diet) before tackling complex ones (transportation, blending).

4. **Experimentation**: Modify the numbers to see how solutions change with different constraints.

## Tips for Problem Solving

- **Identify the objective**: Are you maximizing profit or minimizing cost?
- **List all constraints**: Resource limits, minimum requirements, maximum capacities
- **Choose the right tool**:
  - Use domain templates (production, diet, portfolio) when they fit
  - Use `solve_linear_program` for custom problems
- **Interpret results**: Understand what the solution means in business context
- **Sensitivity analysis**: Try changing one constraint to see impact on solution
