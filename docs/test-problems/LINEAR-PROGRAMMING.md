# Example Problems for Vertex MCP Server

This document contains a collection of optimization problems that can be solved using the Vertex MCP server tools. Each problem includes a description, the relevant tool to use, and the expected approach.

## Table of Contents

- [Production Planning](#production-planning)
- [Diet Optimization](#diet-optimization)
- [Investment Portfolio](#investment-portfolio)
- [General Linear Programming](#general-linear-programming)

## Important Note About Solutions

**Why do solutions often have only one or two non-zero variables?**

This is **normal and mathematically correct** for linear programming! LP solutions occur at "corner points" (vertices) of the feasible region, where constraints intersect. This often means:

- Only a few variables are non-zero
- Some resources are fully utilized (binding constraints)
- Other variables are zero because they're not profitable enough given the constraints

**Example**: In the furniture problem, making only chairs (not tables) maximizes profit because chairs have better profit-per-labor-hour ratio, and labor is the bottleneck.

This is the **optimal** solution - any other mix would give less profit! If you want solutions with multiple products, you need to add constraints like:

- Minimum production requirements ("must make at least 10 tables")
- Maximum production limits ("can't make more than 30 chairs")
- Market demand constraints

**The precision issue** (like 49.999999 instead of 50) was a floating-point artifact - now fixed with rounding to 6 decimal places.

---

## Production Planning

### Problem 1: Furniture Manufacturing

**Scenario**: A furniture company manufactures chairs and tables. Each chair requires 5 units of wood and 2 hours of labor, while each table requires 20 units of wood and 5 hours of labor. The company has 400 units of wood and 100 hours of labor available. Chairs sell for $45 profit each, and tables for $80 profit each.

**Question**: How many chairs and tables should the company produce to maximize profit?

**Tool**: `optimize_production_plan`

**Parameters**:

- Products: `["chairs", "tables"]`
- Resources: `["wood", "labor_hours"]`
- Profits: `{"chairs": 45, "tables": 80}`
- Requirements: `{"chairs": {"wood": 5, "labor_hours": 2}, "tables": {"wood": 20, "labor_hours": 5}}`
- Availability: `{"wood": 400, "labor_hours": 100}`

---

### Problem 2: Electronics Assembly

**Scenario**: An electronics company produces smartphones, tablets, and laptops. The production requirements are:

- **Smartphone**: 1.5 hours assembly, 0.5 hours testing, 30 components, profit $80
- **Tablet**: 2 hours assembly, 0.8 hours testing, 45 components, profit $120
- **Laptop**: 3 hours assembly, 1.2 hours testing, 60 components, profit $180

Available resources: 300 hours assembly time, 120 hours testing time, 6000 components.

**Question**: What is the optimal production mix?

**Tool**: `optimize_production_plan`

**Parameters**:

- Products: `["smartphones", "tablets", "laptops"]`
- Resources: `["assembly", "testing", "components"]`
- Profits: `{"smartphones": 80, "tablets": 120, "laptops": 180}`
- Requirements: `{"smartphones": {"assembly": 1.5, "testing": 0.5, "components": 30}, "tablets": {"assembly": 2, "testing": 0.8, "components": 45}, "laptops": {"assembly": 3, "testing": 1.2, "components": 60}}`
- Availability: `{"assembly": 300, "testing": 120, "components": 6000}`

---

### Problem 3: Bakery Production

**Scenario**: A bakery makes croissants, baguettes, and cakes. Each product requires flour, butter, and oven time:

- **Croissant**: 0.1 kg flour, 0.05 kg butter, 15 min oven, profit $2
- **Baguette**: 0.3 kg flour, 0.01 kg butter, 20 min oven, profit $3
- **Cake**: 0.5 kg flour, 0.2 kg butter, 45 min oven, profit $12

Daily availability: 50 kg flour, 10 kg butter, 480 min oven time.

**Question**: What should the bakery produce daily to maximize profit?

**Tool**: `optimize_production_plan`

---

## Diet Optimization

### Problem 4: Athlete Meal Planning

**Scenario**: An athlete needs to plan a daily diet meeting nutritional requirements at minimum cost. Available foods:

- **Chicken breast** ($8/kg): 300 cal, 30g protein, 5g fat per 100g
- **Brown rice** ($2/kg): 350 cal, 7g protein, 2g fat per 100g
- **Broccoli** ($3/kg): 35 cal, 3g protein, 0.5g fat per 100g
- **Almonds** ($15/kg): 580 cal, 21g protein, 50g fat per 100g

Daily requirements: 2500 calories, 150g protein, max 70g fat.

**Question**: What quantities of each food minimize cost while meeting requirements?

**Tool**: `optimize_diet_plan`

---

### Problem 5: Budget Meal Prep

**Scenario**: A student wants to minimize food costs while meeting basic nutrition:

- **Pasta** ($1.50/lb): 350 cal, 12g protein, 2g carbs per 100g
- **Eggs** ($3/dozen): 70 cal, 6g protein, 5g fat each
- **Beans** ($2/can): 120 cal, 8g protein, 15g carbs per 100g
- **Spinach** ($2/bag): 20 cal, 3g protein, 3g carbs per 100g

Requirements: 2000+ calories, 60+ g protein, 200+ g carbs, max $10/day budget.

**Question**: What's the cheapest meal plan?

**Tool**: `optimize_diet_plan`

---

### Problem 6: Hospital Patient Diet

**Scenario**: A hospital needs to plan patient meals meeting specific dietary restrictions:

- **Oatmeal**: $0.50/serving, 150 cal, 5g protein, 3g fiber
- **Grilled fish**: $4/serving, 200 cal, 25g protein, 0g fiber
- **Vegetables**: $1/serving, 50 cal, 2g protein, 4g fiber
- **Fruit**: $1.50/serving, 80 cal, 1g protein, 3g fiber

Requirements: 1800-2200 calories, 75+ g protein, 25+ g fiber.

**Question**: What meal combination minimizes cost while meeting requirements?

**Tool**: `optimize_diet_plan`

---

## Investment Portfolio

### Problem 7: Retirement Portfolio

**Scenario**: An investor has $100,000 to allocate across different assets:

- **Stocks**: Expected return 12% annually
- **Bonds**: Expected return 5% annually
- **Real Estate**: Expected return 8% annually
- **Cash**: Expected return 2% annually

Constraints: At least $10,000 in bonds (safety), at most $40,000 in stocks (risk limit).

**Question**: How should the investor allocate funds to maximize expected return?

**Tool**: `optimize_investment_portfolio`

---

### Problem 8: Diversified Portfolio

**Scenario**: A wealth manager has $500,000 to invest across 5 asset classes:

- **Tech stocks**: 15% expected return, max $150,000
- **Blue chip stocks**: 10% expected return, min $50,000
- **Government bonds**: 4% expected return, min $100,000
- **Corporate bonds**: 6% expected return
- **Commodities**: 8% expected return, max $100,000

**Question**: What allocation maximizes returns while meeting constraints?

**Tool**: `optimize_investment_portfolio`

---

### Problem 9: Conservative Investment

**Scenario**: A retiree has $200,000 to invest conservatively:

- **Treasury bonds**: 3.5% return
- **Municipal bonds**: 4% return
- **Dividend stocks**: 6% return, max 30% of portfolio
- **Money market**: 2% return, min 10% of portfolio

**Question**: Optimal allocation for maximum return with risk constraints?

**Tool**: `optimize_investment_portfolio`

---

## General Linear Programming

### Problem 10: Transportation Problem

**Scenario**: A company has 3 warehouses and 4 retail stores. Shipping costs vary by route. Warehouses have limited supply, stores have specific demand.

**Warehouses** (supply):

- W1: 100 units
- W2: 150 units
- W3: 120 units

**Stores** (demand):

- S1: 80 units
- S2: 90 units
- S3: 70 units
- S4: 60 units

**Shipping costs** ($/unit): Varies by warehouse-store pair.

**Question**: Minimize total shipping cost while meeting all demands.

**Tool**: `solve_linear_program`

---

### Problem 11: Blending Problem

**Scenario**: A gasoline refinery blends 3 crude oils to produce gasoline meeting octane and sulfur specifications:

- **Crude A**: 95 octane, 0.5% sulfur, $60/barrel
- **Crude B**: 87 octane, 1.2% sulfur, $45/barrel
- **Crude C**: 92 octane, 0.8% sulfur, $55/barrel

Requirements: Final blend must have 90+ octane, max 0.9% sulfur.

**Question**: What blend minimizes cost while meeting specifications?

**Tool**: `solve_linear_program`

---

### Problem 12: Workforce Scheduling

**Scenario**: A call center needs to schedule workers across 3 shifts:

- **Morning** (8am-4pm): Need 15 workers minimum
- **Afternoon** (4pm-12am): Need 20 workers minimum
- **Night** (12am-8am): Need 10 workers minimum

Workers can work:

- **Full-time**: 8 hours, $20/hour
- **Part-time morning**: 4 hours, $15/hour
- **Part-time evening**: 4 hours, $18/hour

**Question**: Minimize labor cost while meeting shift requirements.

**Tool**: `solve_linear_program`

---

### Problem 13: Advertising Budget Allocation

**Scenario**: A marketing team has $50,000 to allocate across 4 channels:

- **Social media**: Reaches 10,000 per $1,000, max $20,000
- **TV ads**: Reaches 50,000 per $5,000, min $10,000
- **Radio**: Reaches 15,000 per $2,000
- **Print**: Reaches 8,000 per $1,500

Goal: Maximize total reach within budget and constraints.

**Question**: How to allocate the advertising budget?

**Tool**: `solve_linear_program`

---

### Problem 14: Crop Planning

**Scenario**: A farmer has 100 acres and wants to plant wheat, corn, and soybeans:

- **Wheat**: $200 profit/acre, 5 hours labor/acre
- **Corn**: $300 profit/acre, 8 hours labor/acre
- **Soybeans**: $250 profit/acre, 6 hours labor/acre

Constraints: 600 hours labor available, at least 20 acres of wheat (crop rotation), at most 40 acres of corn (market demand).

**Question**: What planting plan maximizes profit?

**Tool**: `solve_linear_program`

---

### Problem 15: Manufacturing Mix

**Scenario**: A chemical plant produces 3 products (A, B, C) from 2 raw materials (R1, R2):

- **Product A**: Uses 2 units R1, 1 unit R2, sells for $40
- **Product B**: Uses 1 unit R1, 2 units R2, sells for $35
- **Product C**: Uses 1 unit R1, 1 unit R2, sells for $30

Available: 1000 units R1, 800 units R2. Market demand: Max 300 units A, max 250 units B.

**Question**: What production mix maximizes revenue?

**Tool**: `solve_linear_program`

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
