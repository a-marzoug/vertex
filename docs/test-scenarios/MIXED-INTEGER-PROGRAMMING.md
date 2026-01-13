# Mixed-Integer Programming Test Scenarios

This document contains test scenarios to verify the Vertex MCP server's MIP capabilities (integer constraints, binary decisions). Feed these prompts to an LLM to test the tools.

## Assignment Problems

### Scenario 1: Worker-Task Assignment

**Prompt:**
"I need to assign 3 workers (Alice, Bob, Carol) to 3 tasks (A, B, C).
Each worker handles exactly one task.

**Cost Matrix:**

| Worker | Task A | Task B | Task C |
|--------|--------|--------|--------|
| Alice  | 10     | 15     | 9      |
| Bob    | 9      | 18     | 5      |
| Carol  | 6      | 14     | 3      |

Find the assignment that minimizes the total cost."

*(Expected Tool: `optimize_worker_assignment`)*

---

### Scenario 2: Machine-Job Assignment

**Prompt:**
"Match 4 machines to 4 jobs to minimize processing time.
One machine -> One job.

**Processing Times:**

- **Machine 1**: J1=82, J2=83, J3=69, J4=92
- **Machine 2**: J1=77, J2=37, J3=49, J4=92
- **Machine 3**: J1=11, J2=69, J3=5, J4=86
- **Machine 4**: J1=8, J2=9, J3=98, J4=23

What is the optimal matching?"

*(Expected Tool: `optimize_worker_assignment`)*

---

## Knapsack Problems

### Scenario 3: Backpack Packing

**Prompt:**
"I'm going on a hike and need to choose which items to pack.
My backpack has a max weight capacity of **15 kg**.

**Items Available:**

1. **Tent**: Value 100, Weight 5kg
2. **Sleeping Bag**: Value 80, Weight 3kg
3. **Food**: Value 60, Weight 4kg
4. **Water**: Value 50, Weight 3kg
5. **Camera**: Value 40, Weight 2kg
6. **Book**: Value 10, Weight 1kg

Which items should I pack to maximize total value?"

*(Expected Tool: `optimize_knapsack_selection`)*

---

### Scenario 4: Investment Project Selection

**Prompt:**
"I have an R&D budget of $500,000. I need to select which projects to fund.

**Project Data:**

- **Project A**: Cost $150k, Return $120k
- **Project B**: Cost $100k, Return $80k
- **Project C**: Cost $200k, Return $150k
- **Project D**: Cost $80k, Return $60k
- **Project E**: Cost $120k, Return $90k

Select the combination of projects that yields the highest total return within the budget."

*(Expected Tool: `optimize_knapsack_selection`)*

---

## Facility Location Problems

### Scenario 5: Warehouse Location

**Prompt:**
"I need to decide where to open warehouses to serve my customers.
**Goal**: Minimize Total Cost (Fixed Opening Cost + Shipping Cost).

**Potential Warehouse Sites:**

- **NYC**: Fixed Cost $10,000
- **Chicago**: Fixed Cost $8,000
- **LA**: Fixed Cost $12,000

**Customer Demands**: All customers must be served by exactly one open warehouse.
(Assume simplified model: costs represented as unit shipping * demand, pre-calculated as total cost to serve that city).

**Total Shipping Costs (if served by...):**

| To \ From | NYC | Chicago | LA |
|-----------|-----|---------|----|
| **Boston** | 50  | 100     | 200|
| **Miami**  | 100 | 120     | 180|
| **Seattle**| 200 | 150     | 50 |
| **Denver** | 150 | 80      | 100|

Which warehouses should I open and who should they serve?"

*(Expected Tool: `optimize_facility_locations`)*

---

### Scenario 6: Distribution Center Selection

**Prompt:**
"Select the optimal Distribution Centers (DCs) from 4 candidates:

- **Atlanta**: fixed cost $50k
- **Dallas**: fixed cost $45k
- **Phoenix**: fixed cost $55k
- **Seattle**: fixed cost $60k

**Customers & Ship Costs:**

- **C1**: Atlanta($10), Dallas($20), Phoenix($40), Seattle($50)
- **C2**: Atlanta($20), Dallas($10), Phoenix($30), Seattle($40)
- **C3**: Atlanta($40), Dallas($30), Phoenix($10), Seattle($20)
- **C4**: Atlanta($50), Dallas($40), Phoenix($20), Seattle($10)
- **C5**: Atlanta($15), Dallas($15), Phoenix($35), Seattle($45)
- **C6**: Atlanta($45), Dallas($35), Phoenix($15), Seattle($15)

Minimize total fixed + shipping costs."

*(Expected Tool: `optimize_facility_locations`)*

---

## Generic MIP Problems

### Scenario 7: Production with Setup Costs

**Prompt:**
"Optimize production planning considering fixed setup costs.

**Products:**

- **A**: Setup $500. Profit $10/unit. Max demand 100. Time: 1.5hr/unit.
- **B**: Setup $300. Profit $8/unit. Max demand 150. Time: 1.0hr/unit.
- **C**: Setup $400. Profit $12/unit. Max demand 80. Time: 2.0hr/unit.

**Resource Limit**: 200 hours total machine time.

**Key Rule**: If we produce any amount of a product, we pay the full setup cost.
Objective: Maximize Net Profit (Total Profit - Setup Costs)."

*(Expected Tool: `solve_mixed_integer_program`)*

---

### Scenario 8: Shift Scheduling Constraint

**Prompt:**
"Assign 5 workers to shifts.
**Rules**:

1. Each worker does exactly 1 shift.
2. **Morning Shift** needs exactly 2 workers.
3. **Afternoon Shift** needs exactly 2 workers.
4. **Night Shift** needs exactly 1 worker.

**Preferences (Cost Penalty Matrix):**
(Lower is better)

- W1: Morn(1), Aft(5), Night(10)
- W2: Morn(10), Aft(1), Night(5)
- W3: Morn(5), Aft(5), Night(1)
- W4: Morn(1), Aft(10), Night(10)
- W5: Morn(5), Aft(1), Night(5)

Find the assignment with minimum total penalty."

*(Expected Tool: `solve_mixed_integer_program`)*

---

### Scenario 9: Lot Sizing

**Prompt:**
"Calculate the optimal production schedule for the next 4 weeks.

**Demand:**

- Week 1: 100 units
- Week 2: 150 units
- Week 3: 120 units
- Week 4: 180 units

**Costs:**

- **Setup Cost**: $500 (Payable every week we decide to run the machine).
- **Unit Cost**: $10 per unit produced.
- **Holding Cost**: $2 per unit kept in inventory at the end of the week.

**Constraints:**

- No starting inventory.
- No shortages allowed.

Objective: Minimize total cost."

*(Expected Tool: `solve_mixed_integer_program`)*
