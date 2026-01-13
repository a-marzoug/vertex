# Mixed-Integer Programming Problems

Test problems for MIP tools in Vertex MCP Server.

---

## Assignment Problems

### Problem 1: Worker-Task Assignment

**Scenario**: Assign 3 workers to 3 tasks minimizing total cost.

| Worker | Task A | Task B | Task C |
|--------|--------|--------|--------|
| Alice  | 10     | 15     | 9      |
| Bob    | 9      | 18     | 5      |
| Carol  | 6      | 14     | 3      |

**Tool**: `optimize_assignment`

**Parameters**:
```json
{
  "workers": ["Alice", "Bob", "Carol"],
  "tasks": ["TaskA", "TaskB", "TaskC"],
  "costs": {
    "Alice": {"TaskA": 10, "TaskB": 15, "TaskC": 9},
    "Bob": {"TaskA": 9, "TaskB": 18, "TaskC": 5},
    "Carol": {"TaskA": 6, "TaskB": 14, "TaskC": 3}
  }
}
```

**Expected**: Optimal cost = 24 (Alice→TaskB, Bob→TaskA, Carol→TaskC)

---

### Problem 2: Machine-Job Assignment

**Scenario**: 4 machines, 4 jobs. Each machine processes one job.

| Machine | Job1 | Job2 | Job3 | Job4 |
|---------|------|------|------|------|
| M1      | 82   | 83   | 69   | 92   |
| M2      | 77   | 37   | 49   | 92   |
| M3      | 11   | 69   | 5    | 86   |
| M4      | 8    | 9    | 98   | 23   |

**Tool**: `optimize_assignment`

---

## Knapsack Problems

### Problem 3: Backpack Packing

**Scenario**: Select items for a hiking trip. Backpack capacity: 15 kg.

| Item       | Value | Weight |
|------------|-------|--------|
| Tent       | 100   | 5      |
| Sleeping bag | 80  | 3      |
| Food       | 60    | 4      |
| Water      | 50    | 3      |
| Camera     | 40    | 2      |
| Book       | 10    | 1      |

**Tool**: `optimize_knapsack`

**Parameters**:
```json
{
  "items": ["tent", "sleeping_bag", "food", "water", "camera", "book"],
  "values": {"tent": 100, "sleeping_bag": 80, "food": 60, "water": 50, "camera": 40, "book": 10},
  "weights": {"tent": 5, "sleeping_bag": 3, "food": 4, "water": 3, "camera": 2, "book": 1},
  "capacity": 15
}
```

---

### Problem 4: Investment Selection

**Scenario**: Select projects to fund. Budget: $500,000.

| Project | Return ($K) | Cost ($K) |
|---------|-------------|-----------|
| A       | 120         | 150       |
| B       | 80          | 100       |
| C       | 150         | 200       |
| D       | 60          | 80        |
| E       | 90          | 120       |

**Tool**: `optimize_knapsack`

---

## Facility Location Problems

### Problem 5: Warehouse Location

**Scenario**: Open warehouses to serve 4 cities. Minimize fixed + shipping costs.

**Potential Warehouses**:
- NYC: $10,000 fixed cost
- Chicago: $8,000 fixed cost
- LA: $12,000 fixed cost

**Customers**: Boston, Miami, Seattle, Denver

**Shipping costs** ($/unit):

| From    | Boston | Miami | Seattle | Denver |
|---------|--------|-------|---------|--------|
| NYC     | 50     | 100   | 200     | 150    |
| Chicago | 100    | 120   | 150     | 80     |
| LA      | 200    | 180   | 50      | 100    |

**Tool**: `optimize_facility_location`

**Parameters**:
```json
{
  "facilities": ["NYC", "Chicago", "LA"],
  "customers": ["Boston", "Miami", "Seattle", "Denver"],
  "fixed_costs": {"NYC": 10000, "Chicago": 8000, "LA": 12000},
  "transport_costs": {
    "NYC": {"Boston": 50, "Miami": 100, "Seattle": 200, "Denver": 150},
    "Chicago": {"Boston": 100, "Miami": 120, "Seattle": 150, "Denver": 80},
    "LA": {"Boston": 200, "Miami": 180, "Seattle": 50, "Denver": 100}
  }
}
```

---

### Problem 6: Distribution Center Selection

**Scenario**: Select distribution centers for an e-commerce company.

**Potential DCs**: Atlanta, Dallas, Phoenix, Seattle
**Fixed costs**: $50K, $45K, $55K, $60K respectively
**Customers**: 6 major metro areas with varying shipping costs

**Tool**: `optimize_facility_location`

---

## Generic MIP Problems

### Problem 7: Production with Setup Costs

**Scenario**: Produce products with fixed setup costs per product line.

- Product A: $500 setup, $10/unit profit, max 100 units
- Product B: $300 setup, $8/unit profit, max 150 units
- Product C: $400 setup, $12/unit profit, max 80 units

Shared resource: 200 hours available
- A needs 1.5 hours/unit
- B needs 1 hour/unit
- C needs 2 hours/unit

**Tool**: `solve_mixed_integer_program`

**Formulation**:
- Binary: y_A, y_B, y_C (produce or not)
- Integer: x_A, x_B, x_C (quantity)
- Objective: max 10x_A + 8x_B + 12x_C - 500y_A - 300y_B - 400y_C
- Constraints: x_i ≤ M * y_i (linking), resource constraint

---

### Problem 8: Shift Scheduling

**Scenario**: Schedule workers for shifts. Each worker works exactly one shift.

**Workers**: 5 available
**Shifts**: Morning (need 2), Afternoon (need 2), Night (need 1)
**Costs vary by worker-shift combination**

**Tool**: `solve_mixed_integer_program`

---

### Problem 9: Lot Sizing

**Scenario**: Decide production quantities over 4 periods.

- Demand: [100, 150, 120, 180]
- Setup cost: $500 per period if producing
- Holding cost: $2 per unit per period
- Production cost: $10 per unit

**Tool**: `solve_mixed_integer_program`

---

## Tips for MIP Problems

1. **Binary variables** for yes/no decisions
2. **Integer variables** for countable quantities
3. **Big-M constraints** to link binary and continuous variables
4. **Start simple** - verify small instances before scaling

## Common Patterns

| Pattern | Binary Variable | Constraint |
|---------|-----------------|------------|
| Selection | x_i = 1 if selected | Σx_i ≤ k (select at most k) |
| Assignment | x_ij = 1 if i→j | Σ_j x_ij = 1 (each i assigned once) |
| Facility | y_j = 1 if open | x_ij ≤ y_j (can only use if open) |
| Setup | y = 1 if producing | x ≤ M*y (production requires setup) |
