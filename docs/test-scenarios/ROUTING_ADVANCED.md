# Advanced Routing Test Scenarios

This document contains test scenarios for verifying advanced routing capabilities (Pickup & Delivery, Metaheuristic tuning).

## Pickup and Delivery VRP (PDP)

### Scenario 1: Courier Service

**Prompt:**
"Schedule a courier service with 2 vehicles (Capacity 5).
Depot at (0,0).
Requests (Pickup -> Delivery):
1. **Request A**: (2,2) -> (5,5) [Weight 1]
2. **Request B**: (1,4) -> (4,1) [Weight 2]
3. **Request C**: (-2,3) -> (-2,-3) [Weight 1]

Distance is Manhattan: |x1-x2| + |y1-y2|.
Pickup must happen before Delivery.
Minimize total distance."

*(Expected Tool: `solve_pickup_delivery`)*

---

### Scenario 2: Hospital Transfer

**Prompt:**
"Transport 3 patients between hospitals using 2 ambulances.
Locations:
- H1 (Depot)
- H2
- H3
- H4

Transfers:
1. Patient X: H2 -> H3
2. Patient Y: H4 -> H1
3. Patient Z: H3 -> H4

Constraints:
- Capacity: 2 patients per ambulance.
- Optimization: Minimize time.

(Ask LLM to mock up time matrix if needed)."

*(Expected Tool: `solve_pickup_delivery`)*

---

## Metaheuristic Configuration

### Scenario 3: Large TSP with Guided Local Search

**Prompt:**
"Solve a TSP for 50 cities (randomly generated coordinates).
Use **Guided Local Search** metaheuristic to escape local minima.
Set time limit to 5 seconds.
Compare result with greedy approach if possible (ask for two runs)."

*(Expected Tool: `solve_tsp` with `local_search_metaheuristic='GUIDED_LOCAL_SEARCH'`)*

---

### Scenario 4: VRP with Tabu Search

**Prompt:**
"Solve a VRP instance using **Tabu Search**.
Locations: Depot + 10 customers.
Vehicles: 3.
Check if Tabu Search yields a different route structure compared to default."

*(Expected Tool: `solve_vrp` with `local_search_metaheuristic='TABU_SEARCH'`)*
