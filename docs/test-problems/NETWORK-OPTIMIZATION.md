# Network Optimization Test Problems

## Maximum Flow

### 1. Simple Pipeline
Find max flow from S to T:
- S→A: capacity 10
- S→B: capacity 8
- A→B: capacity 5
- A→T: capacity 7
- B→T: capacity 10

Expected: max_flow = 17

### 2. Bipartite Matching
Match workers {W1, W2, W3} to jobs {J1, J2, J3}:
- S→W1, S→W2, S→W3: capacity 1
- W1→J1, W1→J2: capacity 1
- W2→J2, W2→J3: capacity 1
- W3→J1, W3→J3: capacity 1
- J1→T, J2→T, J3→T: capacity 1

Expected: max_flow = 3 (perfect matching exists)

## Minimum Cost Flow

### 3. Transportation
Ship 100 units from factory to warehouse:
- Factory→Hub1: capacity 60, cost 2
- Factory→Hub2: capacity 80, cost 4
- Hub1→Warehouse: capacity 70, cost 3
- Hub2→Warehouse: capacity 50, cost 1

Supplies: {Factory: 100, Warehouse: -100}

Expected: Route through both hubs to minimize cost

### 4. Assignment as Flow
Assign 3 workers to 3 tasks with costs:
- W1: {T1: 5, T2: 8, T3: 6}
- W2: {T1: 9, T2: 4, T3: 7}
- W3: {T1: 6, T2: 7, T3: 3}

Model as min cost flow with unit capacities.

## Shortest Path

### 5. Simple Graph
Find shortest A→D:
- A→B: cost 1
- A→C: cost 4
- B→C: cost 2
- B→D: cost 5
- C→D: cost 1

Expected: path = [A, B, C, D], distance = 4

### 6. Road Network
Cities: {Home, A, B, C, Work}
- Home→A: 10 min
- Home→B: 15 min
- A→B: 5 min
- A→C: 20 min
- B→C: 10 min
- C→Work: 5 min

Expected: Home→B→C→Work = 30 min

## Minimum Spanning Tree

### 7. Simple MST
Connect nodes A, B, C, D with minimum total weight:
- A-B: 1
- A-C: 4
- B-C: 2
- B-D: 5
- C-D: 3

Expected: edges = {A-B, B-C, C-D}, total_weight = 6

### 8. Network Infrastructure
Connect 5 offices with minimum cable cost:
- Office1-Office2: 10
- Office1-Office3: 15
- Office2-Office3: 8
- Office2-Office4: 12
- Office3-Office4: 6
- Office3-Office5: 9
- Office4-Office5: 7

## Multi-Commodity Flow

### 9. Two Products
Route products A and B through shared network:
- Nodes: S1, S2, Hub, D1, D2
- Arcs: S1→Hub (cap 50), S2→Hub (cap 50), Hub→D1 (cap 40), Hub→D2 (cap 40)
- Product A: S1→D1, demand 20
- Product B: S2→D2, demand 30

Expected: Both products routed through Hub

### 10. Competing Flows
Three commodities sharing limited capacity:
- Single bottleneck arc with capacity 100
- Commodity 1: demand 40
- Commodity 2: demand 35
- Commodity 3: demand 30

Expected: Total flow = 100 (capacity limited)

## Transshipment

### 11. Factory-Warehouse-Store
- Sources: Factory1 (supply 100), Factory2 (supply 80)
- Transshipment: Warehouse1, Warehouse2
- Destinations: Store1 (demand 60), Store2 (demand 70), Store3 (demand 50)
- Costs vary by route

### 12. Multi-Tier Distribution
- Tier 1: 2 plants
- Tier 2: 3 regional warehouses
- Tier 3: 5 local distribution centers
- Tier 4: 10 retail stores

Find minimum cost flow through all tiers.
