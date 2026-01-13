# Combinatorial Optimization Test Problems

## Bin Packing

### 1. Simple Packing
Pack items into bins of capacity 60:
- Items: A(40), B(30), C(35), D(25), E(20)

Expected: 3 bins
- Bin 1: A + E = 60
- Bin 2: C + D = 60
- Bin 3: B = 30

### 2. Tight Fit
Bin capacity: 100
Items: 50, 50, 50, 50, 25, 25, 25, 25

Expected: 3 bins (optimal)

### 3. First Fit vs Optimal
Items: 7, 5, 3, 8, 6, 4, 2, 9
Bin capacity: 10

First Fit might use more bins than optimal.

## Set Covering

### 4. Simple Coverage
Universe: {1, 2, 3, 4, 5}
Sets:
- S1: {1, 2, 3}, cost 5
- S2: {2, 4}, cost 3
- S3: {3, 4}, cost 3
- S4: {4, 5}, cost 4

Expected: S1 + S4, cost = 9

### 5. Facility Coverage
Cover all neighborhoods with fire stations:
- Station A covers: N1, N2, N3 (cost 100)
- Station B covers: N2, N4, N5 (cost 80)
- Station C covers: N3, N5, N6 (cost 90)
- Station D covers: N1, N6 (cost 70)

Neighborhoods: N1-N6

### 6. Airline Crew Scheduling
Cover all flight legs with crew pairings:
- Pairing 1: Flights {101, 102, 103}
- Pairing 2: Flights {102, 104}
- Pairing 3: Flights {103, 105, 106}
- Pairing 4: Flights {104, 106}

## Graph Coloring

### 7. Triangle + Node
Nodes: A, B, C, D
Edges: A-B, B-C, C-A, C-D

Expected: 3 colors (triangle needs 3)

### 8. Bipartite Graph
Nodes: {1, 2, 3, 4, 5, 6}
Edges: 1-4, 1-5, 2-4, 2-6, 3-5, 3-6

Expected: 2 colors (bipartite)

### 9. Exam Scheduling
Exams that share students cannot be at same time:
- Math conflicts with: Physics, CS
- Physics conflicts with: Math, Chemistry
- CS conflicts with: Math, English
- Chemistry conflicts with: Physics
- English conflicts with: CS

Minimum time slots needed?

### 10. Register Allocation
Variables that are live at same time need different registers:
- a conflicts with: b, c
- b conflicts with: a, d
- c conflicts with: a, d
- d conflicts with: b, c

## Cutting Stock

### 11. Simple Cutting
Stock length: 100
Items needed:
- 45cm pieces: 3 required
- 30cm pieces: 4 required
- 25cm pieces: 2 required

Expected: Minimize stock pieces used

### 12. Steel Cutting
Stock bars: 12 meters
Orders:
- 3m bars: 10 needed
- 4m bars: 8 needed
- 5m bars: 6 needed

Minimize waste.

### 13. Paper Rolls
Master roll width: 200cm
Customer orders:
- 50cm width: 20 rolls
- 60cm width: 15 rolls
- 80cm width: 10 rolls

Find cutting patterns minimizing master rolls used.

## Knapsack Variants

### 14. Classic 0/1 Knapsack
Capacity: 50
Items:
| Item | Value | Weight |
|------|-------|--------|
| A    | 60    | 10     |
| B    | 100   | 20     |
| C    | 120   | 30     |

Expected: B + C, value = 220

### 15. Multiple Knapsacks
2 knapsacks, capacity 15 each
Items: (value, weight)
- A: (10, 5)
- B: (15, 8)
- C: (12, 6)
- D: (8, 4)
- E: (20, 10)

### 16. Bounded Knapsack
Capacity: 100
Items with quantities:
- Type A: value 10, weight 5, max 5 available
- Type B: value 15, weight 8, max 3 available
- Type C: value 20, weight 12, max 4 available
