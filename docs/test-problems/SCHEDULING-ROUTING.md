# Scheduling & Routing Test Problems

## Traveling Salesman Problem (TSP)

### 1. Four Cities
Find shortest tour visiting all cities:
- Distance matrix:
  ```
       A   B   C   D
  A [  0, 10, 15, 20]
  B [ 10,  0, 35, 25]
  C [ 15, 35,  0, 30]
  D [ 20, 25, 30,  0]
  ```

Expected: A→B→D→C→A, distance=80

### 2. Symmetric 5-City
```
     0   1   2   3   4
0 [  0, 29, 82, 46, 68]
1 [ 29,  0, 55, 46, 42]
2 [ 82, 55,  0, 68, 46]
3 [ 46, 46, 68,  0, 82]
4 [ 68, 42, 46, 82,  0]
```

## Vehicle Routing Problem (VRP)

### 3. Simple Delivery
- Depot + 4 customers
- 2 vehicles, capacity 30 each
- Demands: [0, 10, 15, 10, 20]
- Distance matrix:
  ```
        Depot  C1  C2  C3  C4
  Depot [  0, 10, 15, 20, 25]
  C1    [ 10,  0, 35, 25, 30]
  C2    [ 15, 35,  0, 30, 20]
  C3    [ 20, 25, 30,  0, 15]
  C4    [ 25, 30, 20, 15,  0]
  ```

Expected: 2 routes, each within capacity 30

### 4. Unbalanced Demands
- Depot + 5 customers
- 3 vehicles, capacity 20 each
- Demands: [0, 5, 8, 12, 15, 10]

## VRP with Time Windows

### 5. Morning Deliveries
- Depot opens at time 0
- Customer time windows:
  - C1: [10, 30]
  - C2: [20, 50]
  - C3: [30, 60]
- Travel times between locations
- 2 vehicles

### 6. Tight Windows
- 4 customers with narrow time windows
- Some windows overlap, some don't
- Test feasibility

## Job Shop Scheduling

### 7. Minimal 3x3
3 jobs, 3 machines:
```
Job 0: (M0, 3) → (M1, 2) → (M2, 2)
Job 1: (M0, 2) → (M2, 1) → (M1, 4)
Job 2: (M1, 4) → (M2, 3)
```

Expected: makespan = 11

### 8. FT06 Benchmark
Classic 6x6 job shop:
```
Job 0: (M2,1) → (M0,3) → (M1,6) → (M3,7) → (M5,3) → (M4,6)
Job 1: (M1,8) → (M2,5) → (M4,10) → (M5,10) → (M0,10) → (M3,4)
Job 2: (M2,5) → (M3,4) → (M5,8) → (M0,9) → (M1,1) → (M4,7)
Job 3: (M1,5) → (M0,5) → (M2,5) → (M3,3) → (M4,8) → (M5,9)
Job 4: (M2,9) → (M1,3) → (M4,5) → (M5,4) → (M0,3) → (M3,1)
Job 5: (M1,3) → (M3,3) → (M5,9) → (M0,10) → (M4,4) → (M2,1)
```

Optimal makespan: 55

### 9. Flow Shop (special case)
All jobs follow same machine order:
```
Job 0: (M0, 4) → (M1, 3) → (M2, 2)
Job 1: (M0, 2) → (M1, 4) → (M2, 3)
Job 2: (M0, 3) → (M1, 2) → (M2, 4)
```

## Resource-Constrained Project Scheduling (RCPSP)

### 10. Simple Project
4 tasks with precedence and resource constraints:
```
Task A: duration=3, workers=2, predecessors=[]
Task B: duration=2, workers=2, predecessors=[]
Task C: duration=4, workers=1, predecessors=[A]
Task D: duration=2, workers=2, predecessors=[A, B]
```
Available workers: 3

Expected: makespan = 7

### 11. Construction Project
10 tasks representing a construction project:
- Foundation → Framing → Roofing → Exterior
- Foundation → Plumbing → Interior
- Foundation → Electrical → Interior
- Interior → Finishing

Resources: workers=5, equipment=2

### 12. Software Development
Sprint planning with dependencies:
- Design (2 days, 2 devs)
- Backend API (5 days, 3 devs, after Design)
- Frontend UI (4 days, 2 devs, after Design)
- Integration (3 days, 2 devs, after Backend and Frontend)
- Testing (2 days, 1 dev, after Integration)

Team size: 4 developers

## Flexible Job Shop

### 13. Simple Flexible
2 jobs, 2 machines, tasks can run on either:
```
Job 0: Task 0 [(M0, 3), (M1, 2)], Task 1 [(M1, 4)]
Job 1: Task 0 [(M0, 2), (M1, 3)], Task 1 [(M0, 3)]
```

Expected: makespan = 6

### 14. Parallel Machines
3 identical machines, 5 jobs with single task each.
Minimize makespan by load balancing.

### 15. Unrelated Machines
Tasks have different durations on different machines.
Some machine-task combinations infeasible.
