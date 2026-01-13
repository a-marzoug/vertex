# Scheduling & Routing Test Scenarios

This document contains test scenarios for validating the Vertex MCP server's scheduling and routing capabilities. Feed these prompts to an LLM to test the tools.

## Traveling Salesman Problem (TSP)

### Scenario 1: Four Cities Tour

**Prompt:**
"I am planning a sales trip. I need to visit 4 cities (A, B, C, D) and return to the start.
The distances between cities are given in this matrix:

- **A to**: B=10, C=15, D=20
- **B to**: A=10, C=35, D=25
- **C to**: A=15, B=35, D=30
- **D to**: A=20, B=25, C=30

Find the shortest route that visits every city exactly once and returns to the origin."

*(Expected Tool: `solve_tsp`)*

---

### Scenario 2: Symmetric 5-City Router

**Prompt:**
"Optimize the route for a delivery truck visiting locations 0 through 4.
The distance matrix is:

```
     0   1   2   3   4
0 [  0, 29, 82, 46, 68]
1 [ 29,  0, 55, 46, 42]
2 [ 82, 55,  0, 68, 46]
3 [ 46, 46, 68,  0, 82]
4 [ 68, 42, 46, 82,  0]
```

What is the minimum total distance to visit all nodes?"

*(Expected Tool: `solve_tsp`)*

---

## Vehicle Routing Problem (VRP)

### Scenario 3: Simple Delivery Fleet

**Prompt:**
"I have 2 delivery trucks, each with a capacity of 30 units.
I need to deliver goods from the Depot to 4 customers (C1, C2, C3, C4).

**Demands:**

- Depot: 0
- C1: 10 units
- C2: 15 units
- C3: 10 units
- C4: 20 units

**Distances:**

- Depot to: C1=10, C2=15, C3=20, C4=25
- C1 to: C2=35, C3=25, C4=30
- C2 to: C3=30, C4=20
- C3 to: C4=15
*(Assume symmetric distances, e.g., C2 to C1 is also 35)*

Plan the routes for the 2 trucks to minimize total distance traveled."

*(Expected Tool: `solve_vrp`)*

---

### Scenario 4: Unbalanced Demands

**Prompt:**
"We have a fleet of 3 small vans (Capacity = 20 each).
We need to serve 5 delivery points with these demands:

- P1: 5
- P2: 8
- P3: 12
- P4: 15
- P5: 10
*(Depot has demand 0)*

**Distance Data:**
Assume a simple linear distance model where distance between any two points `i` and `j` is `abs(i - j) * 10`.
(e.g., Depot=0, P1=1, P2=2... so Depot->P1 is 10, P1->P3 is 20).

Assign customers to vehicles to minimize total fleet usage costs."

*(Expected Tool: `solve_vrp`)*

---

## VRP with Time Windows

### Scenario 5: Morning Deliveries

**Prompt:**
"Schedule deliveries for 2 vehicles (Capacity 100 each).
The Depot opens at time 0.

**Customers & Constraints:**

- **C1**: Needs 10 units. Time Window: [10, 30] (Must arrive between time 10 and 30).
- **C2**: Needs 20 units. Time Window: [20, 50].
- **C3**: Needs 15 units. Time Window: [30, 60].

**Travel Times:**

- Depot to any customer: 10 mins
- Customer to Customer: 15 mins

Find a valid route schedule that respects all time windows."

*(Expected Tool: `solve_vrp_time_windows`)*

---

## Job Shop Scheduling

### Scenario 6: Minimal 3x3 Job Shop

**Prompt:**
"Schedule 3 jobs on 3 machines (M0, M1, M2) to minimize the total completion time (makespan).

**Job Sequences (Machine, Duration):**

- **Job 0**: (M0, 3) -> (M1, 2) -> (M2, 2)
- **Job 1**: (M0, 2) -> (M2, 1) -> (M1, 4)
- **Job 2**: (M1, 4) -> (M2, 3)
*(Note: Job 2 only has 2 steps)*

Please output the optimal schedule."

*(Expected Tool: `solve_job_shop`)*

---

### Scenario 7: FT06 Benchmark Challenge

**Prompt:**
"Solve the classic Fisher and Thompson 6x6 instance (FT06).
There are 6 jobs and 6 machines.

**Job Data:**

- Job 0: (M2,1) -> (M0,3) -> (M1,6) -> (M3,7) -> (M5,3) -> (M4,6)
- Job 1: (M1,8) -> (M2,5) -> (M4,10) -> (M5,10) -> (M0,10) -> (M3,4)
- Job 2: (M2,5) -> (M3,4) -> (M5,8) -> (M0,9) -> (M1,1) -> (M4,7)
- Job 3: (M1,5) -> (M0,5) -> (M2,5) -> (M3,3) -> (M4,8) -> (M5,9)
- Job 4: (M2,9) -> (M1,3) -> (M4,5) -> (M5,4) -> (M0,3) -> (M3,1)
- Job 5: (M1,3) -> (M3,3) -> (M5,9) -> (M0,10) -> (M4,4) -> (M2,1)

Find the schedule that achieves the minimum makespan (Target: 55)."

*(Expected Tool: `solve_job_shop`)*

---

## Resource-Constrained Project Scheduling (RCPSP)

### Scenario 8: Simple Project with Dependencies

**Prompt:**
"Plan a small project with 4 tasks. We have a pool of **3 workers**.

**Task Details:**

1. **Task A**: Duration 3 days. Needs 2 workers. No predecessors.
2. **Task B**: Duration 2 days. Needs 2 workers. No predecessors.
3. **Task C**: Duration 4 days. Needs 1 worker. Must start after A matches.
4. **Task D**: Duration 2 days. Needs 2 workers. Must start after both A and B.

What is the shortest possible project duration?"

*(Expected Tool: `solve_rcpsp`)*

---

### Scenario 9: Software Development Sprint

**Prompt:**
"Optimize our sprint schedule. We have a team of **4 developers**.

**Tasks & Dependencies:**

- **Design**: 2 days, 2 devs. (Start immediately)
- **Backend API**: 5 days, 3 devs. (Must follow Design)
- **Frontend UI**: 4 days, 2 devs. (Must follow Design)
- **Integration**: 3 days, 2 devs. (Must follow Backend AND Frontend)
- **Testing**: 2 days, 1 dev. (Must follow Integration)

Calculate the minimum time to complete the sprint."

*(Expected Tool: `solve_rcpsp`)*

---

## Flexible Job Shop

### Scenario 10: Flexible Machines

**Prompt:**
"Schedule 2 jobs where tasks can be performed on alternative machines.

**Job 0:**

- Task 0: Can run on M0 (3 mins) OR M1 (2 mins).
- Task 1: Must run on M1 (4 mins).

**Job 1:**

- Task 0: Can run on M0 (2 mins) OR M1 (3 mins).
- Task 1: Must run on M0 (3 mins).

Minimize the makespan."

*(Expected Tool: `solve_flexible_job_shop`)*

---

## Bin Packing

### Scenario 11: Shipping Containers

**Prompt:**
"I need to pack items into containers. Each container holds a max weight of 100 kg.

**Items to Pack (Name: Weight):**

- Critter A: 45 kg
- Critter B: 55 kg
- Critter C: 20 kg
- Critter D: 30 kg
- Critter E: 50 kg
- Critter F: 40 kg
- Critter G: 60 kg

What is the minimum number of containers needed?"

*(Expected Tool: `solve_bin_packing`)*
