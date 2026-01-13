# Analysis Tools Test Problems

## Sensitivity Analysis

### 1. Shadow Prices
Production problem:
- Maximize 3x + 4y
- Subject to: x + 2y ≤ 14 (resource A)
-             3x - y ≥ 0 (resource B)
-             x - y ≤ 2 (resource C)

Expected shadow prices:
- Resource A: ~2.33 (binding)
- Resource B: 0 (slack)
- Resource C: ~0.67 (binding)

### 2. Reduced Costs
Same problem - check reduced costs for variables at bounds.

## What-If Analysis

### 3. Resource Variation
Base problem: x + 2y ≤ 10, maximize 3x + 4y
Vary RHS from 8 to 14 in steps of 2.

Expected: Linear increase in objective until another constraint binds.

### 4. Coefficient Sensitivity
How much can objective coefficient change before solution changes?

### 5. Adding Constraints
Test impact of adding new constraints to existing solution.

## Infeasibility Diagnosis

### 6. Simple Conflict
- x ≥ 10
- x ≤ 5

Expected: Both constraints identified as conflicting.

### 7. Hidden Conflict
- x + y ≤ 10
- x ≥ 8
- y ≥ 5

Expected: All three constraints form infeasible system.

### 8. Relaxation Suggestions
For infeasible problem, suggest minimum relaxation to achieve feasibility.

### 9. Multiple IIS
Problem with multiple independent infeasible subsystems.

## Model Validation

### 10. Unbounded Detection
- Maximize x
- No upper bound on x

Expected: Status = UNBOUNDED

### 11. Degenerate Solution
Problem with multiple optimal solutions.

### 12. Numerical Issues
Problem with very large/small coefficients that might cause numerical instability.


## Constraint Programming

### 13. Sudoku - Easy
```
5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9
```

### 14. Sudoku - Hard
Minimal clues puzzle (17 clues).

### 15. N-Queens
- 4-Queens: 2 solutions
- 8-Queens: 92 solutions
- Find one valid placement

### 16. Magic Square
Fill 3x3 grid with 1-9 so all rows, columns, diagonals sum to 15.

## Multi-Objective Optimization

### 17. Profit vs Quality
- Maximize profit: 3x + 2y
- Maximize quality: x + 4y
- Subject to: x + y ≤ 10

Find Pareto frontier.

### 18. Cost vs Time
- Minimize cost
- Minimize completion time
- Resource constraints

### 19. Risk vs Return
Portfolio optimization:
- Maximize expected return
- Minimize variance (risk)
- Budget constraint


## Solution Enumeration

### 20. Multiple Optimal
Binary knapsack with ties:
- Items: A(3), B(3), C(2)
- Capacity: 5
- Multiple optimal solutions exist

### 21. Near-Optimal Pool
Find top 5 solutions within 5% of optimal.

### 22. Diverse Solutions
Find solutions that differ significantly in variable values.
