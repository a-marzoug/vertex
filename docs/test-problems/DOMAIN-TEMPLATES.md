# Domain Template Test Problems

## Production Planning

### 1. Two Products
Maximize profit producing chairs and tables:
- Chair profit: $45, Table profit: $80
- Wood: Chair needs 5 units, Table needs 20 units (400 available)
- Labor: Chair needs 2 hours, Table needs 5 hours (100 available)

Expected: Chairs=60, Tables=10, Profit=$4,500

### 2. Multi-Resource Manufacturing
Products: A, B, C
Resources: Machine time, Labor, Raw material
Maximize total profit.

## Diet Optimization

### 3. Minimum Cost Diet
Foods: Bread, Milk, Eggs
Nutrients: Calories, Protein, Calcium
Costs: Bread=$1, Milk=$2, Eggs=$3
Requirements: Calories≥2000, Protein≥50g, Calcium≥800mg

### 4. Balanced Diet
Add maximum limits on each food to ensure variety.

## Portfolio Optimization

### 5. Simple Portfolio
Assets: Stocks, Bonds, Cash
Expected returns: 12%, 6%, 2%
Budget: $100,000
Constraints: Max 60% stocks, Min 10% cash

### 6. Diversified Portfolio
5 assets with sector constraints:
- Tech: max 30%
- Healthcare: max 25%
- Finance: max 20%
- Energy: max 15%
- Utilities: max 20%

## Inventory (EOQ)

### 7. Single Item EOQ
- Annual demand: 10,000 units
- Ordering cost: $50 per order
- Holding cost: $2 per unit per year
- Lead time: 7 days

Expected: Q* ≈ 707 units

### 8. EOQ with Safety Stock
Same as above but add:
- Demand variability
- Service level 95%
- Safety stock calculation

## Workforce Scheduling

### 9. Weekly Schedule
Workers: Alice, Bob, Carol, Dave, Eve
Days: 7 (Mon-Sun)
Shifts: Morning, Evening
Requirements:
- Morning: [3, 3, 2, 2, 3, 4, 4]
- Evening: [2, 2, 2, 2, 2, 3, 3]

Constraints: Max 5 shifts per worker per week

### 10. Nurse Scheduling
- 10 nurses
- 3 shifts: Day, Evening, Night
- 7 days
- Requirements vary by day/shift
- Constraints: No consecutive night shifts

## Healthcare Resource Allocation

### 11. Hospital Resources
Resources: Doctors (10), Nurses (20), Beds (50)
Locations: Hospital A, Clinic B, Clinic C
Demands:
- Hospital A: 6 doctors, 12 nurses, 30 beds
- Clinic B: 3 doctors, 6 nurses, 15 beds
- Clinic C: 2 doctors, 4 nurses, 10 beds

Maximize coverage with limited resources.

### 12. Emergency Response
Allocate ambulances across districts:
- 15 ambulances available
- 8 districts with varying demand
- Response time requirements

## Supply Chain Network Design

### 13. Facility Location
Potential facilities: F1, F2, F3
Customers: C1, C2, C3, C4, C5
Fixed costs: F1=$100K, F2=$150K, F3=$120K
Capacities: F1=100, F2=150, F3=120
Demands: C1=40, C2=50, C3=30, C4=45, C5=35
Transport costs vary by facility-customer pair.

### 14. Multi-Echelon Supply Chain
- 2 plants
- 3 distribution centers
- 10 retail stores
- Decide which DCs to open
- Assign stores to DCs
- Assign DCs to plants

## Assignment Problems

### 15. Worker-Task Assignment
3 workers, 3 tasks
Cost matrix:
```
       T1  T2  T3
W1 [   5,  8,  6]
W2 [   9,  4,  7]
W3 [   6,  7,  3]
```

Expected: W1→T1, W2→T2, W3→T3, cost=12

### 16. Machine-Job Assignment
5 machines, 5 jobs
Minimize total processing time.
Some machine-job combinations infeasible.

## Facility Location

### 17. Uncapacitated Facility Location
- 4 potential facilities
- 6 customers
- Fixed costs and transport costs
- No capacity limits

### 18. Capacitated Facility Location
Same as above but each facility has max capacity.
Some customers may need to be served by multiple facilities.
