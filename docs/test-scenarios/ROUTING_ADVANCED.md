# Advanced Routing Test Scenarios

This document contains test scenarios for verifying complex routing operations (Pickup & Delivery, Metaheuristic tuning) in logistics and transport.

## Pickup and Delivery VRP (PDP)

### Scenario 1: Ride-Sharing Dispatch

**Prompt:**
"Schedule a ride-sharing service with 2 cars (Capacity 4).
**Depot**: (0,0).
**Requests (Pickup -> Delivery):**

1. **Rider A**: (2,2) -> (5,5) (1 passenger)
2. **Rider B**: (1,4) -> (4,1) (2 passengers)
3. **Rider C**: (-2,3) -> (-2,-3) (1 passenger)
**Constraint**: Pickups must occur before deliveries. Minimize total distance."
*(Expected Tool: `solve_pickup_delivery`)*

### Scenario 2: Dialysis Patient Transport

**Prompt:**
"Coordinate non-emergency medical transport for 3 patients to a dialysis center and back.
**Vehicles**: 2 Vans (Capacity 2 wheelchairs).
**Locations**:

- Depot (Garage)
- Home1, Home2, Home3
- Clinic
**Trips**:
- H1 -> Clinic
- H2 -> Clinic
- H3 -> Clinic
Minimize patient time in transit."
*(Expected Tool: `solve_pickup_delivery`)*

## Metaheuristic Configuration

### Scenario 3: PCB Drilling Path (Large TSP)

**Prompt:**
"Optimize the drilling path for a Printed Circuit Board with 500 holes (random locations).
This is a TSP. Use **Guided Local Search** to escape local minima and find a high-quality solution quickly (limit 5s)."
*(Expected Tool: `solve_tsp` with `local_search_metaheuristic='GUIDED_LOCAL_SEARCH'`)*

### Scenario 4: Large Fleet VRP with Tabu Search

**Prompt:**
"Solve a distribution problem for a major city with 50 customers and 10 trucks.
Use **Tabu Search** to explore the solution space effectively.
Compare the route stability against default settings."
*(Expected Tool: `solve_vrp` with `local_search_metaheuristic='TABU_SEARCH'`)*
