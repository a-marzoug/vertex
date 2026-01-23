# Domain Template Test Scenarios

This document contains test scenarios for verifying the Vertex MCP server's high-level domain templates using realistic industrial contexts.

## Production Planning

### Scenario 1: Petrochemical Refining

**Prompt:**
"Optimize the production mix for a refinery.
**Products**: Gasoline, Diesel, Jet Fuel.
**Prices**: Gas $800/ton, Diesel $750/ton, Jet Fuel $900/ton.
**Resources (Crude Oil Components):**

- Light Crude: 5000 barrels available.
- Heavy Crude: 3000 barrels available.
**Requirements:**
- Gasoline: 0.6 Light, 0.4 Heavy (Processing Time: 2h/ton)
- Diesel: 0.4 Light, 0.6 Heavy (Processing Time: 1.5h/ton)
- Jet Fuel: 0.8 Light, 0.2 Heavy (Processing Time: 2.5h/ton)
**Capacity**: 8000 hours total processing time.
Maximize profit."
*(Expected Tool: `optimize_production_plan`)*

## Portfolio Optimization

### Scenario 2: Institutional Asset Allocation

**Prompt:**
"Allocate a $1B pension fund across asset classes.
**Assets & Expected Returns**:

- US Equities: 8%
- Int'l Equities: 9%
- Gov Bonds: 3%
- Corp Bonds: 5%
- Real Estate: 7%
**Constraints (Investment Policy):**
- Min 30% in Bonds (Gov + Corp)
- Max 50% in Equities
- Max 10% in Real Estate
- Min 5% Cash (Yield 1%)
Maximize expected return."
*(Expected Tool: `optimize_investment_portfolio`)*

## Inventory Management

### Scenario 3: Spare Parts Inventory (EOQ)

**Prompt:**
"Calculate the optimal order quantity for aircraft brake pads.

- **Annual Usage**: 2,500 units
- **Ordering Cost**: $250 per order (admin + shipping)
- **Unit Cost**: $800
- **Holding Cost**: 15% of unit cost per year
- **Lead Time**: 14 days
Calculate EOQ and Reorder Point."
*(Expected Tool: `optimize_inventory_eoq`)*

## Workforce Scheduling

### Scenario 4: Nurse Rostering

**Prompt:**
"Schedule nurses for a 3-shift hospital ward (Day, Evening, Night) over 3 days.
**Staff**: [N1, N2, N3, N4, N5, N6]
**Requirements**:

- Day: 2 nurses
- Evening: 2 nurses
- Night: 1 nurse
**Constraints**:
- Max 1 shift per day per nurse.
- Min 12 hours rest between shifts (cannot do Night then Day).
- Minimize total cost (Day=$300, Eve=$350, Night=$450)."
*(Expected Tool: `optimize_workforce`)*

## Healthcare Resource Allocation

### Scenario 5: Pandemic Ventilator Allocation

**Prompt:**
"Allocate 100 ventilators to 4 regional hospitals based on critical case projections.
**Hospitals**:

- North: 40 critical patients
- South: 30 critical patients
- East: 50 critical patients
- West: 20 critical patients
**Objective**: Maximize the number of patients treated (Coverage).
**Constraint**: Ensure every hospital gets at least 50% of their demand met."
*(Expected Tool: `optimize_healthcare_resources`)*

## Supply Chain Network Design

### Scenario 6: Global Distribution Network

**Prompt:**
"Design a logistics network to serve European markets.
**Potential DCs**: Rotterdam (Cost $1M), Hamburg (Cost $1.2M), Antwerp (Cost $0.9M).
**Markets (Demand)**: Paris (100), Berlin (120), Milan (80).
**Transport Costs ($/unit)**:

- Rot -> Par: 5, Ber: 6, Mil: 10
- Ham -> Par: 7, Ber: 3, Mil: 9
- Ant -> Par: 4, Ber: 7, Mil: 11
Minimize Total Fixed + Variable Costs."
*(Expected Tool: `optimize_supply_chain_network`)*

## Assignment Problems

### Scenario 7: Airport Gate Assignment

**Prompt:**
"Assign arriving flights to gates to minimize passenger walking distance.
**Flights**: F1 (Heavy), F2 (Medium), F3 (Light).
**Gates**: G1 (Far), G2 (Mid), G3 (Near).
**Passenger Volume**: F1=300, F2=150, F3=50.
**Distance from Terminal**: G1=1000m, G2=500m, G3=100m.
Assign 1 flight to 1 gate."
*(Expected Tool: `optimize_worker_assignment`)*

## Facility Location

### Scenario 8: Fire Station Location

**Prompt:**
"Select 2 sites for new fire stations to minimize average response time to 5 neighborhoods.
**Sites**: S1, S2, S3, S4.
**Neighborhoods**: N1..N5.
**Response Times**: (Ask LLM to assume a distance matrix).
**Constraint**: Max response time to any neighborhood must be < 10 mins."
*(Expected Tool: `optimize_facility_locations`)*

## Knapsack

### Scenario 9: IT Project Portfolio

**Prompt:**
"Select IT upgrade projects for the upcoming fiscal year.
**Budget**: $200,000.
**Projects**:

- Server Upgrade: Cost $80k, Value Score 90
- Cloud Migration: Cost $120k, Value Score 150
- Security Patching: Cost $30k, Value Score 60
- New CRM: Cost $90k, Value Score 100
Maximize total Value Score."
*(Expected Tool: `optimize_knapsack_selection`)*
