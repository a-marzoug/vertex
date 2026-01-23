# Network Optimization Test Scenarios

This document contains test scenarios for verifying the Vertex MCP server's network optimization capabilities in industrial infrastructure contexts.

## Maximum Flow

### Scenario 1: Data Network Throughput

**Prompt:**
"Calculate the maximum data throughput from the Main Data Center (S) to the Backup Site (T).
**Links (Bandwidth in Gbps):**

- S -> Node A: 100
- S -> Node B: 80
- Node A -> Node B: 40
- Node A -> T: 60
- Node B -> T: 90
What is the max flow?"
*(Expected Tool: `find_max_flow`)*

## Minimum Cost Flow

### Scenario 2: Water Distribution Network

**Prompt:**
"Optimize pumping from Reservoirs to Towns.
**Supplies**: Res1 (50 ML), Res2 (70 ML).
**Demands**: TownA (40 ML), TownB (50 ML), TownC (30 ML).
**Pumping Costs ($/ML)**:

- Res1 -> A: 50, B: 60, C: 80
- Res2 -> A: 70, B: 40, C: 50
Find the minimum cost flow."
*(Expected Tool: `find_min_cost_flow`)*

## Shortest Path

### Scenario 3: Low-Latency Trading Route

**Prompt:**
"Find the lowest latency path for a trade signal from NY to Tokyo.
**Network Latencies (ms):**

- NY -> London: 160
- NY -> Paris: 70
- Paris -> Frankfurt: 10
- Frankfurt -> Tokyo: 150
- London -> Tokyo: 140
- Paris -> Tokyo: 170
Find the fastest route."
*(Expected Tool: `find_shortest_path`)*

## Minimum Spanning Tree

### Scenario 4: Offshore Wind Farm Cabling

**Prompt:**
"Connect 5 offshore wind turbines (T1..T5) to the substation (S) with minimum cable length.
**Distances (km):**

- S-T1: 5, S-T2: 8
- T1-T2: 4, T1-T3: 7
- T2-T3: 3, T2-T4: 6
- T3-T4: 4, T3-T5: 5
- T4-T5: 3
Design the cable layout."
*(Expected Tool: `find_minimum_spanning_tree`)*

## Multi-Commodity Flow

### Scenario 5: Rail Network Scheduling

**Prompt:**
"Route Freight and Passenger trains on a shared rail network.
**Capacities**: Each segment handles 50 trains/day.
**Demands**:

- Freight: 30 trains from Port -> City.
- Passenger: 25 trains from Suburb -> City.
**Shared Link**: The main line segment is the bottleneck.
Determine if both flows can be accommodated."
*(Expected Tool: `find_multi_commodity_flow`)*

## Transshipment

### Scenario 6: Humanitarian Logistics

**Prompt:**
"Plan the delivery of relief supplies from International Depots to Disaster Zones via Regional Hubs.
**Sources**: Depot A (1000 tons), Depot B (800 tons).
**Hubs**: Hub X, Hub Y.
**Targets**: Zone 1 (600 tons), Zone 2 (700 tons), Zone 3 (500 tons).
**Costs**:

- Depot->Hub: $100/ton.
- Hub->Zone: $200/ton.
Minimize shipping costs."
*(Expected Tool: `solve_transshipment`)*
