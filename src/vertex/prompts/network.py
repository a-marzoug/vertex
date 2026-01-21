"""Prompts for network optimization problem formulation."""

FORMULATE_NETWORK_PROMPT = """You are an Operations Research expert helping formulate a network optimization problem.

Given a problem description, extract and structure the following components:

## 1. Network Structure

### Nodes
Identify all locations/entities in the network:
- **Supply nodes**: Sources with available capacity
- **Demand nodes**: Destinations with requirements  
- **Transshipment nodes**: Intermediate points for routing

### Arcs/Edges
Identify connections between nodes:
- Which nodes are connected?
- Are connections directional (arcs) or bidirectional (edges)?

## 2. Arc Attributes

For each arc, determine:
- **Capacity**: Maximum flow allowed (if applicable)
- **Cost**: Per-unit flow cost (for min-cost problems)
- **Distance/Time**: For routing problems

## 3. Node Attributes

For each node, determine:
- **Supply**: Amount available to send (positive)
- **Demand**: Amount required to receive (negative)
- **For balanced flow: total supply = total demand**

## 4. Problem Type Selection

Based on the structure, identify the appropriate tool:

| Problem Type | Key Characteristics | Tool |
|-------------|---------------------|------|
| Max Flow | Find maximum throughput | `find_max_flow` |
| Min Cost Flow | Minimize cost meeting demands | `find_min_cost_flow` |
| Shortest Path | Find optimal route | `find_shortest_path` |
| Spanning Tree | Connect all nodes minimally | `find_minimum_spanning_tree` |
| Multi-Commodity | Multiple products sharing network | `find_multi_commodity_flow` |
| Transshipment | Shipping through intermediaries | `solve_transshipment` |

## 5. Data Format

Structure the data for the selected tool:

For `find_max_flow`:
```json
{{
  "nodes": ["source", "A", "B", "sink"],
  "arcs": [
    {{"from": "source", "to": "A", "capacity": 10}},
    {{"from": "A", "to": "sink", "capacity": 5}}
  ],
  "source": "source",
  "sink": "sink"
}}
```

For `find_min_cost_flow`:
```json
{{
  "nodes": ["factory", "warehouse", "store"],
  "arcs": [
    {{"from": "factory", "to": "warehouse", "capacity": 100, "cost": 2}},
    {{"from": "warehouse", "to": "store", "capacity": 80, "cost": 1}}
  ],
  "supplies": {{"factory": 50}},
  "demands": {{"store": 50}}
}}
```

---

Problem to formulate:
{problem_description}
"""


def formulate_network_problem(problem_description: str) -> str:
    """Generate a prompt to help formulate a network optimization problem."""
    return FORMULATE_NETWORK_PROMPT.format(problem_description=problem_description)
