# Maintenance Planning Test Scenarios

This document contains test scenarios for verifying the maintenance and replacement optimization capabilities. Feed these prompts to an LLM to test the tools.

## Equipment Replacement (MDP)

### Scenario 1: Machine Replacement Policy

**Prompt:**
"I have a machine that deteriorates over time.
**States**:
- 0: Good (New)
- 1: Minor Wear
- 2: Major Wear
- 3: Failure

**Transition Probabilities (if Kept):**
- From 0: 0.8->0, 0.2->1
- From 1: 0.7->1, 0.3->2
- From 2: 0.6->2, 0.4->3
- From 3: 1.0->3 (Absorbing)

**Costs:**
- Operating Cost: State 0 ($100), State 1 ($150), State 2 ($300), State 3 ($0 but see penalty)
- Replacement Cost: $2000 (Resets to State 0)
- Failure Penalty: $5000 (Cost if in State 3)

**Horizon**: Plan for the next 5 years.
**Discount Factor**: 0.95

Find the optimal replacement strategy for each state and year."

*(Expected Tool: `optimize_equipment_replacement`)*

---

### Scenario 2: Fleet Maintenance

**Prompt:**
"Optimize the maintenance schedule for a fleet vehicle.
Condition states: 0=Excellent, 1=Good, 2=Fair, 3=Poor.
Degradation matrix (probability of staying same vs moving to next worse state):
- 0: [0.6, 0.4, 0, 0]
- 1: [0, 0.5, 0.5, 0]
- 2: [0, 0, 0.4, 0.6]
- 3: [0, 0, 0, 1.0]

Costs:
- Maintenance/Fuel (per period): [50, 80, 150, 400]
- Replacement: 1000
- Failure/Poor Performance penalty (State 3): 500

What should we do if the vehicle is in 'Fair' condition in Year 1 of a 3-year plan?"

*(Expected Tool: `optimize_equipment_replacement`)*
