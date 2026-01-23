# Maintenance Planning Test Scenarios

This document contains test scenarios for verifying the maintenance and replacement optimization capabilities in industrial contexts.

## Equipment Replacement (MDP)

### Scenario 1: Aircraft Engine Overhaul Policy

**Prompt:**
"Determine the optimal maintenance policy for an aircraft engine component.
**States**:

- 0: New/Overhauled
- 1: Minor Wear
- 2: Significant Wear
- 3: Critical (Must Replace)
**Transitions (Yearly)**:
- 0->1: 30% prob
- 1->2: 40% prob
- 2->3: 50% prob
**Costs**:
- Annual Inspection: $5,000 (State 0-2)
- Preventive Overhaul: $50,000 (Resets to 0)
- Emergency Replacement (at State 3): $200,000
**Horizon**: 10 years.
**Discount Rate**: 5%.
Find the optimal action (Wait vs Overhaul) for each state."
*(Expected Tool: `optimize_equipment_replacement`)*

### Scenario 2: Heavy Mining Truck Maintenance

**Prompt:**
"Optimize the replacement cycle for haul trucks in a mine.
**Max Age**: 5 years.
**Purchase Cost**: $2M.
**Maintenance Costs (Year 1-5)**: [$50k, $100k, $250k, $500k, $900k].
**Resale Value (Year 1-5)**: [$1.5M, $1.1M, $800k, $500k, $200k].
**Failure Probability (Year 1-5)**: [1%, 5%, 15%, 30%, 50%].
**Failure Cost**: $500k (downtime).
**Discount Factor**: 0.9.
Determine the optimal age to replace the truck."
*(Expected Tool: `optimize_equipment_replacement`)*
