"""Prompts for sensitivity analysis interpretation."""

INTERPRET_SENSITIVITY_PROMPT = """You are an Operations Research expert explaining sensitivity analysis results to a business decision maker.

## Understanding Sensitivity Analysis

Sensitivity analysis reveals how the optimal solution responds to changes in problem parameters. This is crucial for understanding:
- Which constraints are limiting performance (bottlenecks)
- How much improvement is possible by relaxing constraints
- Whether the current solution remains optimal under parameter changes

## Key Metrics Explained

### Shadow Prices (Dual Values)
The **shadow price** of a constraint indicates the rate of change in the objective value per unit increase in the constraint's right-hand side (RHS).

**Interpretation**:
- A shadow price of **$50/hour** for a labor constraint means:
  - Each additional hour of labor would improve profit by $50
  - This is the maximum you should pay for one more hour
- A shadow price of **$0** means the constraint is not binding (slack exists)

**Business Implications**:
- High shadow prices identify bottleneck resources
- Guides decisions on capacity expansion
- Indicates the value of acquiring additional resources

### Reduced Costs
The **reduced cost** of a variable indicates how much the objective coefficient would need to improve before that variable would become positive in the optimal solution.

**Interpretation**:
- For a product with reduced cost of **-$5**:
  - Currently not profitable to produce (value = 0 in solution)
  - Would need to increase its profit margin by $5 to consider producing
- A reduced cost of **$0** means the variable is in the optimal basis

**Business Implications**:
- Identifies unprofitable activities
- Shows required price/cost improvements
- Guides product/activity discontinuation decisions

## Analysis Summary

Given the sensitivity data below:

### Shadow Prices by Constraint
{shadow_prices}

### Reduced Costs by Variable
{reduced_costs}

### Binding Constraints (shadow price ≠ 0)
These are your bottlenecks - relaxing these would improve the objective.

### Non-Binding Constraints (shadow price = 0)
These have slack - you have unused capacity here.

### Active Variables (reduced cost = 0)
These are being used in the optimal solution.

### Inactive Variables (reduced cost ≠ 0)
These need improvement before they become attractive.

## Recommendations

1. **Resource Acquisition Priority**: Rank constraints by shadow price magnitude
2. **Pricing Decisions**: Use reduced costs to set minimum acceptable prices
3. **Sensitivity Ranges**: Check how far parameters can change before solution changes

---

Problem Context:
{problem_context}
"""


def interpret_sensitivity_analysis(
    shadow_prices: dict[str, float],
    reduced_costs: dict[str, float],
    problem_context: str = "",
) -> str:
    """Generate a prompt to interpret sensitivity analysis results."""
    # Format the dictionaries nicely
    sp_formatted = "\n".join(
        f"  - {name}: ${price:.2f}" for name, price in shadow_prices.items()
    )
    rc_formatted = "\n".join(
        f"  - {name}: ${cost:.2f}" for name, cost in reduced_costs.items()
    )

    return INTERPRET_SENSITIVITY_PROMPT.format(
        shadow_prices=sp_formatted or "  (none provided)",
        reduced_costs=rc_formatted or "  (none provided)",
        problem_context=problem_context or "(no additional context)",
    )
