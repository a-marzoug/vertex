"""Prompts for Linear Programming problem formulation."""

FORMULATE_LP_PROMPT = """You are an Operations Research expert helping to formulate a Linear Programming problem.

Given a problem description, extract and structure the following components:

1. **Decision Variables**: What quantities need to be determined?
   - Identify each variable with a clear name
   - Determine bounds (usually non-negative: lower_bound=0)

2. **Objective Function**: What should be optimized?
   - Identify if we're maximizing (profit, revenue) or minimizing (cost, time)
   - Express as a linear combination of variables with coefficients

3. **Constraints**: What limitations exist?
   - Resource constraints (availability limits)
   - Demand constraints (minimum/maximum requirements)
   - Express each as: sum of (coefficient × variable) ≤/≥/= right-hand-side

Once formulated, call the `solve_lp` tool with:
- variables: list of {name, lower_bound, upper_bound}
- constraints: list of {coefficients: {var: coef}, sense: "<="|">="|"=", rhs: value}
- objective_coefficients: {var: coef}
- objective_sense: "maximize" or "minimize"

Problem to formulate:
{problem_description}"""


INTERPRET_SOLUTION_PROMPT = """You are an Operations Research expert explaining optimization results to decision makers.

Given the solution below, provide a clear business interpretation:

1. **Status**: Explain what the solver status means
   - optimal: Best possible solution found
   - feasible: A valid solution, but may not be optimal
   - infeasible: No solution satisfies all constraints
   - unbounded: Objective can be improved infinitely

2. **Objective Value**: What does this number represent in business terms?

3. **Variable Values**: What decisions should be made based on these values?

4. **Recommendations**: Actionable insights for the decision maker

Solution to interpret:
Status: {status}
Objective Value: {objective_value}
Variable Values: {variable_values}
Solve Time: {solve_time_ms}ms

Original Problem Context:
{problem_context}"""


def formulate_lp(problem_description: str) -> str:
    """Generate a prompt to help formulate an LP problem from natural language."""
    return FORMULATE_LP_PROMPT.format(problem_description=problem_description)


def interpret_solution(
    status: str,
    objective_value: float | None,
    variable_values: dict[str, float],
    solve_time_ms: float | None,
    problem_context: str = "",
) -> str:
    """Generate a prompt to interpret LP solution for decision makers."""
    return INTERPRET_SOLUTION_PROMPT.format(
        status=status,
        objective_value=objective_value,
        variable_values=variable_values,
        solve_time_ms=solve_time_ms,
        problem_context=problem_context,
    )
