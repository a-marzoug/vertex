"""Prompts for Mixed-Integer Programming problem formulation."""

FORMULATE_MIP_PROMPT = """You are an Operations Research expert helping to formulate a Mixed-Integer Programming problem.

Given a problem description, extract and structure:

1. **Decision Variables**: Identify type for each:
   - Binary (0/1): yes/no decisions, select/don't select
   - Integer: whole number quantities (workers, machines, items)
   - Continuous: fractional values allowed

2. **Objective Function**: What to maximize/minimize?

3. **Constraints**: Resource limits, logical constraints, linking constraints

Common MIP patterns:
- **Assignment**: Binary x[i,j] = 1 if i assigned to j
- **Knapsack**: Binary x[i] = 1 if item i selected
- **Facility Location**: Binary y[j] = 1 if facility j open
- **Scheduling**: Binary x[i,t] = 1 if task i starts at time t

Once formulated, call `solve_mixed_integer_program` with:
- variables: list of {{name, var_type, lower_bound, upper_bound}}
- constraints: list of {{coefficients, sense, rhs}}
- objective_coefficients, objective_sense

Problem to formulate:
{problem_description}"""


def formulate_mip(problem_description: str) -> str:
    """Generate a prompt to help formulate a MIP problem."""
    return FORMULATE_MIP_PROMPT.format(problem_description=problem_description)
