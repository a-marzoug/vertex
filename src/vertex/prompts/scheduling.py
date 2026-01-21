"""Prompts for scheduling problem formulation."""

FORMULATE_SCHEDULING_PROMPT = """You are an Operations Research expert helping formulate a scheduling problem.

Given a problem description, extract and structure the following components:

## 1. Entities to Schedule

### Jobs/Tasks
- What work needs to be done?
- How many jobs/tasks are there?
- Are there multiple operations per job?

### Machines/Resources
- What performs the work?
- How many machines/resources are available?
- Are machines identical or different?

## 2. Temporal Characteristics

### Processing Times
- How long does each task take on each machine?
- Are processing times fixed or variable?

### Time Windows (if applicable)
- Release times: When can tasks start?
- Due dates: When must tasks complete?
- Deadlines: Hard vs. soft constraints

### Precedence Constraints
- Must some tasks complete before others can start?
- Are there dependencies between jobs?

## 3. Problem Type Selection

| Problem Type | Characteristics | Tool |
|-------------|-----------------|------|
| **Job Shop** | Each job has ordered operations on specific machines | `solve_job_shop` |
| **Flexible Job Shop** | Operations can run on alternative machines | `solve_flexible_job_shop` |
| **Flow Shop** | All jobs follow same machine sequence | `solve_flow_shop` |
| **Parallel Machines** | Jobs assigned to identical parallel machines | `solve_parallel_machines` |
| **RCPSP** | Project tasks with precedence and resource constraints | `solve_rcpsp` |

## 4. Objective Function

Common scheduling objectives:
- **Makespan**: Minimize total completion time
- **Total Tardiness**: Minimize lateness sum
- **Weighted Completion**: Minimize weighted sum of completion times
- **Maximum Lateness**: Minimize worst-case delay

## 5. Data Format Examples

For `solve_job_shop`:
```json
{{
  "jobs": [
    {{
      "name": "Job1",
      "operations": [
        {{"machine": "M1", "duration": 3}},
        {{"machine": "M2", "duration": 2}}
      ]
    }}
  ],
  "machines": ["M1", "M2", "M3"]
}}
```

For `solve_rcpsp`:
```json
{{
  "tasks": [
    {{"name": "A", "duration": 5, "resources": {{"labor": 2}}}},
    {{"name": "B", "duration": 3, "resources": {{"labor": 1}}, "predecessors": ["A"]}}
  ],
  "resource_capacities": {{"labor": 3}}
}}
```

For `solve_parallel_machines`:
```json
{{
  "jobs": [
    {{"name": "J1", "processing_time": 5, "weight": 1}},
    {{"name": "J2", "processing_time": 3, "weight": 2}}
  ],
  "num_machines": 3
}}
```

---

Problem to formulate:
{problem_description}
"""


def formulate_scheduling_problem(problem_description: str) -> str:
    """Generate a prompt to help formulate a scheduling problem."""
    return FORMULATE_SCHEDULING_PROMPT.format(problem_description=problem_description)
