"""Models for solver selection and self-tuning."""

from pydantic import BaseModel, Field


class ProblemCharacteristics(BaseModel):
    """Characteristics of an optimization problem."""

    num_variables: int
    num_constraints: int
    has_integer_variables: bool = False
    has_binary_variables: bool = False
    is_nonlinear: bool = False
    is_quadratic: bool = False
    is_network_flow: bool = False
    is_scheduling: bool = False
    is_routing: bool = False
    has_uncertainty: bool = False
    sparsity: float | None = Field(
        None, description="Fraction of zero coefficients (0-1)"
    )


class SolverRecommendation(BaseModel):
    """Recommended solver and approach for a problem."""

    recommended_tool: str = Field(description="Primary tool to use")
    alternative_tools: list[str] = Field(
        default_factory=list, description="Alternative approaches"
    )
    reasoning: str = Field(description="Why this tool was recommended")
    solver_hints: dict[str, str] = Field(
        default_factory=dict, description="Solver-specific configuration hints"
    )
    expected_performance: str = Field(
        description="Expected solve time category: fast/medium/slow"
    )
