"""Visualization models."""

from pydantic import BaseModel, Field


class GanttTask(BaseModel):
    """Visual representation of a task in a Gantt chart."""

    id: str
    label: str
    resource_id: str  # Row ID (e.g., "Machine 1")
    start: float
    end: float
    duration: float
    dependencies: list[str] = Field(default_factory=list)


class GanttChart(BaseModel):
    """Structured data for plotting a Gantt chart."""

    title: str
    tasks: list[GanttTask]
    resources: list[str]  # Ordered list of resources (Y-axis)
