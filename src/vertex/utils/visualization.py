"""Visualization helpers for scheduling problems."""

from vertex.models.scheduling import ScheduledTask
from vertex.models.viz import GanttChart, GanttTask


def create_gantt_chart(
    schedule: list[ScheduledTask] | list[dict],
    title: str = "Schedule",
    resource_key: str = "machine",
    task_key: str = "task",
) -> GanttChart:
    """
    Generate GanttChart data from a schedule.

    Args:
        schedule: List of ScheduledTask objects or dictionaries.
        title: Chart title.
        resource_key: Key/Attribute name for the resource (row).
        task_key: Key/Attribute name for the task label.

    Returns:
        GanttChart model.
    """
    tasks: list[GanttTask] = []
    resources: set[str] = set()

    for item in schedule:
        # Handle both Pydantic models and dicts
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        else:
            data = item

        # Extract fields
        start = float(data.get("start", 0))
        end = float(data.get("end", 0))
        duration = float(data.get("duration", end - start))

        # Resource ID (e.g., Machine 1)
        res_val = data.get(resource_key)
        resource_id = (
            f"Resource {res_val}" if isinstance(res_val, int) else str(res_val)
        )
        resources.add(resource_id)

        # Task Label
        task_val = data.get(task_key)
        label = f"Task {task_val}" if isinstance(task_val, int) else str(task_val)

        # Add job info if available
        if "job" in data:
            label = f"Job {data['job']} - {label}"

        tasks.append(
            GanttTask(
                id=f"{label}_{start}",
                label=label,
                resource_id=resource_id,
                start=start,
                end=end,
                duration=duration,
            )
        )

    # Sort resources naturally if they look like "Resource X"
    sorted_resources = sorted(list(resources))

    return GanttChart(
        title=title,
        tasks=tasks,
        resources=sorted_resources,
    )
