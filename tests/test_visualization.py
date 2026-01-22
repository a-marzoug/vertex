"""Tests for visualization helpers."""

from vertex.models.scheduling import ScheduledTask
from vertex.utils.visualization import create_gantt_chart


def test_create_gantt_chart():
    """Test converting a schedule to Gantt chart data."""
    schedule = [
        ScheduledTask(job=0, task=0, machine=1, start=0, duration=10, end=10),
        ScheduledTask(job=1, task=0, machine=2, start=5, duration=5, end=10),
    ]

    chart = create_gantt_chart(schedule, title="Test Chart")

    assert chart.title == "Test Chart"
    assert len(chart.tasks) == 2
    assert len(chart.resources) == 2

    # Check task 1
    t1 = chart.tasks[0]
    assert t1.start == 0
    assert t1.end == 10
    assert t1.resource_id == "Resource 1"
    assert "Job 0" in t1.label

    # Check resources
    assert "Resource 1" in chart.resources
    assert "Resource 2" in chart.resources
