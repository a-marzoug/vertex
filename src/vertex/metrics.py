"""Prometheus metrics for Vertex MCP Server."""

import inspect
import time
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

P = ParamSpec("P")
R = TypeVar("R")

# Metrics definitions
SOLVE_REQUESTS = Counter(
    "vertex_solve_requests_total",
    "Total number of optimization requests",
    ["tool", "status"],
)

SOLVE_DURATION = Histogram(
    "vertex_solve_duration_seconds",
    "Time taken to solve optimization problems",
    ["tool"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

PROBLEM_SIZE_VARS = Histogram(
    "vertex_problem_variables",
    "Number of variables in optimization problems",
    ["tool"],
    buckets=(10, 100, 1000, 10000, 100000, float("inf")),
)

PROBLEM_SIZE_CONSTRAINTS = Histogram(
    "vertex_problem_constraints",
    "Number of constraints in optimization problems",
    ["tool"],
    buckets=(10, 100, 1000, 10000, 100000, float("inf")),
)


def track_solve_metrics(
    tool_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to track solve duration and status. Supports sync and async."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            status = "error"
            try:
                result = await func(*args, **kwargs)  # type: ignore
                status = _extract_status(result)
                return result
            except Exception:
                status = "exception"
                raise
            finally:
                _record_metrics(tool_name, status, start_time)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            status = "error"
            try:
                result = func(*args, **kwargs)
                status = _extract_status(result)
                return result
            except Exception:
                status = "exception"
                raise
            finally:
                _record_metrics(tool_name, status, start_time)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _extract_status(result: Any) -> str:
    if hasattr(result, "status"):
        return str(result.status).lower()
    return "success"


def _record_metrics(tool_name: str, status: str, start_time: float) -> None:
    duration = time.perf_counter() - start_time
    SOLVE_DURATION.labels(tool=tool_name).observe(duration)
    SOLVE_REQUESTS.labels(tool=tool_name, status=status).inc()


def get_metrics() -> tuple[bytes, str]:
    """Get latest metrics in Prometheus format."""
    return generate_latest(), CONTENT_TYPE_LATEST
