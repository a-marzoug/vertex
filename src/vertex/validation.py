"""Input validation and resource limit enforcement for Vertex MCP Server."""

import inspect
from collections.abc import Callable, Coroutine
from enum import StrEnum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel

from vertex.config import (
    MAX_CONSTRAINTS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_JOBS,
    MAX_MACHINES,
    MAX_VARIABLES,
)


class ErrorCode(StrEnum):
    """Standardized error codes for API responses."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    SOLVER_ERROR = "SOLVER_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(message)


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    suggestion: str | None = None

    @classmethod
    def from_exception(cls, exc: ValidationError) -> "ErrorResponse":
        """Create ErrorResponse from ValidationError."""
        return cls(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            suggestion=exc.suggestion,
        )


P = ParamSpec("P")
R = TypeVar("R")


def validate_problem_size(
    max_variables: int = MAX_VARIABLES,
    max_constraints: int = MAX_CONSTRAINTS,
    variables_param: str = "variables",
    constraints_param: str = "constraints",
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """
    Decorator to validate LP/MIP problem size.
    Supports both sync and async functions.
    """

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_problem_size(
                kwargs,
                max_variables,
                max_constraints,
                variables_param,
                constraints_param,
            )
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_problem_size(
                kwargs,
                max_variables,
                max_constraints,
                variables_param,
                constraints_param,
            )
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_problem_size(
    kwargs: dict[str, Any],
    max_variables: int,
    max_constraints: int,
    variables_param: str,
    constraints_param: str,
) -> None:
    variables = kwargs.get(variables_param, [])
    constraints = kwargs.get(constraints_param, [])

    if isinstance(variables, list) and len(variables) > max_variables:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many variables: {len(variables)} exceeds limit of {max_variables}",
            details={
                "provided": len(variables),
                "limit": max_variables,
                "parameter": variables_param,
            },
            suggestion=f"Reduce the number of variables to {max_variables} or fewer. "
            "Consider problem decomposition for larger instances.",
        )

    if isinstance(constraints, list) and len(constraints) > max_constraints:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many constraints: {len(constraints)} exceeds limit of {max_constraints}",
            details={
                "provided": len(constraints),
                "limit": max_constraints,
                "parameter": constraints_param,
            },
            suggestion=f"Reduce the number of constraints to {max_constraints} or fewer. "
            "Consider constraint aggregation or decomposition.",
        )


def validate_graph_size(
    max_nodes: int = MAX_GRAPH_NODES,
    max_edges: int = MAX_GRAPH_EDGES,
    nodes_param: str = "nodes",
    edges_param: str = "arcs",
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to validate network/graph problem size."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_graph_size(kwargs, max_nodes, max_edges, nodes_param, edges_param)
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_graph_size(kwargs, max_nodes, max_edges, nodes_param, edges_param)
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_graph_size(
    kwargs: dict[str, Any],
    max_nodes: int,
    max_edges: int,
    nodes_param: str,
    edges_param: str,
) -> None:
    nodes = kwargs.get(nodes_param, [])
    edges = kwargs.get(edges_param, [])

    if isinstance(nodes, list) and len(nodes) > max_nodes:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many nodes: {len(nodes)} exceeds limit of {max_nodes}",
            details={
                "provided": len(nodes),
                "limit": max_nodes,
                "parameter": nodes_param,
            },
            suggestion="Consider graph partitioning or hierarchical approaches.",
        )

    if isinstance(edges, list) and len(edges) > max_edges:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many edges: {len(edges)} exceeds limit of {max_edges}",
            details={
                "provided": len(edges),
                "limit": max_edges,
                "parameter": edges_param,
            },
            suggestion="Consider sparse graph representation or edge filtering.",
        )


def validate_scheduling_size(
    max_jobs: int = MAX_JOBS,
    max_machines: int = MAX_MACHINES,
    jobs_param: str = "jobs",
    machines_param: str = "machines",
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to validate scheduling problem size."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_scheduling_size(
                kwargs, max_jobs, max_machines, jobs_param, machines_param
            )
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_scheduling_size(
                kwargs, max_jobs, max_machines, jobs_param, machines_param
            )
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_scheduling_size(
    kwargs: dict[str, Any],
    max_jobs: int,
    max_machines: int,
    jobs_param: str,
    machines_param: str,
) -> None:
    jobs = kwargs.get(jobs_param, [])
    machines = kwargs.get(machines_param, [])

    if isinstance(jobs, list) and len(jobs) > max_jobs:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many jobs: {len(jobs)} exceeds limit of {max_jobs}",
            details={
                "provided": len(jobs),
                "limit": max_jobs,
                "parameter": jobs_param,
            },
            suggestion="Consider job batching or hierarchical scheduling.",
        )

    machine_count = (
        len(machines)
        if isinstance(machines, list)
        else machines
        if isinstance(machines, int)
        else 0
    )
    if machine_count > max_machines:
        raise ValidationError(
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            message=f"Too many machines: {machine_count} exceeds limit of {max_machines}",
            details={
                "provided": machine_count,
                "limit": max_machines,
                "parameter": machines_param,
            },
            suggestion="Consider machine grouping or pooling.",
        )


def validate_positive(
    *param_names: str,
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to validate that specified parameters are positive numbers."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_positive(kwargs, param_names)
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_positive(kwargs, param_names)
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_positive(kwargs: dict[str, Any], param_names: tuple[str, ...]) -> None:
    for name in param_names:
        value = kwargs.get(name)
        if value is not None and isinstance(value, (int, float)):
            if value <= 0:
                raise ValidationError(
                    code=ErrorCode.VALIDATION_ERROR,
                    message=f"Parameter '{name}' must be positive, got {value}",
                    details={"parameter": name, "value": value},
                    suggestion=f"Provide a positive value for '{name}'.",
                )


def validate_non_negative(
    *param_names: str,
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to validate that specified parameters are non-negative."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_non_negative(kwargs, param_names)
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_non_negative(kwargs, param_names)
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_non_negative(kwargs: dict[str, Any], param_names: tuple[str, ...]) -> None:
    for name in param_names:
        value = kwargs.get(name)
        if value is not None and isinstance(value, (int, float)):
            if value < 0:
                raise ValidationError(
                    code=ErrorCode.VALIDATION_ERROR,
                    message=f"Parameter '{name}' must be non-negative, got {value}",
                    details={"parameter": name, "value": value},
                    suggestion=f"Provide a non-negative value for '{name}'.",
                )


def validate_timeout(
    timeout_param: str = "time_limit_ms",
    max_timeout_ms: int = 3_600_000,  # 1 hour
) -> Callable[[Callable[P, R]], Callable[P, R | Coroutine[Any, Any, R]]]:
    """Decorator to validate and cap timeout values."""

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R | Coroutine[Any, Any, R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_timeout(kwargs, timeout_param, max_timeout_ms)
            return await func(*args, **kwargs)  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_timeout(kwargs, timeout_param, max_timeout_ms)
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def _check_timeout(
    kwargs: dict[str, Any], timeout_param: str, max_timeout_ms: int
) -> None:
    timeout = kwargs.get(timeout_param)
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid timeout: {timeout}. Must be a positive number.",
                details={"parameter": timeout_param, "value": timeout},
                suggestion="Provide a positive timeout value in milliseconds.",
            )
        if timeout > max_timeout_ms:
            # Cap the timeout rather than rejecting
            kwargs[timeout_param] = max_timeout_ms
