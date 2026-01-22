"""Structured logging configuration for Vertex MCP Server."""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> structlog.BoundLogger:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON logs (for production).
                     If False, output colored console logs (for development).

    Returns:
        Configured logger instance.
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Common processors for all environments
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # Production: JSON output
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional logger name (typically __name__)

    Returns:
        Bound logger instance.
    """
    logger: structlog.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


def log_solve_start(
    logger: structlog.BoundLogger,
    tool_name: str,
    num_variables: int,
    num_constraints: int,
    **extra: Any,
) -> None:
    """Log the start of a solve operation."""
    logger.info(
        "solve_started",
        tool=tool_name,
        num_variables=num_variables,
        num_constraints=num_constraints,
        **extra,
    )


def log_solve_complete(
    logger: structlog.BoundLogger,
    tool_name: str,
    status: str,
    solve_time_ms: float | None,
    objective_value: float | None = None,
    **extra: Any,
) -> None:
    """Log the completion of a solve operation."""
    logger.info(
        "solve_completed",
        tool=tool_name,
        status=status,
        solve_time_ms=solve_time_ms,
        objective_value=objective_value,
        **extra,
    )


def log_solve_error(
    logger: structlog.BoundLogger,
    tool_name: str,
    error: Exception,
    **extra: Any,
) -> None:
    """Log a solve error."""
    logger.error(
        "solve_error",
        tool=tool_name,
        error_type=type(error).__name__,
        error_message=str(error),
        **extra,
        exc_info=True,
    )
