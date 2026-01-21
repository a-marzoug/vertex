"""Async utilities for CPU-bound tasks."""

import asyncio
from collections.abc import Callable, Coroutine
from functools import partial, wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Global executor for CPU-bound tasks
# using ProcessPoolExecutor is safer for OR-Tools which releases GIL but
# some Python-side graph construction might not.
# For now, we use the default loop executor (ThreadPool) because OR-Tools
# releases the GIL during Solve(), so threads are sufficient and have lower overhead
# than processes for data serialization.
_executor = None


async def run_in_executor(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking function in an executor."""
    loop = asyncio.get_running_loop()
    # Partial is needed to pass kwargs
    pfunc = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_executor, pfunc)


def asyncify(func: Callable[P, R]) -> Callable[P, Coroutine[Any, Any, R]]:
    """Decorator to make a sync blocking function async."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await run_in_executor(func, *args, **kwargs)

    return wrapper
