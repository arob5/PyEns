"""Sequential (single-process) execution backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pyens.backends.base import Backend


class SequentialBackend(Backend):
    """Run each model call one at a time in the current process.

    This is the simplest possible backend: a plain Python loop with no
    parallelism. It is useful for:

    - **Debugging.** Exceptions have full tracebacks and are straightforward
      to inspect. Multiprocessing backends swallow stack frames and can make
      failures hard to diagnose.
    - **Serial workflows.** Some models manage their own parallelism
      internally (e.g. via OpenMP), or the ensemble is small enough that
      parallelism overhead outweighs the benefit.
    - **Environments where multiprocessing is unavailable.** Interactive
      notebooks, some HPC login nodes, and certain container environments
      restrict subprocess creation.

    No external dependencies are required.
    """

    def map(
        self,
        fn: Callable[..., Any],
        runs: Iterable[dict[str, Any]],
    ) -> list[Any]:
        results: list[Any] = []
        for inputs in runs:
            try:
                results.append(fn(**inputs))
            except Exception as exc:
                results.append(exc)
        return results
