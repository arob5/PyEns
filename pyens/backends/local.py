"""Local multiprocessing execution backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from pyens.backends.base import Backend


class LocalBackend(Backend):
    """Run each model call in a separate worker process.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` to distribute runs
    across multiple CPU cores. Results are returned in the same order as the
    input runs regardless of completion order.

    .. important:: **Pickling requirement**

        Python's multiprocessing machinery serialises (pickles) the model
        callable and every field value before sending them to worker
        processes. This means:

        - The model function must be defined at **module level** — not as a
          lambda, a closure, or a method of a local class.
        - Every field value in the ``EnsembleSpec`` must be picklable. This
          rules out open file handles, database connections, GUI objects, and
          many C extension objects.

        If you encounter ``PicklingError`` or ``AttributeError`` during a
        run, switch to :class:`~pyens.backends.sequential.SequentialBackend`
        to reproduce the failure with a full traceback, then fix the
        pickling issue before returning to ``LocalBackend``.

    Args:
        n_workers: Number of worker processes. Defaults to ``None``, which
            lets :class:`ProcessPoolExecutor` choose (typically
            ``min(32, os.cpu_count() + 4)`` on Python ≥ 3.8).
    """

    def __init__(self, n_workers: int | None = None) -> None:
        self._n_workers = n_workers

    def map(
        self,
        fn: Callable[..., Any],
        runs: Iterable[dict[str, Any]],
    ) -> list[Any]:
        runs_list = list(runs)
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = [executor.submit(fn, **inputs) for inputs in runs_list]
        results: list[Any] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
        return results
