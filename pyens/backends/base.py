"""Abstract base class for execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any


class Backend(ABC):
    """Abstract execution backend.

    A backend is responsible for applying a model callable to a sequence of
    input dicts and returning the results in the same order. Exceptions raised
    by the model are **not** re-raised by the backend — they are caught and
    returned in place as exception instances, so one failing run does not
    prevent the others from completing.

    Implementors must define :meth:`map`. Everything else — scheduling,
    parallelism, serialisation — is the backend's concern. The rest of the
    library only depends on this interface.
    """

    @abstractmethod
    def map(
        self,
        fn: Callable[..., Any],
        runs: Iterable[dict[str, Any]],
    ) -> list[Any]:
        """Apply ``fn(**inputs)`` to each element of *runs*.

        Args:
            fn: The model callable. Invoked as ``fn(**inputs)`` for each run.
            runs: An iterable of input dicts, one per run. Each dict maps
                field names to their concrete values for that run.

        Returns:
            A list of results in the same order as *runs*. Each element is
            either the return value of ``fn`` or the exception instance raised
            by ``fn`` for that run.
        """
