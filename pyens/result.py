"""Result containers for completed ensemble runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from pyens.spec import Coordinate


@dataclass(frozen=True)
class RunRecord:
    """The outcome of one model evaluation.

    Attributes:
        coordinate: The axis labels that identified this run (e.g.
            ``{'site': 'A', 'param_set': 2}``). Empty for specs with no axes.
        output: The model's return value, or the exception instance if the run
            failed. Check ``failed`` before using this value as a result.
    """

    coordinate: Coordinate
    output: Any

    @property
    def failed(self) -> bool:
        """``True`` if the model raised an exception for this run."""
        return isinstance(self.output, BaseException)


class EnsembleResult:
    """Collection of outcomes from a completed ensemble run.

    Results are stored in the same order as ``EnsembleSpec.iter_runs()``
    produces runs. Failed runs (where the model raised an exception) occupy
    their natural position in the list — the exception is stored as the
    ``output`` rather than re-raised, so the rest of the results are not
    discarded.

    Args:
        records: One ``RunRecord`` per run, in iteration order.
    """

    def __init__(self, records: list[RunRecord]) -> None:
        self._records = records

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    @property
    def n_runs(self) -> int:
        """Total number of runs (succeeded + failed)."""
        return len(self._records)

    @property
    def n_failed(self) -> int:
        """Number of runs that raised an exception."""
        return sum(1 for r in self._records if r.failed)

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    @property
    def outputs(self) -> list[Any]:
        """Model outputs in run order.

        Failed runs appear as their exception instance at their natural
        position. Check ``record.failed`` or ``n_failed`` before assuming
        every element is a valid model output.
        """
        return [r.output for r in self._records]

    @property
    def coordinates(self) -> list[Coordinate]:
        """Coordinate dicts in run order, one per run."""
        return [r.coordinate for r in self._records]

    @property
    def succeeded(self) -> list[RunRecord]:
        """Records for runs that completed without raising an exception."""
        return [r for r in self._records if not r.failed]

    @property
    def failed(self) -> list[RunRecord]:
        """Records for runs that raised an exception, in run order."""
        return [r for r in self._records if r.failed]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __getitem__(self, coordinate: Coordinate) -> RunRecord:
        """Return the record whose coordinate exactly matches *coordinate*.

        Args:
            coordinate: A dict mapping axis names to labels, e.g.
                ``{'site': 'A', 'param_set': 2}``.

        Raises:
            KeyError: If no record with that coordinate exists.
        """
        for record in self._records:
            if record.coordinate == coordinate:
                return record
        raise KeyError(coordinate)

    # ------------------------------------------------------------------
    # Iteration and length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records)

    def __repr__(self) -> str:
        return (
            f"EnsembleResult({self.n_runs} runs, {self.n_failed} failed)"
        )
