"""pyens — structured ensemble runner for scientific models.

Core public API:

Specification layer:

- :class:`Axis` — a named dimension along which ensemble inputs vary.
- :class:`Fixed` — a field with the same value for every run.
- :class:`Grid` — a field whose value is indexed along one or more axes.
- :class:`EnsembleSpec` — the full specification of an ensemble.
- :class:`BoundSpec` — a partially-applied spec returned by
  :meth:`EnsembleSpec.freeze`.

Execution layer:

- :class:`EnsembleRunner` — drives a model callable over an ``EnsembleSpec``.
- :class:`Backend` — abstract base for execution backends.
- :class:`SequentialBackend` — runs each model call in a plain loop (no
  parallelism; recommended for debugging).
- :class:`LocalBackend` — runs model calls in parallel worker processes via
  :mod:`concurrent.futures`.

Result layer:

- :class:`EnsembleResult` — collection of outcomes from a completed run.
- :class:`RunRecord` — the outcome (output or exception) of one model call.
"""

from __future__ import annotations

from pyens.axis import Axis
from pyens.backends import Backend, LocalBackend, SequentialBackend
from pyens.fields import FieldSpec, Fixed, Grid
from pyens.result import EnsembleResult, RunRecord
from pyens.runner import EnsembleRunner
from pyens.spec import BoundSpec, EnsembleSpec

__all__ = [
    # Specification
    "Axis",
    "FieldSpec",
    "Fixed",
    "Grid",
    "EnsembleSpec",
    "BoundSpec",
    # Execution
    "EnsembleRunner",
    "Backend",
    "SequentialBackend",
    "LocalBackend",
    # Results
    "EnsembleResult",
    "RunRecord",
]
