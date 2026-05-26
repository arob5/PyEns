"""pyens — structured ensemble runner for scientific models.

Core public API:

- :class:`Axis` — a named dimension along which ensemble inputs vary.
- :class:`Fixed` — a field with the same value for every run.
- :class:`Grid` — a field whose value is indexed along one or more axes.
- :class:`EnsembleSpec` — the full specification of an ensemble.
- :class:`BoundSpec` — a partially-applied spec returned by
  :meth:`EnsembleSpec.freeze`.
"""

from __future__ import annotations

from pyens.axis import Axis
from pyens.fields import FieldSpec, Fixed, Grid
from pyens.spec import BoundSpec, EnsembleSpec

__all__ = [
    "Axis",
    "FieldSpec",
    "Fixed",
    "Grid",
    "EnsembleSpec",
    "BoundSpec",
]
