"""Field specifications that describe how each model input varies across an ensemble."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Sequence

from pyens.axis import Axis


class FieldSpec(ABC):
    """Abstract base for ensemble field specifications.

    A ``FieldSpec`` describes how one named input to the model is determined
    for each run in the ensemble. Concrete subclasses are ``Fixed`` (same value
    every run) and ``Grid`` (value indexed along one or more axes).
    """

    @property
    @abstractmethod
    def axes(self) -> tuple[Axis, ...]:
        """The ``Axis`` instances this field varies along.

        Returns an empty tuple for ``Fixed`` fields. The axes determine which
        dimensions of the ensemble coordinate space this field participates in.
        """

    @abstractmethod
    def value_at(self, idx_coords: dict[Axis, int]) -> Any:
        """Return the concrete value for a given run coordinate.

        Args:
            idx_coords: Mapping from each ``Axis`` in the ensemble to its
                zero-based integer index for this run.

        Returns:
            The concrete input value for this field at these coordinates.
        """


class Fixed(FieldSpec):
    """A field that takes the same value for every run in the ensemble.

    Args:
        value: The fixed value. Must be safe to pass to multiple workers
            concurrently — either immutable or the model callable must not
            mutate it.

    Examples:
        >>> from pyens import Fixed
        >>> f = Fixed(42)
        >>> f.value_at({})
        42
        >>> f.axes
        ()
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def axes(self) -> tuple[Axis, ...]:
        return ()

    def value_at(self, idx_coords: dict[Axis, int]) -> Any:
        return self._value

    def __repr__(self) -> str:
        return f"Fixed({self._value!r})"


class Grid(FieldSpec):
    """A field whose value is indexed along one or more axes.

    The value for a given run is looked up by indexing into ``values`` using
    the integer index of each axis at that run's coordinate.

    ``values`` may be provided as a **sequence**, a **mapping**, or any
    **nesting of sequences and mappings** — one nesting level per axis,
    outermost first:

    - **Sequence** at a level: values are assigned positionally, in the same
      order as the axis labels. The length must equal the axis size at that
      level.
    - **Mapping** at a level: values are assigned by label name. The axis at
      that level must have been created with explicit labels (``labels=[...]``).
      Keys must match the axis labels exactly. Order does not matter.

    For a two-axis grid this gives four combinations:

    - **List of lists** — both axes positional.
    - **Dict of lists** — outer axis labeled, inner axis positional.
    - **List of dicts** — outer axis positional, inner axis labeled.
    - **Dict of dicts** — both axes labeled.

    Two ``Grid`` fields that share the same ``Axis`` **instance** are aligned
    (zip semantics). Two ``Grid`` fields on **different** ``Axis`` instances
    contribute independent dimensions and are crossed as a Cartesian product.

    Args:
        values: A sequence or mapping at each nesting level, one level per
            axis. The outermost level corresponds to the first axis.
        along: A single ``Axis``, or a list of ``Axis`` objects defining the
            indexing dimensions. At least one axis is required.

    Raises:
        ValueError: If ``along`` is empty; if the length of a sequence at any
            level does not match the corresponding axis size; if a mapping is
            used at a level whose axis has no explicit labels; or if a
            mapping's keys do not exactly match the axis labels.

    Examples:
        Positional sequence::

            >>> from pyens import Axis, Grid
            >>> sites = Axis("site", labels=["s1", "s2", "s3"])
            >>> g = Grid(["cold", "hot", "wet"], along=sites)
            >>> g.value_at({sites: 1})
            'hot'

        Label-keyed mapping (order-independent, self-documenting)::

            >>> g = Grid({"s3": "wet", "s1": "cold", "s2": "hot"}, along=sites)
            >>> g.value_at({sites: 0})
            'cold'

        Multi-axis grid — dict of lists (outer axis labeled, inner positional)::

            >>> members = Axis("member", size=2)
            >>> g2 = Grid(
            ...     {"s2": ["c", "d"], "s1": ["a", "b"], "s3": ["e", "f"]},
            ...     along=[sites, members],
            ... )
            >>> g2.value_at({sites: 2, members: 0})
            'e'
    """

    def __init__(
        self,
        values: Any,
        *,
        along: Axis | Sequence[Axis],
    ) -> None:
        if isinstance(along, Axis):
            self._axes: tuple[Axis, ...] = (along,)
        else:
            self._axes = tuple(along)

        if len(self._axes) == 0:
            raise ValueError("Grid: 'along' must specify at least one Axis.")

        self._values: Any = self._resolve_at_level(values, 0)

    def _resolve_at_level(self, values: Any, axis_index: int) -> list:
        """Recursively resolve *values* at *axis_index* into a plain list.

        Each nesting level may independently be a sequence (positional) or a
        mapping (label-keyed). The resolved list is always positionally ordered
        to match axis label order so that ``value_at`` can index with a plain
        integer at every level.
        """
        axis = self._axes[axis_index]
        is_last = axis_index == len(self._axes) - 1

        if isinstance(values, Mapping):
            if axis._labels is None:
                raise ValueError(
                    f"Grid: a mapping at axis '{axis.name}' requires explicit "
                    f"labels, but this axis was created with size={axis.size} "
                    f"(integer-indexed only). Re-create the axis with "
                    f"labels=[...] to use label-keyed assignment."
                )
            provided = set(values.keys())
            expected = set(axis.labels)
            missing = expected - provided
            extra = provided - expected
            if missing:
                raise ValueError(
                    f"Grid: mapping for axis '{axis.name}' is missing values "
                    f"for labels: {sorted(str(k) for k in missing)}."
                )
            if extra:
                raise ValueError(
                    f"Grid: mapping for axis '{axis.name}' contains unexpected "
                    f"labels: {sorted(str(k) for k in extra)}."
                )
            ordered = [values[label] for label in axis.labels]
        else:
            try:
                n = len(values)
            except TypeError:
                raise ValueError(
                    f"Grid: values at axis '{axis.name}' must support len() "
                    f"(expected a sequence of length {axis.size})."
                )
            if n != axis.size:
                raise ValueError(
                    f"Grid: axis '{axis.name}' has size {axis.size} but "
                    f"values has length {n}."
                )
            ordered = list(values)

        if is_last:
            return ordered
        return [self._resolve_at_level(item, axis_index + 1) for item in ordered]

    @property
    def axes(self) -> tuple[Axis, ...]:
        return self._axes

    def value_at(self, idx_coords: dict[Axis, int]) -> Any:
        result = self._values
        for ax in self._axes:
            result = result[idx_coords[ax]]
        return result

    def __repr__(self) -> str:
        if len(self._axes) == 1:
            axes_repr = repr(self._axes[0])
        else:
            axes_repr = f"[{', '.join(repr(a) for a in self._axes)}]"
        return f"Grid(<{len(self._values)} values>, along={axes_repr})"
