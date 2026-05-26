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

    ``values`` may be provided as either a **sequence** or a **mapping**:

    - **Sequence** (list or any indexable): values are assigned positionally,
      in the same order as the axis labels. The outermost level must have length
      equal to the first axis size; for multi-axis grids each subsequent level
      indexes the next axis.
    - **Mapping** (dict or any ``Mapping``): values are assigned by label name.
      Only supported for single-axis grids whose axis was created with explicit
      labels (``labels=[...]``). Keys must match the axis labels exactly. Order
      does not matter.

    Two ``Grid`` fields that share the same ``Axis`` **instance** are aligned
    (zip semantics). Two ``Grid`` fields on **different** ``Axis`` instances
    contribute independent dimensions and are crossed as a Cartesian product.

    Args:
        values: Sequence, nested sequence, or label-keyed mapping of values.
            The shape must match the sizes of the provided axes, outermost
            axis first. A mapping is only supported for single-axis grids
            with explicit labels.
        along: A single ``Axis``, or a list of ``Axis`` objects defining the
            indexing dimensions. At least one axis is required.

    Raises:
        ValueError: If ``along`` is empty; if the top-level length of a
            sequence does not match the first axis size; if a mapping is
            supplied for a multi-axis grid or an integer-labeled axis; or
            if a mapping's keys do not exactly match the axis labels.

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

        Multi-axis nested sequence::

            >>> members = Axis("member", size=2)
            >>> g2 = Grid([["a", "b"], ["c", "d"], ["e", "f"]], along=[sites, members])
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

        if isinstance(values, Mapping):
            self._values: Any = self._resolve_mapping(values)
        else:
            self._values = values

        self._validate_top_level_shape()

    def _resolve_mapping(self, mapping: Mapping) -> list:
        """Convert a label-keyed mapping to a positionally-ordered list.

        Only valid for single-axis grids with explicit labels.
        """
        if len(self._axes) != 1:
            raise ValueError(
                "Grid: a mapping can only be used for single-axis grids. "
                "For multi-axis grids, provide a nested sequence."
            )
        axis = self._axes[0]
        if axis._labels is None:
            raise ValueError(
                f"Grid: a mapping requires an axis with explicit labels, but "
                f"axis '{axis.name}' was created with size={axis.size} "
                f"(integer-indexed only). Re-create the axis with labels=[...] "
                f"to use label-keyed assignment."
            )
        provided = set(mapping.keys())
        expected = set(axis.labels)
        missing = expected - provided
        extra = provided - expected
        if missing:
            raise ValueError(
                f"Grid: mapping is missing values for labels: "
                f"{sorted(str(k) for k in missing)}."
            )
        if extra:
            raise ValueError(
                f"Grid: mapping contains unexpected labels: "
                f"{sorted(str(k) for k in extra)}."
            )
        return [mapping[label] for label in axis.labels]

    def _validate_top_level_shape(self) -> None:
        """Validate that the outermost length of values matches the first axis."""
        first_axis = self._axes[0]
        try:
            n = len(self._values)
        except TypeError:
            raise ValueError(
                f"Grid: values must support len() to be indexed along axis "
                f"'{first_axis.name}' (expected a sequence of length {first_axis.size})."
            )
        if n != first_axis.size:
            raise ValueError(
                f"Grid: axis '{first_axis.name}' has size {first_axis.size} "
                f"but values has length {n}."
            )

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
