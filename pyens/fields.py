"""Field specifications that describe how each model input varies across an ensemble."""

from __future__ import annotations

from abc import ABC, abstractmethod
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

    For a single axis, ``values`` is a flat sequence of length ``axis.size``.
    For multiple axes ``[a0, a1, ...]``, ``values`` is a nested sequence where
    ``values[i0][i1][...]`` gives the value at index ``(i0, i1, ...)``.

    Two ``Grid`` fields that share the same ``Axis`` **instance** are aligned
    (zip semantics). Two ``Grid`` fields on **different** ``Axis`` instances
    contribute independent dimensions and are crossed as a Cartesian product.

    Args:
        values: Sequence (or nested sequence) of values. The shape must match
            the sizes of the provided axes, outermost axis first.
        along: A single ``Axis``, or a list of ``Axis`` objects defining the
            indexing dimensions. At least one axis is required.

    Raises:
        ValueError: If ``along`` is empty, or if the top-level length of
            ``values`` does not match the first axis size.

    Examples:
        >>> from pyens import Axis, Grid
        >>> sites = Axis("site", labels=["s1", "s2", "s3"])
        >>> g = Grid(["cold", "hot", "wet"], along=sites)
        >>> g.value_at({sites: 1})
        'hot'

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

        self._values = values
        self._validate_top_level_shape()

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
