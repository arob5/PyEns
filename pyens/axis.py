"""Named ensemble dimensions."""

from __future__ import annotations

from typing import Any, Hashable, Sequence


class Axis:
    """A named dimension along which ensemble inputs vary.

    Two ``Grid`` fields that reference the **same** ``Axis`` instance are
    *aligned* (zip semantics — they co-vary along that dimension). Two fields
    referencing **different** ``Axis`` instances are *crossed* (Cartesian
    product). Identity is therefore object identity, not name equality.

    Args:
        name: Human-readable identifier for this dimension. Used in result
            coordinates, error messages, and ``describe()`` output.
        labels: Sequence of unique, hashable labels — one per position. If
            omitted, integer indices ``0, 1, ..., size-1`` are used as labels.
        size: Number of positions. Required when ``labels`` is not provided;
            inferred automatically from ``labels`` otherwise.

    Raises:
        ValueError: If neither ``labels`` nor ``size`` is given, if both are
            given simultaneously, or if ``labels`` contains duplicates.

    Examples:
        >>> sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])
        >>> sites.size
        2
        >>> sites.label_at(0)
        'harvard_forest'

        >>> members = Axis("member", size=100)
        >>> members.label_at(42)
        42
    """

    def __init__(
        self,
        name: str,
        *,
        labels: Sequence[Hashable] | None = None,
        size: int | None = None,
    ) -> None:
        if labels is not None and size is not None:
            raise ValueError(
                f"Axis '{name}': provide 'labels' or 'size', not both."
            )
        if labels is None and size is None:
            raise ValueError(
                f"Axis '{name}': one of 'labels' or 'size' is required."
            )

        self.name = name

        if labels is not None:
            self._labels: tuple[Hashable, ...] | None = tuple(labels)
            if len(set(self._labels)) != len(self._labels):
                raise ValueError(f"Axis '{name}': labels must be unique.")
            self.size: int = len(self._labels)
        else:
            self._labels = None
            if size is not None and size < 1:
                raise ValueError(f"Axis '{name}': size must be >= 1, got {size}.")
            self.size = size  # type: ignore[assignment]

    @property
    def labels(self) -> tuple[Hashable, ...]:
        """Labels for all positions.

        Returns the explicit labels if provided, otherwise a tuple of integer
        indices ``(0, 1, ..., size-1)``.
        """
        if self._labels is not None:
            return self._labels
        return tuple(range(self.size))

    def label_at(self, index: int) -> Hashable:
        """Return the label at position ``index``.

        Args:
            index: Zero-based position index.

        Returns:
            The label at that position (explicit label or integer index).

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index < 0 or index >= self.size:
            raise IndexError(
                f"Axis '{self.name}': index {index} out of range for size {self.size}."
            )
        if self._labels is not None:
            return self._labels[index]
        return index

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        if self._labels is not None:
            return f"Axis({self.name!r}, labels={list(self._labels)!r})"
        return f"Axis({self.name!r}, size={self.size})"
