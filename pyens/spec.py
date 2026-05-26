"""Ensemble specification and partial-application utilities."""

from __future__ import annotations

import itertools
from typing import Any, Iterator

from pyens.axis import Axis
from pyens.fields import FieldSpec, Fixed, Grid

# Type aliases for clarity in signatures and docstrings.
RunInputs = dict[str, Any]
"""Mapping from field name to its concrete value for one run."""

Coordinate = dict[str, Any]
"""Mapping from axis name to its label for one run (for bookkeeping)."""


class EnsembleSpec:
    """Specification for a structured ensemble of model runs.

    An ``EnsembleSpec`` declares what inputs each model evaluation receives.
    Inputs are named fields, each described by a ``FieldSpec`` (``Fixed`` or
    ``Grid``). The full set of runs is determined by the unique ``Axis``
    instances across all fields:

    - Fields on the **same** ``Axis`` instance vary together (zip semantics).
    - Fields on **different** ``Axis`` instances are crossed (Cartesian product).

    The total number of runs is the product of all unique axis sizes.

    Args:
        inputs: Mapping from field name to ``FieldSpec``.

    Examples:
        Two Grid fields on the same axis → zip (2 runs)::

            sites = Axis("site", labels=["s1", "s2"])
            spec = EnsembleSpec(inputs={
                "climate": Grid([c1, c2], along=sites),
                "ic":      Grid([ic1, ic2], along=sites),  # aligned with climate
            })
            assert spec.n_runs == 2

        Two Grid fields on different axes → Cartesian product (6 runs)::

            sites   = Axis("site", labels=["s1", "s2", "s3"])
            members = Axis("member", size=2)
            spec = EnsembleSpec(inputs={
                "climate": Grid([c1, c2, c3], along=sites),
                "ic":      Grid([ic1, ic2], along=members),
            })
            assert spec.n_runs == 6

        Fixed field contributes no axis::

            spec = EnsembleSpec(inputs={
                "params":  Fixed(base_params),
                "climate": Grid([c1, c2], along=sites),
            })
            assert spec.n_runs == 2
    """

    def __init__(self, *, inputs: dict[str, FieldSpec]) -> None:
        if not inputs:
            raise ValueError("EnsembleSpec: 'inputs' must contain at least one field.")
        self._inputs: dict[str, FieldSpec] = dict(inputs)
        self._axes: tuple[Axis, ...] = self._collect_axes()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_runs(cls, *run_dicts: dict[str, Any]) -> EnsembleSpec:
        """Build a spec from a flat list of input dictionaries.

        This is a simple entry point for cases where the axis algebra is
        overkill. Each dict becomes one run. All dicts must have the same
        keys.

        Args:
            *run_dicts: One dict per run, mapping field name to value.

        Returns:
            An ``EnsembleSpec`` backed by a single anonymous axis of size
            ``len(run_dicts)``, with integer labels ``0, 1, ...``.

        Raises:
            ValueError: If no run dicts are provided, or if the dicts have
                inconsistent keys.

        Examples:
            >>> spec = EnsembleSpec.from_runs(
            ...     {"x": 1.0, "label": "a"},
            ...     {"x": 2.0, "label": "b"},
            ... )
            >>> spec.n_runs
            2
        """
        if not run_dicts:
            raise ValueError("EnsembleSpec.from_runs: at least one run dict is required.")

        keys = set(run_dicts[0].keys())
        for i, d in enumerate(run_dicts[1:], start=1):
            if set(d.keys()) != keys:
                raise ValueError(
                    f"EnsembleSpec.from_runs: run dict at index {i} has keys "
                    f"{set(d.keys())} but expected {keys}."
                )

        run_axis = Axis("run", size=len(run_dicts))
        inputs: dict[str, FieldSpec] = {
            field: Grid([d[field] for d in run_dicts], along=run_axis)
            for field in keys
        }
        return cls(inputs=inputs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def axes(self) -> tuple[Axis, ...]:
        """Unique ``Axis`` instances across all fields, in insertion order.

        This defines the coordinate space of the ensemble. The Cartesian
        product of all axes gives the full set of run coordinates.
        """
        return self._axes

    @property
    def n_runs(self) -> int:
        """Total number of model evaluations this spec will produce."""
        result = 1
        for ax in self._axes:
            result *= ax.size
        return result

    @property
    def field_names(self) -> tuple[str, ...]:
        """Names of all input fields, in insertion order."""
        return tuple(self._inputs.keys())

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable summary of the ensemble structure.

        Useful for validating the spec before committing to a large run.

        Returns:
            Multi-line string summarising axes, fields, and total run count.

        Examples:
            >>> print(spec.describe())
            EnsembleSpec: 6 runs
              Axes (2):
                Axis('site', labels=['s1', 's2', 's3'])
                Axis('member', size=2)
              Fields (2):
                climate: Grid [site]
                ic:      Grid [member]
        """
        lines = [f"EnsembleSpec: {self.n_runs} run{'s' if self.n_runs != 1 else ''}"]
        lines.append(f"  Axes ({len(self._axes)}):")
        for ax in self._axes:
            lines.append(f"    {ax!r}")
        lines.append(f"  Fields ({len(self._inputs)}):")
        for name, field in self._inputs.items():
            type_name = type(field).__name__
            ax_names = ", ".join(a.name for a in field.axes) if field.axes else "fixed"
            lines.append(f"    {name}: {type_name} [{ax_names}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_runs(self) -> Iterator[tuple[RunInputs, Coordinate]]:
        """Iterate over all runs as ``(inputs, coordinate)`` pairs.

        ``inputs`` is a dict mapping each field name to its concrete value
        for this run. ``coordinate`` maps each axis name to its label for
        this run, and is suitable for use as a result key or metadata tag.

        Yields:
            ``(inputs_dict, coordinate_dict)`` for each run, in the order
            determined by the Cartesian product of all axes.

        Examples:
            >>> for inputs, coord in spec.iter_runs():
            ...     output = model(**inputs)
            ...     print(coord, "→", output)
        """
        for idx_coords in self._iter_index_coords():
            run_inputs: RunInputs = {
                name: field.value_at(idx_coords)
                for name, field in self._inputs.items()
            }
            coordinate: Coordinate = {
                ax.name: ax.label_at(idx_coords[ax])
                for ax in self._axes
            }
            yield run_inputs, coordinate

    # ------------------------------------------------------------------
    # Point selection
    # ------------------------------------------------------------------

    def sel(self, **coords: Any) -> RunInputs:
        """Return the inputs for a specific run, selected by coordinate label.

        Looks up the concrete input values for the single run identified by
        the given coordinate. All axes in the spec must be specified.

        Args:
            **coords: One keyword argument per axis, mapping axis name to
                the desired label value.

        Returns:
            A ``RunInputs`` dict mapping each field name to its concrete value
            at the given coordinate. Equivalent to the ``inputs`` dict that
            ``iter_runs()`` would yield for this run.

        Raises:
            ValueError: If the spec contains two axes with the same name
                (use ``iter_runs()`` and filter manually in that case); if
                an unknown axis name is provided; if a provided label is not
                found on its axis; or if not all axes are specified.

        Examples:
            >>> sites   = Axis("site", labels=["A", "B"])
            >>> members = Axis("member", size=3)
            >>> spec = EnsembleSpec(inputs={
            ...     "x": Grid(["a", "b"], along=sites),
            ...     "y": Grid([10, 20, 30], along=members),
            ... })
            >>> spec.sel(site="B", member=1)
            {'x': 'b', 'y': 20}
        """
        # Build name → axis mapping, detecting duplicate names.
        axis_by_name: dict[str, Axis] = {}
        for ax in self._axes:
            if ax.name in axis_by_name:
                raise ValueError(
                    f"EnsembleSpec.sel(): ambiguous — the spec contains two axes "
                    f"named '{ax.name}'. Use iter_runs() and filter by coordinate "
                    f"manually when axis names are not unique."
                )
            axis_by_name[ax.name] = ax

        unknown = set(coords.keys()) - set(axis_by_name.keys())
        missing = set(axis_by_name.keys()) - set(coords.keys())
        if unknown:
            raise ValueError(
                f"sel(): unknown axis name(s): {sorted(unknown)}. "
                f"Available axes: {sorted(axis_by_name.keys())}."
            )
        if missing:
            raise ValueError(
                f"sel(): all axes must be specified. "
                f"Missing: {sorted(missing)}."
            )

        idx_coords: dict[Axis, int] = {}
        for name, label in coords.items():
            ax = axis_by_name[name]
            labels_list = list(ax.labels)
            try:
                idx = labels_list.index(label)
            except ValueError:
                raise ValueError(
                    f"sel(): label {label!r} not found in axis '{name}'. "
                    f"Available labels: {labels_list}."
                )
            idx_coords[ax] = idx

        return {
            name: field.value_at(idx_coords)
            for name, field in self._inputs.items()
        }

    # ------------------------------------------------------------------
    # Partial application
    # ------------------------------------------------------------------

    def freeze(self, free: list[str]) -> BoundSpec:
        """Create a partially-applied spec with certain fields left open.

        Returns a ``BoundSpec`` callable. Calling it with ``FieldSpec``
        values (or plain values, wrapped automatically in ``Fixed``) for
        the free fields returns a fully-bound ``EnsembleSpec`` ready to run.

        This is the primary mechanism for constructing parameter-to-output
        maps for use in optimisation and sampling algorithms.

        Args:
            free: Names of fields to expose as free variables. Must all be
                present in this spec.

        Returns:
            A ``BoundSpec`` that accepts the free fields and returns an
            ``EnsembleSpec``.

        Raises:
            ValueError: If any name in ``free`` is not a field in this spec.

        Examples:
            >>> bound = spec.freeze(free=["parameters"])
            >>> runnable = bound(parameters=Grid(sampled_params, along=param_axis))
            >>> for inputs, coord in runnable.iter_runs():
            ...     ...
        """
        unknown = [name for name in free if name not in self._inputs]
        if unknown:
            raise ValueError(
                f"EnsembleSpec.freeze(): unknown field name(s): {unknown}. "
                f"Available fields: {list(self._inputs.keys())}."
            )
        base_inputs = {k: v for k, v in self._inputs.items() if k not in free}
        return BoundSpec(base_inputs, free_field_names=set(free))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_axes(self) -> tuple[Axis, ...]:
        """Collect unique Axis instances across all fields, insertion-ordered."""
        seen: dict[int, Axis] = {}
        for field in self._inputs.values():
            for ax in field.axes:
                if id(ax) not in seen:
                    seen[id(ax)] = ax
        return tuple(seen.values())

    def _iter_index_coords(self) -> Iterator[dict[Axis, int]]:
        """Iterate over all run coordinates as ``{Axis: index}`` dicts."""
        if not self._axes:
            yield {}
            return
        ranges = [range(ax.size) for ax in self._axes]
        for combo in itertools.product(*ranges):
            yield dict(zip(self._axes, combo))


class BoundSpec:
    """A partially-applied ``EnsembleSpec`` with designated free fields.

    Obtain via ``EnsembleSpec.freeze(free=[...])``. Calling this object
    with concrete ``FieldSpec`` values (or plain values) for the free
    fields returns a fully-bound ``EnsembleSpec``.

    Args:
        base_inputs: The fixed portion of the spec (fields not in ``free``).
        free_field_names: Names of fields that must be supplied at call time.

    Examples:
        >>> bound = spec.freeze(free=["parameters"])
        >>> # Supply a Grid for parameters; all other fields come from spec:
        >>> runnable = bound(parameters=Grid(param_list, along=param_axis))
        >>> # Or supply a plain value (auto-wrapped in Fixed):
        >>> runnable = bound(parameters=some_single_param_set)
    """

    def __init__(
        self,
        base_inputs: dict[str, FieldSpec],
        free_field_names: set[str],
    ) -> None:
        self._base = base_inputs
        self._free = free_field_names

    @property
    def free_field_names(self) -> frozenset[str]:
        """Names of the fields that must be supplied when calling this spec."""
        return frozenset(self._free)

    def __call__(self, **kwargs: FieldSpec | Any) -> EnsembleSpec:
        """Bind the free fields and return a complete ``EnsembleSpec``.

        Args:
            **kwargs: One keyword argument per free field. Values that are
                already ``FieldSpec`` instances are used as-is; any other
                value is wrapped in ``Fixed``.

        Returns:
            A fully-bound ``EnsembleSpec`` ready for iteration or execution.

        Raises:
            ValueError: If any required free field is missing, or if an
                unexpected field name is provided.

        Examples:
            >>> runnable = bound(parameters=Grid(params, along=p_axis))
            >>> assert runnable.n_runs == len(params) * base_n_runs
        """
        missing = self._free - kwargs.keys()
        extra = kwargs.keys() - self._free
        if missing:
            raise ValueError(
                f"BoundSpec: missing required free field(s): {sorted(missing)}."
            )
        if extra:
            raise ValueError(
                f"BoundSpec: unexpected field(s) {sorted(extra)}. "
                f"Free fields are: {sorted(self._free)}."
            )

        new_inputs = dict(self._base)
        for name, value in kwargs.items():
            new_inputs[name] = value if isinstance(value, FieldSpec) else Fixed(value)

        return EnsembleSpec(inputs=new_inputs)

    def __repr__(self) -> str:
        return (
            f"BoundSpec(free={sorted(self._free)!r}, "
            f"fixed_fields={sorted(self._base.keys())!r})"
        )
