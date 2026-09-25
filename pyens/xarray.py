"""Build PyEns fields from xarray objects.

An xarray ``Dataset`` of model inputs already has the shape of an ensemble:
its dims are the ensemble's axes, and each data variable is a field that
varies along some of them. This module converts one into the other so that
you do not write the translation by hand::

    from pyens import EnsembleSpec
    from pyens.xarray import fields_from_dataset

    spec = EnsembleSpec(inputs=fields_from_dataset(table))

It requires xarray (``pip install pyens[xarray]``). The core of PyEns does not
import it.

Functions
---------
- :func:`axes_of` — one ``Axis`` per dim of a ``Dataset`` or ``DataArray``.
- :func:`field_from_dataarray` — one field from one ``DataArray``.
- :func:`fields_from_dataset` — one field per data variable of a ``Dataset``.
- :func:`dataset_as_field` — one field whose value for each run is a
  sub-``Dataset`` (for example, all the forcing variables at one site).

How the conversion works
------------------------
The rules below are the whole of the conversion. Nothing else is inferred,
cast, or dropped.

**1. Every dim has labels.** A dim's labels are the values of its coordinate
(the coordinate variable with the dim's own name), converted as in rule 2. A
dim without a coordinate has the positions ``0, 1, ..., n - 1`` as its labels.
Its ``Axis`` is ``Axis(dim, labels=[...])`` when it has a coordinate and
``Axis(dim, size=n)`` when it does not. A coordinate with duplicate labels is
refused, because an ``Axis`` needs one label per position.

**2. Coordinate labels become plain Python values.** Labels go through
``.tolist()``, which yields ``int``, ``float``, ``str``, ``bool``, or, for an
object coordinate, the objects themselves. Date and time labels are the one
exception. They become ISO 8601 strings, so that they compare equal whatever
the array's time resolution and survive :meth:`~pyens.EnsembleSpec.dump`:

- ``datetime64`` of any resolution, ``pandas.Timestamp``,
  ``datetime.datetime``, and ``cftime`` datetimes (non-standard calendars)
  become ``"2020-01-01T00:00:00"``, with fractional seconds only when they
  are nonzero (``"2020-01-01T06:30:00.500000"``) and an offset only when the
  value has a time zone (``"2020-01-01T00:00:00+00:00"``).
- ``datetime.date`` becomes ``"2020-01-01"``.
- ``timedelta64`` and ``datetime.timedelta`` become ISO 8601 durations,
  such as ``"P0DT6H0M0S"`` for six hours.

The string is the value's ``isoformat()``, taken through
``pandas.Timestamp`` for ``datetime64`` and through ``pandas.Timedelta`` for
both kinds of duration.
xarray accepts these strings in ``.sel``, so
``dataset.sel(time=record.coordinate["time"])`` selects the run's time.

**3. ``along`` chooses which dims are ensemble axes.** By default every dim
of a variable is an axis. Pass ``along=["site"]`` to make only ``site`` an
axis. The other dims stay inside each run's value. The field's axes are in
the order of ``along``, or in the variable's dim order when ``along`` is not
given. The order matters only for the order of runs:
:meth:`~pyens.EnsembleSpec.iter_runs` crosses axes in the order the spec's
fields introduce them.

**4. A run's value depends on which dims are left over.**

- *No dims left over* (every dim of the variable is an axis): the value is a
  plain Python value, one element of ``variable.values.tolist()``. The field
  is ``Grid(variable.values.tolist(), along=[...])``. Nothing is cast, and
  NaN stays NaN. Coordinates, attributes, and the variable's name are not
  part of the value.
- *Some dims left over*: the value is the ``DataArray`` slice
  ``variable.isel(site=i, ...)``. It keeps its remaining dims, their
  coordinates, its attributes, and a scalar coordinate recording the label it
  was selected at (so ``value.site`` tells a model which site it has).
- :func:`dataset_as_field` always gives a sub-``Dataset``,
  ``dataset.isel(site=i, ...)``, which holds every data variable.

**5. A variable on no ensemble axis becomes** ``Fixed``. A 0-d variable
becomes ``Fixed`` of its plain value (``variable.values.tolist()``). In
:func:`fields_from_dataset` with ``along``, a variable on none of the
``along`` dims becomes ``Fixed`` of the whole ``DataArray``.

**6. Shared dims zip, other dims cross.** :func:`fields_from_dataset` builds
one ``Axis`` per dim and gives it to every variable on that dim. A variable on
``(member, site)`` and one on ``(site,)`` share the site axis. PyEns therefore
zips them on ``site`` and crosses them with ``member``, the same way xarray
aligns them. ``Axis`` equality is structural, so an ``Axis`` you build
yourself with the same name and labels, such as
``Axis("site", labels=[...])``, is the same dimension and zips too.

**7. ``axes=`` matches by label, not by position.** Pass
``axes={"site": site_axis}`` to use an ``Axis`` you already have. It is
matched to the dim by label, and the data is reordered to the ``Axis``'s
label order, so an ``Axis`` listing the sites in a different order than the
coordinate still gets each site's own values. The ``Axis`` must have the
dim's size and exactly the dim's labels (for a dim without a coordinate,
those are the positions ``0, ..., n - 1``, which ``Axis(dim, size=n)`` has).
Otherwise the conversion is refused. The ``Axis``'s name need not be the dim's
name. Its name is the key of the run coordinate. Entries for dims the object
does not have are ignored.

Caveats
-------
- **Datetime values are not labels.** Rule 2 applies to coordinate labels
  only. A ``datetime64[ns]`` *data variable* on ensemble dims becomes integer
  nanoseconds under ``.tolist()``. Convert it first if your model wants
  datetimes.
- **Float labels compare exactly.** Two coordinates that differ in the last
  bit, such as latitudes computed in different ways, give unequal axes.
  ``1`` and ``1.0`` compare equal. NaN labels do not compare equal to each
  other, so avoid them.
- **Slices share memory with your array.** The ``DataArray`` and ``Dataset``
  values of rule 4 are views, not copies. A model that modifies its input in
  place under :class:`~pyens.SequentialBackend` modifies your array.
- **Sending slices to workers.** Process and cluster backends pickle each
  run's value. A slice pickles only its own data, plus the coordinates it
  keeps (a full ``time`` coordinate travels with every run). A slice of a
  file opened lazily with ``xarray.open_dataset``, or of a dask array,
  pickles as a reference to the file or a task graph. Each worker then reads
  its slice itself, so the file must exist at the same path on the worker.
  Call ``.load()`` on the object before converting to send the data instead.
  In :func:`dataset_as_field`, variables that do not vary along ``along``
  travel with every run.
- **Serialization.** A spec whose values are plain Python values round-trips
  through :meth:`~pyens.EnsembleSpec.dump` and
  :meth:`~pyens.EnsembleSpec.load`. A spec whose values are ``DataArray`` or
  ``Dataset`` slices can be dumped for the record, but ``load`` refuses it
  (see :mod:`pyens.serialize`).
- **Laziness.** Rule 4's first case reads the whole variable into memory
  (``.values``). Slices do not.

The other direction, from an :class:`~pyens.EnsembleResult` to a ``Dataset``
with the axes as dims, belongs in this module when the result layer lands.

Examples:
    A table of parameters over ensemble members and sites, and one input on
    sites only::

        >>> import xarray as xr
        >>> from pyens import EnsembleSpec
        >>> from pyens.xarray import fields_from_dataset
        >>> table = xr.Dataset(
        ...     {
        ...         "leaf_area": (("member", "site"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ...         "soil_depth": ("site", [0.5, 0.8]),
        ...     },
        ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
        ... )
        >>> fields = fields_from_dataset(table)
        >>> fields["leaf_area"].axes
        (Axis('member', size=3), Axis('site', labels=['harvard_forest', 'niwot_ridge']))
        >>> spec = EnsembleSpec(inputs=fields)
        >>> spec.n_runs
        6
        >>> spec.sel(member=2, site="niwot_ridge")
        {'leaf_area': 6.0, 'soil_depth': 0.8}
"""

from __future__ import annotations

import datetime
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

try:
    import pandas as pd
    import xarray as xr
except ImportError as exc:
    raise ImportError(
        "pyens.xarray requires xarray. Install it with `pip install pyens[xarray]`."
    ) from exc

from pyens.axis import Axis
from pyens.fields import FieldSpec, Fixed, Grid

try:
    import cftime

    _CFTIME_TYPES: tuple[type, ...] = (cftime.datetime,)
except ImportError:
    _CFTIME_TYPES = ()

__all__ = [
    "axes_of",
    "dataset_as_field",
    "field_from_dataarray",
    "fields_from_dataset",
]

_PREVIEW = 6


def axes_of(obj: xr.Dataset | xr.DataArray) -> dict[str, Axis]:
    """Return one ``Axis`` per dim of a ``Dataset`` or ``DataArray``.

    Each ``Axis`` is named after its dim. Its labels are the dim's coordinate
    values converted to plain Python values, with dates and times as ISO 8601
    strings. A dim without a coordinate gets ``Axis(dim, size=n)``, whose
    labels are the positions ``0, ..., n - 1``. See the module docstring for
    the full conversion rules.

    Two objects with the same dim names and coordinate labels give equal
    axes, so axes built from one ``Dataset`` zip with fields built from
    another.

    Args:
        obj: The ``Dataset`` or ``DataArray``.

    Returns:
        A dict from dim name to ``Axis``, in the object's dim order.

    Raises:
        ValueError: If a coordinate has duplicate labels.
        TypeError: If a coordinate's labels are not hashable.

    Examples:
        >>> import xarray as xr
        >>> from pyens.xarray import axes_of
        >>> table = xr.Dataset(
        ...     {"leaf_area": (("member", "site"), [[1.0, 2.0], [3.0, 4.0]])},
        ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
        ... )
        >>> axes_of(table)
        {'member': Axis('member', size=2), 'site': Axis('site', labels=['harvard_forest', 'niwot_ridge'])}
    """
    where = _describe(obj)
    return {dim: _axis_for(obj, dim, None, where)[0] for dim in obj.sizes}


def field_from_dataarray(
    array: xr.DataArray,
    *,
    along: str | Sequence[str] | None = None,
    axes: Mapping[str, Axis] | None = None,
) -> FieldSpec:
    """Return the field that a ``DataArray`` describes.

    With ``along`` omitted, every dim of ``array`` is an ensemble axis and
    each run's value is a plain Python value, one element of
    ``array.values.tolist()``. With ``along`` given, only those dims are
    axes and each run's value is the ``DataArray`` slice
    ``array.isel({dim: i, ...})``, keeping the other dims. A 0-d array
    becomes ``Fixed`` of its plain value. See the module docstring for the
    full conversion rules.

    Args:
        array: The ``DataArray``.
        along: The dims to make ensemble axes, in the order the field should
            list them. A single dim may be given as a string. Defaults to all
            of ``array``'s dims, in ``array``'s order.
        axes: ``Axis`` objects to use for some dims, keyed by dim name. Each
            is matched to its dim by label, and the values are reordered to
            the ``Axis``'s label order. Dims not in ``axes`` get the ``Axis``
            that :func:`axes_of` would give them.

    Returns:
        A ``Grid`` along the axes of the ``along`` dims, or ``Fixed`` for a
        0-d array.

    Raises:
        ValueError: If ``along`` is empty, repeats a dim, or names a dim
            ``array`` does not have; if a coordinate has duplicate labels; or
            if an ``Axis`` in ``axes`` does not have its dim's size and
            labels.
        TypeError: If a coordinate's labels are not hashable.

    Examples:
        Every dim an axis, plain values::

            >>> import xarray as xr
            >>> from pyens.xarray import field_from_dataarray
            >>> leaf_area = xr.DataArray(
            ...     [[1.0, 2.0], [3.0, 4.0]],
            ...     dims=("member", "site"),
            ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
            ... )
            >>> grid = field_from_dataarray(leaf_area)
            >>> grid.axes
            (Axis('member', size=2), Axis('site', labels=['harvard_forest', 'niwot_ridge']))

        Only ``site`` an axis; each run gets that site's time series::

            >>> tair = xr.DataArray(
            ...     [[280.0, 281.0, 282.0], [270.0, 271.0, 272.0]],
            ...     dims=("site", "time"),
            ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
            ... )
            >>> grid = field_from_dataarray(tair, along="site")
            >>> sites = grid.axes[0]
            >>> grid.value_at({sites: 1}).values.tolist()
            [270.0, 271.0, 272.0]
    """
    where = _describe(array)
    if along is None:
        dims = [str(d) for d in array.dims]
    else:
        dims = _check_along(along, array.dims, where)
    resolved: dict[str, tuple[Axis, list[int] | None]] = {}
    for dim in dims:
        resolved[dim] = _axis_for(array, dim, axes, where)
    return _field(_reorder(array, dims, resolved), dims, resolved)


def fields_from_dataset(
    dataset: xr.Dataset,
    *,
    along: str | Sequence[str] | None = None,
    axes: Mapping[str, Axis] | None = None,
) -> dict[str, FieldSpec]:
    """Return one field per data variable of a ``Dataset``.

    Each variable becomes the field :func:`field_from_dataarray` would give
    it, along its own dims only. Every variable on a dim gets the same
    ``Axis``, so a variable on ``(member, site)`` and one on ``(site,)`` zip
    on ``site`` and cross on ``member`` once they are in an
    ``EnsembleSpec``. With ``along`` given, each variable is along the
    ``along`` dims it has, and a variable with none of them becomes ``Fixed``
    of the whole ``DataArray``. Coordinates are not fields. See the module
    docstring for the full conversion rules.

    Args:
        dataset: The ``Dataset``.
        along: The dims to make ensemble axes, in the order each field should
            list them. A single dim may be given as a string. Defaults to all
            of each variable's dims, in the variable's order.
        axes: ``Axis`` objects to use for some dims, keyed by dim name,
            matched by label as in :func:`field_from_dataarray`.

    Returns:
        A dict from variable name to field, in the dataset's variable order,
        ready for ``EnsembleSpec(inputs=...)``.

    Raises:
        ValueError: If ``along`` is empty, repeats a dim, or names a dim
            ``dataset`` does not have; if a coordinate has duplicate labels;
            or if an ``Axis`` in ``axes`` does not have its dim's size and
            labels. The message names the first variable on the dim.
        TypeError: If a coordinate's labels are not hashable.

    Examples:
        >>> import xarray as xr
        >>> from pyens import EnsembleSpec
        >>> from pyens.xarray import fields_from_dataset
        >>> table = xr.Dataset(
        ...     {
        ...         "leaf_area": (("member", "site"), [[1.0, 2.0], [3.0, 4.0]]),
        ...         "soil_depth": ("site", [0.5, 0.8]),
        ...         "timestep": ((), 3600),
        ...     },
        ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
        ... )
        >>> fields = fields_from_dataset(table)
        >>> fields["timestep"]
        Fixed(3600)
        >>> EnsembleSpec(inputs=fields).n_runs
        4
    """
    if along is None:
        wanted: list[str] | None = None
    else:
        wanted = _check_along(along, dataset.sizes, _describe(dataset))
    resolved: dict[str, tuple[Axis, list[int] | None]] = {}
    fields: dict[str, FieldSpec] = {}
    for name, variable in dataset.data_vars.items():
        var_dims = [str(d) for d in variable.dims]
        if wanted is None:
            dims = var_dims
        else:
            dims = [d for d in wanted if d in var_dims]
        for dim in dims:
            if dim not in resolved:
                resolved[dim] = _axis_for(dataset, dim, axes, f"variable {name!r}")
        fields[str(name)] = _field(_reorder(variable, dims, resolved), dims, resolved)
    return fields


def dataset_as_field(
    dataset: xr.Dataset,
    *,
    along: str | Sequence[str],
    axes: Mapping[str, Axis] | None = None,
) -> Grid:
    """Return one field whose value for each run is a sub-``Dataset``.

    The value at each run is ``dataset.isel({dim: i, ...})`` over the
    ``along`` dims. It holds every data variable, keeps the other dims and
    their coordinates, and records the selected labels as scalar
    coordinates. Use this when the model takes one object holding several
    variables, such as all the forcing variables at one site. Variables not
    on any ``along`` dim appear, unchanged, in every run's value. See the
    module docstring for the full conversion rules.

    Args:
        dataset: The ``Dataset``.
        along: The dims to make ensemble axes, in the order the field should
            list them. A single dim may be given as a string. Required.
        axes: ``Axis`` objects to use for some dims, keyed by dim name,
            matched by label as in :func:`field_from_dataarray`.

    Returns:
        A ``Grid`` along the axes of the ``along`` dims whose values are
        ``Dataset`` objects.

    Raises:
        ValueError: If ``along`` is empty, repeats a dim, or names a dim
            ``dataset`` does not have; if a coordinate has duplicate labels;
            or if an ``Axis`` in ``axes`` does not have its dim's size and
            labels.
        TypeError: If a coordinate's labels are not hashable.

    Examples:
        >>> import xarray as xr
        >>> from pyens.xarray import dataset_as_field
        >>> forcing = xr.Dataset(
        ...     {
        ...         "tair": (("site", "time"), [[280.0, 281.0], [270.0, 271.0]]),
        ...         "precip": (("site", "time"), [[0.0, 1.2], [0.4, 0.0]]),
        ...     },
        ...     coords={"site": ["harvard_forest", "niwot_ridge"]},
        ... )
        >>> climate = dataset_as_field(forcing, along="site")
        >>> sites = climate.axes[0]
        >>> climate.value_at({sites: 1}).precip.values.tolist()
        [0.4, 0.0]
    """
    where = _describe(dataset)
    dims = _check_along(along, dataset.sizes, where)
    resolved: dict[str, tuple[Axis, list[int] | None]] = {}
    for dim in dims:
        resolved[dim] = _axis_for(dataset, dim, axes, where)
    ordered = _reorder(dataset, dims, resolved)
    return Grid(_slices(ordered, dims), along=[resolved[d][0] for d in dims])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe(obj: xr.Dataset | xr.DataArray) -> str:
    if isinstance(obj, xr.DataArray):
        return "DataArray" if obj.name is None else f"DataArray {obj.name!r}"
    return "Dataset"


def _check_along(
    along: str | Sequence[str], available: Any, where: str
) -> list[str]:
    dims = [along] if isinstance(along, str) else list(along)
    if not dims:
        raise ValueError(f"{where}: 'along' must name at least one dim.")
    repeated = sorted({d for d in dims if dims.count(d) > 1})
    if repeated:
        raise ValueError(f"{where}: 'along' repeats the dim(s) {repeated}.")
    known = [str(d) for d in available]
    unknown = [d for d in dims if d not in known]
    if unknown:
        raise ValueError(
            f"{where}: 'along' names dim(s) {unknown} that it does not have. "
            f"Its dims are {known}."
        )
    return dims


def _label(value: Any) -> Hashable:
    if isinstance(value, (datetime.date, *_CFTIME_TYPES)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return pd.Timedelta(value).isoformat()
    return value


def _dim_labels(obj: xr.Dataset | xr.DataArray, dim: str) -> list[Hashable] | None:
    """Return a dim's converted coordinate labels, or ``None`` without a coordinate."""
    if dim not in obj.coords:
        return None
    values = obj[dim].values
    # ``.tolist()`` turns datetime64[ns] and timedelta64[ns] into integers, so
    # these go through pandas, which gives Timestamps and Timedeltas.
    if values.dtype.kind in "mM":
        items = pd.Index(values).tolist()
    else:
        items = values.tolist()
    return [_label(v) for v in items]


def _preview(labels: Sequence[Hashable]) -> str:
    shown = ", ".join(repr(label) for label in labels[:_PREVIEW])
    more = f", ... ({len(labels)} in all)" if len(labels) > _PREVIEW else ""
    return f"[{shown}{more}]"


def _axis_for(
    obj: xr.Dataset | xr.DataArray,
    dim: str,
    supplied: Mapping[str, Axis] | None,
    where: str,
) -> tuple[Axis, list[int] | None]:
    """Return the ``Axis`` for ``dim`` and the positions that reorder ``obj`` to it.

    The positions are ``None`` when the dim's labels are already in the
    ``Axis``'s order.
    """
    labels = _dim_labels(obj, dim)
    size = obj.sizes[dim]

    if supplied is None or dim not in supplied:
        try:
            if labels is None:
                return Axis(str(dim), size=size), None
            return Axis(str(dim), labels=labels), None
        except (ValueError, TypeError) as exc:
            detail = _duplicates(labels)
            raise type(exc)(
                f"{where}: cannot make an Axis for dim {dim!r}{detail}: {exc}"
            ) from exc

    axis = supplied[dim]
    if axis.size != size:
        raise ValueError(
            f"{where}: the Axis {axis!r} given for dim {dim!r} has size "
            f"{axis.size}, but the dim has size {size}."
        )
    dim_labels = labels if labels is not None else list(range(size))
    axis_labels = list(axis.labels)
    if axis_labels == dim_labels:
        return axis, None

    try:
        position = {label: i for i, label in enumerate(dim_labels)}
    except TypeError as exc:
        raise TypeError(
            f"{where}: the labels of dim {dim!r} are not hashable: {exc}"
        ) from exc
    if len(position) != size:
        raise ValueError(
            f"{where}: dim {dim!r}{_duplicates(dim_labels)}, so its values cannot "
            f"be matched to the labels of {axis!r}."
        )
    missing = [label for label in axis_labels if label not in position]
    if missing:
        source = (
            "its coordinate" if labels is not None
            else "its positions, because it has no coordinate"
        )
        hint = (
            "" if labels is not None
            else f" Give the dim a coordinate, or pass Axis({axis.name!r}, size={size})."
        )
        raise ValueError(
            f"{where}: the Axis given for dim {dim!r} has labels "
            f"{_preview(missing)} that the dim does not. The dim's labels, from "
            f"{source}, are {_preview(dim_labels)}.{hint}"
        )
    return axis, [position[label] for label in axis_labels]


def _duplicates(labels: Sequence[Hashable] | None) -> str:
    if labels is None:
        return ""
    try:
        seen: set[Hashable] = set()
        repeated: list[Hashable] = []
        for label in labels:
            if label in seen and label not in repeated:
                repeated.append(label)
            seen.add(label)
    except TypeError:
        return ""
    return f" (duplicate labels {_preview(repeated)})" if repeated else ""


def _reorder(
    obj: Any,
    dims: Sequence[str],
    resolved: Mapping[str, tuple[Axis, list[int] | None]],
) -> Any:
    indexers = {d: resolved[d][1] for d in dims if resolved[d][1] is not None}
    return obj.isel(indexers) if indexers else obj


def _field(
    array: xr.DataArray,
    dims: Sequence[str],
    resolved: Mapping[str, tuple[Axis, list[int] | None]],
) -> FieldSpec:
    if not dims:
        return Fixed(array.values.tolist() if array.ndim == 0 else array)
    along = [resolved[d][0] for d in dims]
    if len(dims) == array.ndim:
        return Grid(array.transpose(*dims).values.tolist(), along=along)
    return Grid(_slices(array, dims), along=along)


def _slices(obj: Any, dims: Sequence[str]) -> list[Any]:
    """Return ``obj.isel`` at every position of ``dims``, nested one list per dim."""

    def level(prefix: tuple[int, ...]) -> Any:
        if len(prefix) == len(dims):
            return obj.isel(dict(zip(dims, prefix)))
        return [level((*prefix, i)) for i in range(obj.sizes[dims[len(prefix)]])]

    return level(())
