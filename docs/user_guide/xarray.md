# Building Fields from xarray

PyEns's data model and xarray's line up closely (see "Connection to xarray"
on the [Data Model](data_model.md) page). An xarray dim is
an ensemble axis, and a data variable on some dims is a field that varies along
them. If your inputs already live in an xarray `Dataset`, the `pyens.xarray`
module does this translation for you, following a small set of fixed rules.

This page states those rules exactly, so that you can predict the spec you get
from any `Dataset`.

```bash
pip install pyens[xarray]
```

The core of PyEns does not depend on xarray. Only `import pyens.xarray` needs
it, and without xarray that import fails with an `ImportError` that says what
to install.

---

## The four functions

| Function | Returns | Use it for |
|---|---|---|
| `axes_of(obj)` | `dict[str, Axis]` | The `Axis` for each dim of a `Dataset` or `DataArray` |
| `field_from_dataarray(array, along=None, axes=None)` | one field | One `DataArray` |
| `fields_from_dataset(dataset, along=None, axes=None)` | `dict[str, FieldSpec]` | One field per data variable, ready for `EnsembleSpec(inputs=...)` |
| `dataset_as_field(dataset, along, axes=None)` | one `Grid` | One field whose value for each run is a sub-`Dataset` |

---

## Dims become axes

Every dim has **labels**. For a dim with a coordinate (a coordinate variable
with the dim's own name), the labels are the coordinate's values. A dim
without a coordinate has the positions `0, 1, …, n - 1` as its labels.

```python
import xarray as xr
from pyens.xarray import axes_of

table = xr.Dataset(
    {"leaf_area": (("member", "site"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
    coords={"site": ["harvard_forest", "niwot_ridge"]},
)
axes_of(table)
# {'member': Axis('member', size=3),
#  'site':   Axis('site', labels=['harvard_forest', 'niwot_ridge'])}
```

A coordinate with duplicate labels is refused, because each position along an
axis needs its own label.

### How labels are converted

Labels go through `.tolist()`, so they are plain Python values: `int`,
`float`, `str`, `bool`, or, for a coordinate of Python objects, the objects
themselves.

Dates and times are the one exception. They become **ISO 8601 strings**:

| Coordinate values | Label |
|---|---|
| `datetime64` (any resolution), `pandas.Timestamp`, `datetime.datetime`, `cftime` datetimes | `"2020-01-01T00:00:00"` |
| … with nonzero fractional seconds | `"2020-01-01T06:30:00.500000"` |
| … with a time zone | `"2020-01-01T00:00:00+00:00"` |
| `datetime.date` | `"2020-01-01"` |
| `timedelta64`, `datetime.timedelta` | `"P0DT6H0M0S"` (six hours) |

Strings are used for three reasons:

- **They are the same for every caller.** `.tolist()` on a `datetime64[ns]`
  array gives integer nanoseconds, but on a `datetime64[us]` array it gives
  `datetime` objects. Two datasets covering the same times at different
  resolutions would otherwise give unequal axes.
- **They survive serialization.** `spec.dump()` writes axis labels as JSON,
  which has no date type.
- **xarray accepts them.** `dataset.sel(time="2020-01-01T06:00:00")` works, so
  a run's coordinate selects its own time.

---

## Variables become fields

`fields_from_dataset` makes one field per data variable, along that
variable's own dims. **Every variable on a dim gets the same `Axis`**, so
variables that share a dim zip on it, and dims they do not share are crossed:

```python
from pyens import EnsembleSpec
from pyens.xarray import fields_from_dataset

table = xr.Dataset(
    {
        "leaf_area":  (("member", "site"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "soil_depth": ("site", [0.5, 0.8]),
        "timestep":   ((), 3600),
    },
    coords={"site": ["harvard_forest", "niwot_ridge"]},
)
fields = fields_from_dataset(table)
# leaf_area:  Grid along [member, site]
# soil_depth: Grid along [site]        (the same site Axis)
# timestep:   Fixed(3600)

spec = EnsembleSpec(inputs=fields)
spec.n_runs                          # 6 = 3 members × 2 sites
spec.sel(member=2, site="niwot_ridge")
# {'leaf_area': 6.0, 'soil_depth': 0.8, 'timestep': 3600}
```

The rules for what each run receives:

- **A variable whose dims are all ensemble axes** gives a plain Python value,
  one element of `variable.values.tolist()`. Nothing is cast: an `int32`
  variable gives `int`, and NaN stays NaN. Coordinates, attributes, and the
  variable's name are not part of the value.
- **A 0-d variable** becomes `Fixed` of its plain value.
- **Coordinates are not fields.** Non-dimension coordinates (such as `lon`
  on `site`) and scalar coordinates are ignored.

`field_from_dataarray` applies the same rules to a single `DataArray`.

### Zipping with fields you build yourself

`Axis` equality is structural (see "Equality semantics" on the
[Data Model](data_model.md) page). An `Axis` you build
by hand with the same name and labels is the same dimension as the one
`pyens.xarray` builds, and fields on it zip:

```python
from pyens import Axis, Grid

sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])
spec = EnsembleSpec(inputs={
    **fields_from_dataset(table),
    "climate": Grid({"niwot_ridge": "nr.nc", "harvard_forest": "hf.nc"}, along=sites),
})
spec.n_runs   # still 6: climate zips with the table on site
```

The same holds for fields built from two different datasets with the same site
labels.

---

## Keeping dims inside the value: `along`

Not every dim is an ensemble dimension. A forcing variable on `(site, time)`
should give one run per site, each receiving that site's whole time series,
not one run per site and time step. Pass `along` to name the dims that are
ensemble axes:

```python
from pyens.xarray import field_from_dataarray

forcing = xr.Dataset(
    {
        "tair":   (("site", "time"), [[280.0, 281.0, 282.0], [270.0, 271.0, 272.0]]),
        "precip": (("site", "time"), [[0.0, 1.2, 0.3], [0.4, 0.0, 0.0]]),
        "co2":    ("time", [410.0, 411.0, 412.0]),
    },
    coords={"site": ["harvard_forest", "niwot_ridge"]},
)

tair = field_from_dataarray(forcing["tair"], along="site")
tair.axes   # (Axis('site', labels=['harvard_forest', 'niwot_ridge']),)
```

When some of a variable's dims are left over, **each run's value is a
`DataArray` slice**, `variable.isel(site=i)`. It keeps the other dims, their
coordinates, and its attributes. It also keeps a scalar coordinate recording
the label it was selected at, so `value.site` tells the model which site it
has. When `along` lists every dim of the variable, values are plain Python
values, as above.

The field's axes are listed in the order you give in `along`, or in the
variable's dim order without `along`. The order matters only for the order of
runs: `iter_runs()` crosses axes in the order the spec's fields introduce them.

`fields_from_dataset(dataset, along=[...])` puts each variable along the
`along` dims it has. A variable with none of them, like `co2` above, is the
same for every run and becomes `Fixed` of the whole `DataArray`.

### A whole `Dataset` as one field

A model often takes all of its forcing as one object. `dataset_as_field` gives
one field whose value for each run is the sub-`Dataset`
`dataset.isel(site=i)`, holding every data variable:

```python
from pyens.xarray import dataset_as_field

climate = dataset_as_field(forcing, along="site")
spec = EnsembleSpec(inputs={
    **fields_from_dataset(table),
    "climate": climate,
})
spec.n_runs   # 6
# For each run, inputs["climate"] is a Dataset with tair, precip and co2 over time.
```

Variables that do not vary along `along` (here `co2`) appear unchanged in every
run's value. `along` is required.

---

## Using an `Axis` you already have: `axes`

Pass `axes={"site": sites}` to make a field use an `Axis` you built
elsewhere, for example one shared with other fields. **The `Axis` is matched
to the dim by label, not by position.** The data is reordered to the `Axis`'s
label order, so an `Axis` listing the sites in a different order than the
coordinate still gets each site's own values:

```python
sites = Axis("site", labels=["niwot_ridge", "harvard_forest"])   # reversed
fields = fields_from_dataset(table, axes={"site": sites})
spec = EnsembleSpec(inputs=fields)
spec.sel(member=0, site="niwot_ridge")["leaf_area"]   # 2.0, the niwot_ridge value
```

The rules:

- The `Axis` must have the dim's size and exactly the dim's labels. Otherwise
  the conversion is refused with a message naming the labels that do not
  match.
- For a dim without a coordinate, the labels are the positions, so pass
  `Axis(dim, size=n)`. An `Axis` with string labels is refused, because
  nothing ties a string to a position. Give the dim a coordinate first
  (`dataset.assign_coords(member=[...])`).
- The `Axis` may have a different name from the dim. Its name is the key used
  in run coordinates.
- Entries for dims the object does not have are ignored.

---

## Things to know

**Datetime values are not labels.** The ISO conversion applies to coordinate
labels only. A `datetime64[ns]` *data variable* on ensemble dims becomes
integer nanoseconds under `.tolist()`. Convert it first if your model needs
dates.

**Float labels must match exactly.** Two coordinates that differ in the last
bit, such as latitudes computed in two ways, give unequal axes, and
`EnsembleSpec` reports two conflicting axes with the same name. `1` and `1.0`
compare equal. Avoid NaN labels, since NaN is not equal to itself.

**Slices share memory with your data.** `DataArray` and `Dataset` values are
views into the original arrays, not copies. Under `SequentialBackend`, a model
that modifies its input in place modifies your data.

**Sending slices to workers.** `LocalBackend` and `GridEngineBackend` pickle
each run's inputs. A slice pickles only its own data, plus the coordinates it
keeps. A full `time` coordinate therefore travels with every run.
Variables that do not vary along `along` travel with every run in
`dataset_as_field`.

**Lazily loaded data.** A slice of a file opened with `xarray.open_dataset`,
or of a dask array, stays lazy. It pickles as a reference to the file or a
task graph, and each worker reads its own slice. That works on a cluster only
if the file exists at the same path on the worker nodes. To send the data
itself instead, call `.load()` before converting. Plain-value fields always
read the whole variable into memory.

**Serialization.** A spec of plain values round-trips through `spec.dump()`
and `EnsembleSpec.load()`. A spec with `DataArray` or `Dataset` values can be
dumped for the record, but `load()` refuses to rebuild it (see
[Reproducibility and Serialization](reproducibility.md)).

---

## Not yet: results back to xarray

This module covers inputs only. The other direction, from an `EnsembleResult`
to a `Dataset` with the ensemble axes as dims, will live in `pyens.xarray` when
the result layer is built.
