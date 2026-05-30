# Reproducibility and Serialization

A common requirement in scientific computing is to be able to reproduce a
run exactly — to record precisely which inputs were used, so that results can
be verified, audited, or re-generated at a later date. This page explains how
to serialize an `EnsembleSpec` or `PartialSpec` to a JSON file and reload it,
and offers guidance on building a reproducible workflow around PyEns.

---

## Writing and reading a spec file

Call `dump()` on any spec to write it to disk, and `EnsembleSpec.load()` to
read it back:

```python
from pyens import Axis, EnsembleSpec, Fixed, Grid

sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])

spec = EnsembleSpec(inputs={
    "parameters": Fixed({"aMax": 12.0, "k": 0.5, "baseR": 2.0}),
    "climate":    Grid(["hf_forcing.nc", "nr_forcing.nc"], along=sites),
})

# Write to disk
spec.dump("runs/ensemble_spec.json")

# Read back — returns a fully functional EnsembleSpec
recovered = EnsembleSpec.load("runs/ensemble_spec.json")
assert recovered.n_runs == spec.n_runs
```

The file is human-readable JSON with a metadata envelope that records the PyEns
version, Python version, and timestamp:

```json
{
  "__pyens_spec__": true,
  "__spec_type__": "EnsembleSpec",
  "__pyens_version__": "0.1.0",
  "__created_at__": "2026-05-29T14:30:00+00:00",
  "__python_version__": "3.11.9 ...",
  "axes": [
    {"name": "site", "labels": ["harvard_forest", "niwot_ridge"]}
  ],
  "fields": {
    "parameters": {
      "field_type": "Fixed",
      "value": {"aMax": 12.0, "k": 0.5, "baseR": 2.0}
    },
    "climate": {
      "field_type": "Grid",
      "along": ["site"],
      "values": ["hf_forcing.nc", "nr_forcing.nc"]
    }
  }
}
```

The file is suitable for committing to a version control system alongside the
code and results that produced it.

:::{note}
Module-level functions `dump_spec(spec, path)` and `load_spec(path)` are also
available from `pyens.serialize` for situations where you prefer importing
functions directly — for example, when integrating with other tooling or writing
generic utilities that accept any spec as an argument. The methods and functions
are equivalent.
:::

---

## Value encoding

PyEns encodes values according to their Python type:

| Value type | Encoding |
|---|---|
| `None`, `bool`, `int`, `float`, `str` | Stored directly as JSON literals |
| `list`, `tuple` | Stored as a JSON array; elements encoded recursively |
| `dict` (string keys, JSON-native values) | Stored as a JSON object |
| `pathlib.Path` | Tagged object: `{"__type__": "path", "value": "/the/path"}` |
| `numpy.ndarray` | Tagged object with dtype, shape, and flattened data |
| Any other type | Tagged as `"unknown"` (see below) |

**Tagged objects** use a `{"__type__": "<tag>", ...}` wrapper in the JSON.
They are decoded back to their original Python type when the spec is loaded.

### File paths

`pathlib.Path` objects round-trip exactly:

```python
from pathlib import Path

spec = EnsembleSpec(inputs={
    "climate": Grid(
        [Path("/data/hf_forcing.nc"), Path("/data/nr_forcing.nc")],
        along=sites,
    ),
})
spec.dump("spec.json")
recovered = EnsembleSpec.load("spec.json")
# Values are Path objects, not strings
```

Storing paths rather than loaded data is recommended for large files: the spec
records *what* was used without embedding the data itself.

### NumPy arrays

If NumPy is installed, `numpy.ndarray` values are encoded as dtype, shape, and
a flattened data list:

```json
{
  "__type__": "ndarray",
  "dtype": "float64",
  "shape": [3, 4],
  "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
}
```

For very large arrays, embedding data directly in the spec file is not
recommended. Store the array to a `.npy` or NetCDF file and reference its path
in the spec instead.

---

## Opaque objects and strict mode

Not every Python object can be serialized as human-readable text. When PyEns
encounters a type it does not recognize, it stores a `repr()` string and the
class name:

```json
{
  "__type__": "unknown",
  "class": "mypackage.models.SIPNETModel",
  "repr": "SIPNETModel(version='2.1', ...)"
}
```

This is enough information to identify what was used, even if the exact object
cannot be reconstructed.

**By default, `load()` raises `SerializationError` for unknown-tagged values**,
because loading a spec with unresolvable values produces an incomplete spec that
would silently produce wrong inputs. Pass `strict=False` to allow loading anyway
— unknown values are replaced by their repr string:

```python
# For logging / auditing only — not for re-running:
spec = EnsembleSpec.load("spec.json", strict=False)
```

To enable full round-tripping for custom types, register a
:class:`~pyens.serialize.ValueCodec`.

---

## Custom codecs

Register a codec to teach PyEns how to serialize and reconstruct your own
types. A codec is any object implementing three methods:

```python
from pyens.serialize import register_codec

class ParameterSetCodec:
    type_tag = "parameter_set"    # unique tag, appears in the JSON

    def can_encode(self, value):
        return isinstance(value, ParameterSet)

    def encode(self, value):
        # Return a JSON-serialisable dict (do not include "__type__")
        return {"values": value.to_dict(), "version": value.schema_version}

    def decode(self, data):
        return ParameterSet.from_dict(data["values"], version=data["version"])

register_codec(ParameterSetCodec())
```

After registration, `ParameterSet` values in `Fixed` and `Grid` fields will
round-trip through `dump()` and `load()` exactly.

Register codecs before calling `dump()` or `load()`. A good place to register
them is at module import time in the same file that defines your model or
parameter types.

---

## Serializing a PartialSpec

`PartialSpec` objects (returned by `EnsembleSpec.freeze()`) can also be
serialized. Free fields are recorded with a `"FreeField"` placeholder:

```python
base_spec = EnsembleSpec(inputs={
    "parameters": Fixed(default_theta),
    "climate":    Grid(forcing_list, along=sites),
})

param_map = base_spec.freeze(free=["parameters"])
param_map.dump("param_map.json")

# Later — reload and use:
recovered = PartialSpec.load("param_map.json")
p_axis = Axis("param_set", size=10)
runnable = recovered(parameters=Grid(proposed_params, along=p_axis))
```

The spec file records which fields are free, giving a complete picture of the
ensemble structure even before the free fields are filled in.

---

## Recommended workflow for a production run

A reproducible production ensemble typically looks like this:

**1. Define and save the spec**

```python
spec = EnsembleSpec(inputs={...})
spec.dump("runs/2026-05-29_baseline/spec.json")
```

**2. Run the ensemble**

```python
runner = EnsembleRunner(my_model, backend=LocalBackend(n_workers=8))
result = runner.run(spec)
result.save("runs/2026-05-29_baseline/results.nc")
```

**3. Commit the spec file to version control**

The `spec.json` file is small, human-readable, and records the complete input
structure. Commit it alongside the code version used to produce the results:

```
git add runs/2026-05-29_baseline/spec.json
git commit -m "Save spec for baseline ensemble (paper Fig. 3)"
```

**4. Verify later**

```python
spec = EnsembleSpec.load("runs/2026-05-29_baseline/spec.json")
print(spec.describe())
```

This workflow gives you a complete, version-controlled record of every
production run: which inputs, which axis structure, which PyEns version, and
when it was run.

---

## Working with the dict representation

`spec_to_dict` and `spec_from_dict` are lower-level functions that give you
the intermediate Python dict rather than writing to disk. They are useful for
integrating with other serialization systems (e.g., YAML, TOML, or a database)
or for embedding a spec in a larger metadata structure:

```python
from pyens import spec_to_dict, spec_from_dict

d = spec_to_dict(spec)

# Embed in a larger experiment record
record = {
    "experiment_id": "baseline_2026",
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "spec": d,
}

with open("experiment_record.json", "w") as f:
    json.dump(record, f, indent=2)

# Extract and reconstruct later
with open("experiment_record.json") as f:
    record = json.load(f)
spec = spec_from_dict(record["spec"])
```
