# Changelog

All user-visible API changes are documented here.

## Unreleased

### Breaking changes

- **`Axis` equality is now structural, not object identity.** Two `Axis` objects
  with the same `name`, `size`, and `labels` are now considered equal and hash to
  the same value. This means:
  - Constructing two `Axis` objects with identical arguments and using them in the
    same `EnsembleSpec` now produces **zip semantics** (they are treated as the same
    dimension), not Cartesian product. Previously this required passing the exact
    same Python object.
  - An `EnsembleSpec` now raises `ValueError` if two fields reference axes that
    share a name but have different sizes or labels. Previously this was silently
    allowed (treated as two independent dimensions with an ambiguous name).

### New features

- **`EnsembleSpec.dump(path)` / `EnsembleSpec.load(path)`** and
  **`PartialSpec.dump(path)` / `PartialSpec.load(path)`** — convenience methods
  for writing and reading specs without importing from `pyens.serialize` directly.
  `load()` is a class method that raises `TypeError` if the file contains the
  wrong spec type.

- **Serialization** (`pyens.serialize`): `EnsembleSpec` and `PartialSpec` can now
  be written to and read from JSON files.
  - `dump_spec(spec, path)` — write a spec to a JSON file with a metadata envelope
    (PyEns version, Python version, creation timestamp).
  - `load_spec(path)` — read a spec back from a file written by `dump_spec`.
  - `spec_to_dict(spec)` / `spec_from_dict(d)` — lower-level dict representation
    for embedding specs in larger metadata structures.
  - `register_codec(codec)` / `ValueCodec` — protocol for teaching PyEns how to
    encode and decode custom value types.
  - `SerializationError` — raised when a spec cannot be serialized or deserialized.
  - All serialization names are re-exported from the top-level `pyens` namespace.
  - Built-in support for `pathlib.Path` and `numpy.ndarray` values.
  - Unknown types are serialized as a tagged `repr` string for logging; strict mode
    (default) raises `SerializationError` on load; `strict=False` allows partial
    reconstruction.
  - `PartialSpec` serializes with `"FreeField"` placeholders and a `free_fields`
    list; `load_spec` returns a `PartialSpec` in this case.

### Documentation

- New user guide page: **Reproducibility and Serialization** covering `dump_spec`,
  `load_spec`, value encoding rules, custom codecs, `PartialSpec` serialization,
  and a recommended workflow for production ensemble runs.
- Updated **Data Model** page: revised axis equality semantics section and xarray
  comparison to reflect structural equality.
