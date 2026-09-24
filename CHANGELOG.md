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

- **`GridEngineBackend`** — runs each `map` call as one Grid Engine array job
  (`qsub -t 1-K`), for clusters such as BU SCC. No extra dependencies.
  - Split the runs with `n_jobs=K` or `runs_per_job=R`; `slots=N` requests
    `-pe <parallel_env> N` and runs each task's chunk in `N` processes;
    `max_concurrent` maps to `-tc`; `walltime` (required) maps to `-l h_rt`.
  - Group- and site-specific settings go in `directives` (raw `qsub` options)
    and `setup` (shell lines run before each task). Options the backend sets
    itself are rejected.
  - Results are exchanged through a batch directory on a shared filesystem.
    Tasks append each run's result as it finishes, so a task killed partway
    keeps its finished runs.
  - Failures are returned in place: runs lost because their task died, went
    into `Eqw`, hit the backend `timeout`, or could not start get a
    `TaskFailedError` with `kind`, `reason`, `task_id`, `job_id` and
    `log_path`. A failed submission raises `GridEngineError`.
  - `Ctrl-C`, `SIGTERM` and `SIGHUP` delete the job with `qdel` and keep the
    batch directory.
  - Progress is logged on the `pyens.backends.gridengine` logger.
- **`RemoteError`** — returned in place of a model exception that cannot be
  sent back from a worker process intact. Carries `type_name`, `message`,
  `traceback` and picklable `attributes`.
- **`TaskFailedError`** and **`GridEngineError`**, exported from
  `pyens.backends` (and `RemoteError`, `TaskFailedError` and
  `GridEngineBackend` also from `pyens`).

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

### Bug fixes

- **`LocalBackend` no longer breaks when a model raises an exception that cannot
  be unpickled.** Previously, an exception class whose `__init__` requires
  keyword-only arguments (such as pySIPNET's `SIPNETRunError`) broke the whole
  process pool: that run and every later one came back as `BrokenProcessPool`.
  Such exceptions are now returned as a `RemoteError` carrying the original type
  name, message, traceback text and picklable attributes, and the other runs are
  unaffected. Exceptions that do unpickle keep their type, as before.

### Documentation

- New user guide page: **Running on a Grid Engine Cluster** covering
  `GridEngineBackend`: sizing tasks and walltime, environment requirements,
  failure handling, where to run the driver, and troubleshooting.
- **Running an Ensemble** introduces `GridEngineBackend`, `RemoteError` and
  `TaskFailedError`.
- New user guide page: **Reproducibility and Serialization** covering `dump_spec`,
  `load_spec`, value encoding rules, custom codecs, `PartialSpec` serialization,
  and a recommended workflow for production ensemble runs.
- Updated **Data Model** page: revised axis equality semantics section and xarray
  comparison to reflect structural equality.
