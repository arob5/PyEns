# Running an Ensemble

The [Data Model](data_model.md) page explains how to describe an ensemble —
axes, fields, and specs. This page explains how to execute one: how to drive a
model callable over all the runs in a spec, collect the outputs, and work with
the results.

---

## From specification to model calls

Once you have an `EnsembleSpec`, you can iterate it directly:

```python
for inputs, coord in spec.iter_runs():
    output = my_model(**inputs)
```

For small exploratory runs this is often sufficient. But as ensembles grow
there are three concerns this approach leaves unaddressed.

**Result association.** Each `output` must be manually paired with its
`coord`. It is easy to fall out of sync — especially when some runs are
skipped, retried, or processed out of order.

**Failure isolation.** If `my_model` raises an exception for one run, the
loop stops. The outputs from the runs that completed before the failure are
useful, but there is no structure to hold them or distinguish them from the
one that failed.

**Backend flexibility.** Switching from a serial loop to parallel execution
requires rewriting the loop. Switching again for an HPC cluster requires
rewriting it once more.

PyEns addresses all three with a small execution layer built on three
abstractions: a **backend**, an **`EnsembleRunner`**, and an
**`EnsembleResult`**.

---

## The three abstractions

A **backend** encapsulates *how* each model call is executed — in a plain
loop, across multiple CPU cores, or on a remote cluster. The rest of the
code never sees this detail.

An **`EnsembleRunner`** is the bridge between a spec and a backend. You give
it a model callable and a backend; calling `run(spec)` hands the runs to the
backend and collects the results.

An **`EnsembleResult`** is the structured collection of outcomes. Every
output is paired with the coordinate that produced it. Failures are stored
rather than discarded, so a single bad run does not prevent you from
recovering the rest.

Together they turn this:

```python
results = []
for inputs, coord in spec.iter_runs():
    output = my_model(**inputs)
    results.append((output, coord))
```

into this:

```python
from pyens.runner import EnsembleRunner
from pyens.backends import SequentialBackend

runner = EnsembleRunner(my_model, SequentialBackend())
result = runner.run(spec)
```

---

## EnsembleRunner

`EnsembleRunner` takes two arguments: the model callable and a backend.

```python
runner = EnsembleRunner(my_model, SequentialBackend())
```

The model must accept the field names from the spec as keyword arguments. If
the spec has fields `"parameters"`, `"forcing"`, and `"ic"`, PyEns calls

```python
my_model(parameters=..., forcing=..., ic=...)
```

for each run, substituting the concrete field values for that run's
coordinate. The return value can be anything — a scalar, a dict, a dataframe,
a custom result object. PyEns treats it as opaque.

To execute the ensemble, pass a fully-bound `EnsembleSpec` to `run()`:

```python
result = runner.run(spec)
```

`run()` accepts an `EnsembleSpec` directly. If you have a `BoundSpec` (the
object returned by `EnsembleSpec.freeze()`), bind it to a concrete spec
first:

```python
runnable = param_map(parameters=theta)   # BoundSpec → EnsembleSpec
result   = runner.run(runnable)
```

`EnsembleRunner` is stateless: you can call `run()` on the same runner
multiple times with different specs, or call it on the same spec multiple
times.

---

## Backends

### What a backend does

A backend is responsible for one thing: applying a callable to a sequence of
input dicts and returning the results in the same order. Every backend in
PyEns honours two guarantees:

1. **Order is preserved.** The result at position `i` corresponds to the run
   at position `i` in `iter_runs()`, regardless of the order in which the
   backend completes the work internally.

2. **Exceptions are caught and returned, not re-raised.** If the model raises
   for run `i`, that exception instance is stored at position `i` in the
   results. The remaining runs continue normally.

These guarantees mean you can write result-processing code once and it works
identically across all backends.

### SequentialBackend

`SequentialBackend` runs each model call in the current process, one at a
time, in a plain Python loop. It requires no extra dependencies.

```python
from pyens.backends import SequentialBackend

runner = EnsembleRunner(my_model, SequentialBackend())
result = runner.run(spec)
```

Use `SequentialBackend`:

- **When developing and debugging.** Start here before switching to any
  parallel backend. Exceptions produce full Python tracebacks with correct
  line numbers and local variable values. Parallel backends make failures
  harder to diagnose.
- **When the model manages its own parallelism.** If `my_model` uses threads,
  OpenMP, or a parallel library internally, adding a process-level backend on
  top rarely helps and can hurt performance.
- **In constrained environments.** Some HPC login nodes, interactive
  notebooks, and container environments restrict subprocess creation.

:::{tip}
When a run fails unexpectedly with `LocalBackend` and the error message is
hard to interpret, switch to `SequentialBackend` to reproduce it. You will
get a full traceback with the original exception in context.
:::

### LocalBackend

`LocalBackend` distributes runs across multiple CPU cores using Python's
:mod:`concurrent.futures.ProcessPoolExecutor`. Runs execute in separate
worker processes; results are collected and returned in submission order.

```python
from pyens.backends import LocalBackend

runner = EnsembleRunner(my_model, LocalBackend(n_workers=4))
result = runner.run(spec)
```

`n_workers` controls how many worker processes to start. The default
(`None`) lets Python choose, typically around the number of available CPU
cores.

:::{important} **Pickling requirement**

Python's multiprocessing machinery serialises (pickles) the model callable
and every field value before sending them to worker processes. Two
requirements follow directly:

- **The model function must be defined at module level.** Lambdas, closures,
  and methods of locally-defined classes cannot be pickled and will raise
  `PicklingError` or `AttributeError`.
- **Every field value in the spec must be picklable.** Open file handles,
  database connections, GUI objects, and many C-extension objects cannot be
  pickled.

If you encounter a pickling error, switch to `SequentialBackend` to
reproduce the failure with a full traceback, fix the underlying issue, then
switch back to `LocalBackend`.
:::

Use `LocalBackend` when:

- The ensemble is large enough that serial execution is the bottleneck.
- The model is CPU-bound and benefits from true parallelism (separate
  processes bypass Python's GIL).
- The model function and all field values satisfy the pickling requirement.

---

## EnsembleResult

`runner.run(spec)` returns an `EnsembleResult` — an ordered collection of
`RunRecord` objects, one per run, in `iter_runs()` order.

### RunRecord

Each `RunRecord` holds two pieces of information:

**`coordinate`** — the axis labels that identified this run, exactly as
`iter_runs()` would produce. For example, `{'site': 'A', 'param_set': 2}`.
Fields declared with `Fixed` do not appear (they have no axis and do not
vary between runs).

**`output`** — the return value of the model callable for this run, or the
exception instance if the run failed.

Iterating the result yields `RunRecord` objects:

```python
for record in result:
    if record.failed:
        print(f"  failed: {record.coordinate}  —  {record.output!r}")
    else:
        process(record.output, record.coordinate)
```

The `failed` property is `True` when `output` is an exception instance and
`False` otherwise. It is the right way to distinguish successful runs from
failed ones.

### Counts

```python
result.n_runs    # total number of runs (succeeded + failed)
result.n_failed  # number of runs where the model raised an exception
```

Check `n_failed` before processing outputs. A non-zero value means some
elements of `outputs` are exception instances rather than model results.

### Outputs and coordinates

`outputs` and `coordinates` return plain lists in run order:

```python
result.outputs      # [output_0, output_1, ..., output_n]
result.coordinates  # [{'site': 'A', ...}, {'site': 'B', ...}, ...]
```

The element at position `i` in `outputs` corresponds to the element at
position `i` in `coordinates` — the same correspondence as `iter_runs()`.
This holds even when some runs have failed: failed runs appear as their
exception instance at the appropriate position in `outputs`.

### Succeeded and failed runs

`succeeded` and `failed` return filtered lists of `RunRecord` objects:

```python
good = result.succeeded   # records where the model returned normally
bad  = result.failed      # records where the model raised an exception
```

Because failed runs are stored in position rather than discarded, the
indices of `result.outputs` and `result.coordinates` always correspond.
The `succeeded` and `failed` lists are filtered views for convenience; they
do not affect the underlying ordering.

### Looking up a run by coordinate

`result[coord]` retrieves the `RunRecord` for a specific coordinate:

```python
record = result[{"site": "A", "param_set": 2}]
record.output      # the model output for that run
record.failed      # False if it succeeded
```

This is most useful for spot-checking a particular run after the ensemble
completes. A `KeyError` is raised if the coordinate does not exist in the
result.

The coordinates in `EnsembleResult` are the same dicts that `iter_runs()`
produces, so you can combine `sel()` on the spec with `result[coord]` on the
result:

```python
# Find the inputs for a run of interest using the spec
inputs = spec.sel(site="A", param_set=2)   # returns the inputs dict

# Find the corresponding output in the result
record = result[{"site": "A", "param_set": 2}]
```

---

## Practical guidance

**Start with `SequentialBackend`.** Develop and test your ensemble
sequentially first. Only switch to `LocalBackend` once you have confirmed
that the model runs correctly and that all field values are picklable.

**Check `n_failed` before processing outputs.** A guard at the start of
result-processing code prevents downstream errors from obscuring the real
failure:

```python
result = runner.run(spec)

if result.n_failed > 0:
    for rec in result.failed:
        print(f"Failed: {rec.coordinate!r}  —  {rec.output!r}")

for rec in result.succeeded:
    store(rec.coordinate, rec.output)
```

**The runner is reusable.** The same `EnsembleRunner` can be called with
different specs — useful in iterative workflows where the spec changes each
iteration but the model and backend stay fixed:

```python
runner = EnsembleRunner(my_model, LocalBackend())

for iteration in range(n_iterations):
    new_spec = algorithm.suggest_spec()
    result   = runner.run(new_spec)
    algorithm.update(result)
```

**Use `result[coord]` for spot-checking.** Rather than searching through
`outputs` manually, look up a specific run's outcome directly:

```python
record = result[{"site": "B", "param_set": 0}]
```
