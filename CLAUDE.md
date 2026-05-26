# pyens — Developer Context

## What is this project?

**pyens** is a Python package for specifying and executing structured ensemble model runs. The motivation is scientific computing with land surface models (LSMs) and similar "run a model at many different inputs" workflows, but the design is fully generic — it wraps any Python callable.

The project fills a specific gap: no existing tool provides a clean, **programmatic** Python API for specifying structured, named ensemble inputs combined with a backend-agnostic execution layer. Parsl and Dask handle task scheduling; Hydra handles config management; pyens handles the *ensemble input algebra* and ties execution together.

## What this project is NOT

- **Not an execution framework.** Execution is delegated to pluggable backends (local multiprocessing, Parsl for HPC). pyens specifies *what* to run, not *how*.
- **Not a config management tool.** That's Hydra. pyens focuses on the combinatorial structure of ensemble inputs.
- **Not a probabilistic inference library.** That's [ProbPipe](https://github.com/TARPS-group/prob-pipe). pyens is a complementary tool — ProbPipe generates parameter samples, pyens executes the model at those samples efficiently.
- **Not model-specific.** The first concrete use case is SIPNET via [pySIPNET](https://github.com/TARPS-group/pySIPNET), but the design is intentionally generic.

## Core Concept: The Axis Model

An ensemble is a mapping from a multi-dimensional coordinate space to individual model inputs. Each "dimension" of that space is an **`Axis`** — a named object with a size and optional labels. Each input field is either:

- **`Fixed(value)`** — same value for every run; contributes no axis.
- **`Grid(values, along=axis_or_axes)`** — a sequence of values indexed along one or more axes.

**The key rule governing ensemble structure:**

> Two `Grid` fields referencing the **same `Axis` instance** are *aligned* (zip semantics — they co-vary along that dimension). Two fields referencing **different `Axis` instances** are *crossed* (Cartesian product).

This single rule handles all ensemble structures:

```python
sites   = Axis("site", labels=["harvard_forest", "niwot_ridge"])
members = Axis("member", size=100)

# Paired inputs: climate and IC share the same `sites` axis → 2 runs
spec = EnsembleSpec(inputs={
    "parameters":         Fixed(base_params),
    "climate":            Grid([c1, c2], along=sites),
    "initial_conditions": Grid([ic1, ic2], along=sites),  # aligned with climate
})

# Crossed inputs: different axes → Cartesian product → 200 runs
spec = EnsembleSpec(inputs={
    "climate":            Grid([c1, c2], along=sites),
    "initial_conditions": Grid(ic_list, along=members),  # independent axis
})
```

## Partial Application (`freeze`)

`EnsembleSpec.freeze(free=["parameters"])` returns a `PartialSpec` — a callable that accepts only the specified free fields and returns a complete `EnsembleSpec`. This is the mechanism for constructing parameter-to-output maps usable by sampling and optimization algorithms.

```python
# Build spec with climate/IC fixed, parameters to be provided later
bound = full_spec.freeze(free=["parameters"])

# Later, supply the free field (e.g., from a sampler):
runnable = bound(parameters=Grid(sampled_params, along=param_axis))
results = runner.run(runnable)
```

## Simple Entry Point

For cases where the algebra is overkill, `EnsembleSpec.from_runs()` accepts a flat list of input dicts:

```python
spec = EnsembleSpec.from_runs(
    {"x": 1.0, "y": "a"},
    {"x": 2.0, "y": "b"},
    {"x": 3.0, "y": "c"},
)
# Equivalent to a single Axis of size 3 with integer labels
```

## Backend Abstraction

`EnsembleRunner(model, backend)` separates *what to run* (the model callable) from *how to run it* (the backend):

```python
# Development: local multiprocessing
runner = EnsembleRunner(my_model, backend=LocalBackend(n_workers=8))

# Production: BU SCC via Parsl + SGE
runner = EnsembleRunner(my_model, backend=ParslBackend(
    provider="SGE",
    nodes_per_block=1,
    walltime="08:00:00",
    scheduler_options="#$ -pe omp 4 -l mem_total=16G",
    worker_init="module load python/3.11; source activate myenv",
    max_blocks=20,
))
```

The same `EnsembleSpec` and model work unchanged across backends.

## Documentation Standards

> **Before writing or editing any documentation, check
> [`docs/conventions.md`](docs/conventions.md)** for naming rules, terminology,
> and tone guidelines. The most important rule: use **PyEns** in prose and headings,
> `pyens` in code.

**Documentation quality is a first-class requirement in this project.** pyens is a research tool used by people who may not be professional software engineers. Clear, complete documentation is part of correctness — an undocumented interface is an incomplete interface.

### Rules

- **Every public class, method, and function has a docstring. No exceptions.**
- Docstrings use **Google style**: one-line summary, then `Args:`, `Returns:`, `Raises:`, and `Examples:` sections as appropriate. The one-liner must stand alone as a summary.
- **`Examples:` sections are required** for any class or function whose usage is not immediately obvious from the signature. Examples should be runnable (or clearly illustrative).
- **Type annotations are required everywhere.** Use `from __future__ import annotations` at the top of each file.
- Unit metadata, coordinate conventions, and any non-obvious invariants are documented at their point of definition.
- `CHANGELOG.md` is updated with every user-visible API change.

### What NOT to comment

- Don't describe what the code does if well-named identifiers already do that.
- Don't reference issue numbers, PR details, or task context in code.
- Reserve inline `#` comments for non-obvious invariants, subtle constraints, or workarounds for specific bugs.

## Planned Package Structure

```
pyens/
├── pyens/
│   ├── __init__.py          # Public API exports
│   ├── axis.py              # Axis — named ensemble dimension
│   ├── fields.py            # Fixed, Grid field specs
│   ├── spec.py              # EnsembleSpec, PartialSpec
│   ├── result.py            # EnsembleResult — structured output with coordinates
│   ├── runner.py            # EnsembleRunner — ties spec + model + backend
│   └── backends/
│       ├── __init__.py
│       ├── base.py          # Backend abstract base class
│       ├── local.py         # LocalBackend (concurrent.futures)
│       └── parsl.py         # ParslBackend (HPC via SGE/SLURM via Parsl)
├── tests/
│   ├── conftest.py
│   ├── test_axis.py
│   ├── test_fields.py
│   ├── test_spec.py
│   ├── test_runner.py
│   └── test_backends.py
├── docs/
│   └── plan.md              # Living implementation plan
├── CLAUDE.md
├── README.md
├── CHANGELOG.md
└── pyproject.toml
```

The `pyens/` core (axis, fields, spec) has **zero dependencies** and can be tested with plain Python. Backends are optional extras.

## Target HPC Environment

Primary deployment: **Boston University SCC**, which runs **SGE (Sun Grid Engine)**. The `ParslBackend` wraps Parsl's `GridEngineProvider`. Parsl's pilot-job model (provision a block of SGE resources, then schedule many tasks within it) efficiently handles large ensembles without per-task job-submission overhead.

## Open Questions (to resolve in future sessions)

1. **ProbPipe-native integration.** Should pyens accept a ProbPipe `Distribution` as a field value and automatically sample from it? This would blur the package boundary but could significantly improve ergonomics for inference workflows. Needs design discussion before implementing.

2. **Metadata and queryability.** How should ensemble run metadata be persisted? Options range from flat files (simple, always works) to SQLite/MLflow (enables cross-run queries like "show all runs where `aMax > 12`"). The right answer depends on how large ensembles grow and user workflows.

3. **Result structure.** The current plan is to return an `EnsembleResult` wrapping an `xarray.Dataset` with axes as named dimensions. Whether xarray is a hard dependency or optional needs to be decided when the result layer is implemented.

## Primary Use Cases

1. **pySIPNET ensemble runs.** Sweep over parameters, initial conditions, and climate drivers across sites.
2. **ProbPipe integration.** Expose the model as a parameter-to-output map for use inside inference and optimization algorithms.
3. **Generic Python model ensemble.** Wrap any `fn(**inputs) -> output` callable.

## Development Conventions

- **Python ≥ 3.11**
- **Pydantic v2** for any validated data models (not currently used in core)
- **xarray** for the result layer (optional dependency)
- **Parsl** for HPC backend (optional dependency; `pip install pyens[parsl]`)
- Tests must run without HPC — `LocalBackend` is always available.
- No comments unless the WHY is non-obvious.
- Type hints everywhere, `from __future__ import annotations` in every file.
