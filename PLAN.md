# pyens — Implementation Plan

This is a living document. Update it as decisions are made and phases complete.

---

## Phase 1: Core Ensemble Spec (no dependencies)

**Goal:** A pure-Python layer for specifying ensemble input structure, with no external dependencies. Testable with toy functions immediately.

### 1.1 `Axis` (`pyens/axis.py`)
- [ ] `Axis(name, *, labels=None, size=None)` — named dimension
  - Identity-based equality (two `Axis` objects are the same dimension iff they are the same Python object)
  - `labels` accepted as any `Sequence`; stored as `tuple` internally
  - `label_at(index)` returns the label at a position
  - `__repr__` that clearly shows name + size or labels
  - Validation: labels must be unique; exactly one of `labels`/`size` required
- [ ] Tests: creation, validation errors, `label_at`, repr

### 1.2 Field Specs (`pyens/fields.py`)
- [ ] `FieldSpec` abstract base with `.axes -> tuple[Axis, ...]` and `.value_at(idx_coords) -> Any`
- [ ] `Fixed(value)` — zero axes; always returns `value`
- [ ] `Grid(values, along=axis_or_list_of_axes)` — values indexed along axes
  - Single axis: `values` is a flat sequence of length `axis.size`
  - Multiple axes: `values` is nested; `values[i][j]` for two axes
  - Validate that top-level `len(values) == axes[0].size` at construction
- [ ] Tests: Fixed, Grid single-axis, Grid multi-axis, shape validation errors

### 1.3 `EnsembleSpec` and `BoundSpec` (`pyens/spec.py`)
- [ ] `EnsembleSpec(*, inputs: dict[str, FieldSpec])`
  - `_collect_axes()` — unique Axis instances across all fields, insertion-ordered
  - `axes` property — tuple of unique axes
  - `n_runs` property — product of all axis sizes
  - `field_names` property
  - `describe()` — human-readable summary for pre-run inspection
  - `iter_runs()` — `Iterator[tuple[RunInputs, Coordinate]]`; lazy Cartesian product
  - `freeze(free: list[str]) -> BoundSpec`
- [ ] `EnsembleSpec.from_runs(*run_dicts)` classmethod — simple flat-list entry point
- [ ] `BoundSpec(base_inputs, free_field_names)` — callable; `__call__(**kwargs) -> EnsembleSpec`
- [ ] Tests:
  - Single Fixed field → 1 run
  - Single Grid → n runs
  - Two Grids on same Axis → n runs (zip, not product)
  - Two Grids on different Axes → n×m runs (product)
  - Multi-axis Grid → correct value indexing
  - `describe()` output
  - `from_runs()`
  - `freeze()` and `BoundSpec.__call__()`
  - Error cases: mismatched Grid shape, freeze on unknown field

### 1.4 Public API (`pyens/__init__.py`)
- [ ] Export: `Axis`, `Fixed`, `Grid`, `EnsembleSpec`, `BoundSpec`
- [ ] `pyproject.toml` with zero core dependencies, `dev` extra for pytest

---

## Phase 2: Execution Layer

**Goal:** Run an `EnsembleSpec` against a model and collect results. Testable locally.

### 2.1 `EnsembleResult` (`pyens/result.py`)
- [ ] `EnsembleResult` — stores list of `(output, coordinate)` pairs
  - `coordinates` — list of coordinate dicts (one per run)
  - `outputs` — list of raw model outputs
  - `to_xarray()` — convert to `xr.Dataset` (optional; requires xarray)
  - `__getitem__(coord_dict)` — retrieve result by coordinate
  - `failed` — list of coordinates where the run raised an exception
- [ ] Design decision: eager vs lazy collection (start eager)
- [ ] Tests: basic collection, `to_xarray()`, error handling

### 2.2 Backend abstraction (`pyens/backends/base.py`)
- [ ] `Backend` abstract base class
  - `map(fn, inputs_list) -> list[Any]` — execute fn on each element, return results
  - `shutdown()` context manager
- [ ] Tests: protocol check

### 2.3 `LocalBackend` (`pyens/backends/local.py`)
- [ ] Wraps `concurrent.futures.ProcessPoolExecutor`
- [ ] `n_workers` parameter
- [ ] Configurable error handling: raise on first failure vs. collect all failures
- [ ] Tests: correctness, exception propagation

### 2.4 `EnsembleRunner` (`pyens/runner.py`)
- [ ] `EnsembleRunner(model: Callable, backend: Backend)`
  - `model` is any `fn(**inputs) -> output` callable
  - `run(spec: EnsembleSpec) -> EnsembleResult`
- [ ] Tests: toy function (e.g., sum of inputs), LocalBackend, error propagation

---

## Phase 3: HPC Backend (Parsl + SGE)

**Goal:** Run ensembles on BU SCC without changing any spec or model code.

### 3.1 `ParslBackend` (`pyens/backends/parsl.py`)
- [ ] Wraps Parsl `HighThroughputExecutor` + `GridEngineProvider`
- [ ] Constructor accepts standard Parsl provider kwargs + BU SCC presets
- [ ] `BUSCCConfig` helper with sensible defaults for BU SCC (module loads, OMP threads, memory)
- [ ] Lazy import of `parsl` (not a hard dependency)
- [ ] Tests: mocked Parsl executor (no actual HPC needed in CI)

### 3.2 Backend configuration helpers
- [ ] `SGEConfig` dataclass — typed interface for common SGE options
- [ ] `BUSCCConfig(n_cores, memory_gb, walltime, env_setup)` — BU SCC preset

---

## Phase 4: Advanced Spec Features

**Goal:** Handle the more complex ensemble structures from the design requirements.

### 4.1 `BoundSpec` improvements
- [ ] `max_runs` guard parameter to `freeze()` — raises if the bound spec would exceed the limit
- [ ] `BoundSpec.describe()` — show the structure with the free fields indicated

### 4.2 Nested/structured field values
- [ ] Currently `Grid` and `Fixed` accept any Python object as values
- [ ] Decide: should pyens know about pySIPNET `SIPNETParametersV1` or stay generic? (Stay generic — no.)
- [ ] Document the pattern for users: create a list of Pydantic model instances, pass to `Grid`

### 4.3 Metadata per run
- [ ] Attach arbitrary metadata to runs in `EnsembleResult`
- [ ] Design: metadata passed through `coordinate` dict or separate dict?

---

## Phase 5: Integrations

**Goal:** First-class support for the two primary use cases.

### 5.1 pySIPNET adapter
- [ ] `SIPNETAdapter(runner: SIPNETRunner)` — wraps `runner.run(parameters, climate)` as a pyens-compatible callable
- [ ] Document the field naming convention (`parameters`, `climate`, `initial_conditions`)
- [ ] Example notebook: multi-site IC ensemble with pySIPNET

### 5.2 ProbPipe integration (requires design discussion first)
- [ ] Define the interface contract: what does ProbPipe expect from a "model"?
- [ ] Option A: expose `param_map` as a plain callable — no ProbPipe-awareness in pyens
- [ ] Option B: pyens `EnsembleSpec` accepts ProbPipe `Distribution` as a field value
- [ ] Recommendation: start with Option A; revisit Option B when the interface is clearer

---

## Open Design Questions

| Question | Status | Notes |
|---|---|---|
| ProbPipe Distribution as field value | Unresolved | See CLAUDE.md open questions |
| Result persistence (files vs. DB) | Unresolved | Start with in-memory; add file output when needed |
| xarray as hard vs. optional dep | Unresolved | Keep optional for now; `to_xarray()` raises `ImportError` if missing |
| `Free` field spec | Deferred | Not in v1; `freeze()` on concrete specs is sufficient |
| Adaptive ensembles (iterative, not one-shot) | Future | Out of scope for v1 |

---

## Milestone Summary

| Milestone | Contents | Testable with |
|---|---|---|
| M1 | Phase 1 complete | Pure Python, no deps |
| M2 | Phase 2 complete | Local multiprocessing |
| M3 | Phase 3 complete | BU SCC |
| M4 | Phases 4–5 | pySIPNET + ProbPipe |
