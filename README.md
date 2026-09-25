# PyEns

**PyEns** is a Python library for specifying and executing structured ensemble
model runs. It is designed for scientific workflows where the same model must
be evaluated at many combinations of inputs and where keeping track of which output came
from which input is important.

PyEns fills a specific gap: a clean, programmatic Python API for describing
the *structure* of an ensemble (which inputs vary independently, which co-vary)
combined with a backend-agnostic execution layer. You describe what to run
once; PyEns handles the combinatorics and drives execution locally or on an
HPC cluster.

---

## Installation

```bash
pip install pyens
```

Running on a Grid Engine cluster (`qsub`) needs no extras: see
[Running on a Grid Engine Cluster](https://arob5.github.io/PyEns/user_guide/gridengine.html).

---

## Quick start

### Define the ensemble structure

An `Axis` names one dimension of variation. A `Grid` assigns values along an
axis. Fields on the **same axis** co-vary (zip semantics); fields on
**different axes** are crossed (Cartesian product).

```python
from pyens import Axis, EnsembleSpec, Fixed, Grid

# Two named sites
sites = Axis("site", labels=["harvard_forest", "niwot_ridge"])

# 50 anonymous parameter draws
members = Axis("member", size=50)

spec = EnsembleSpec(inputs={
    # Fixed: same value every run
    "config":     Fixed({"dt": 3600, "spin_up": True}),

    # Grid on sites: climate and IC co-vary — paired by design
    "climate":    Grid([climate_hf, climate_nr], along=sites),
    "ic":         Grid([ic_hf,      ic_nr],      along=sites),

    # Grid on members: independent parameter sets crossed with sites
    "parameters": Grid(param_draws, along=members),
})

spec.n_runs     # 100 = 2 sites × 50 members
print(spec.describe())
```

```
EnsembleSpec: 100 runs
  Axes (2):
    Axis('site', labels=['harvard_forest', 'niwot_ridge'])
    Axis('member', size=50)
  Fields (4):
    config:     Fixed  [fixed]
    climate:    Grid   [site]
    ic:         Grid   [site]
    parameters: Grid   [member]
```

### Run it

Iterate the spec directly for simple cases:

```python
results = {}
for inputs, coord in spec.iter_runs():
    results[tuple(coord.items())] = my_model(**inputs)
```

Or use `EnsembleRunner` for parallel execution and structured results:

```python
from pyens import EnsembleRunner, LocalBackend

runner = EnsembleRunner(my_model, backend=LocalBackend(n_workers=8))
result = runner.run(spec)
```

Switch to an HPC cluster by swapping the backend — the spec and model are
unchanged:

```python
from pyens.backends import GridEngineBackend

backend = GridEngineBackend(walltime="01:00:00", n_jobs=50,
                            work_dir="/shared/myproject/pyens_batches")
result = EnsembleRunner(my_model, backend).run(spec)
```

### Save for reproducibility

```python
spec.dump("runs/2026-05-29/spec.json")

# Later — exact reconstruction
spec = EnsembleSpec.load("runs/2026-05-29/spec.json")
```

The JSON file is human-readable, records the PyEns version and timestamp, and
is suitable for committing alongside results in version control.

---

## From xarray

If your inputs are already in an xarray `Dataset`, `pyens.xarray` builds the
fields for you (`pip install pyens[xarray]`). Each dim becomes an `Axis` whose
labels are the dim's coordinate values, or the positions `0, 1, …` when it has
no coordinate. Each data variable becomes a `Grid` along its own dims.
Variables that share a dim share its axis, so they zip on it, just as xarray
aligns them:

```python
import xarray as xr
from pyens import EnsembleSpec
from pyens.xarray import dataset_as_field, fields_from_dataset

# Parameters over (member, site); soil depth over site only.
table = xr.Dataset(
    {
        "leaf_area":  (("member", "site"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "soil_depth": ("site", [0.5, 0.8]),
    },
    coords={"site": ["harvard_forest", "niwot_ridge"]},
)

# Forcing over (site, time): each run gets its site's whole time series.
forcing = xr.Dataset(
    {"tair": (("site", "time"), [[280.0, 281.0, 282.0], [270.0, 271.0, 272.0]])},
    coords={"site": ["harvard_forest", "niwot_ridge"]},
)

spec = EnsembleSpec(inputs={
    **fields_from_dataset(table),                          # leaf_area, soil_depth
    "climate": dataset_as_field(forcing, along="site"),    # one sub-Dataset per site
})
spec.n_runs   # 6 = 3 members × 2 sites
```

Pass `along=` to choose which dims are ensemble axes. The other dims stay
inside each run's value, which is then a `DataArray` (or, with
`dataset_as_field`, a `Dataset`) slice instead of a plain number. See
[Building Fields from xarray](https://arob5.github.io/PyEns/user_guide/xarray.html)
for the exact conversion rules, including how date labels are handled.

---

## Core concepts

| Concept | Description |
|---|---|
| `Axis` | A named dimension — sites, ensemble members, parameter draws, … |
| `Fixed` | A field that takes the same value every run |
| `Grid` | A field whose value is indexed along one or more axes |
| `EnsembleSpec` | Assembles fields into a complete ensemble description |
| `PartialSpec` | A partially-applied spec with free fields — useful for optimizers and samplers |
| `EnsembleRunner` | Drives a model callable over a spec via a pluggable backend |

---

## Documentation

Full documentation is at **https://arob5.github.io/PyEns/**.

| Page | Description |
|---|---|
| [Data Model](https://arob5.github.io/PyEns/user_guide/data_model.html) | Axes, fields, specs — zip vs Cartesian product |
| [Building Fields from xarray](https://arob5.github.io/PyEns/user_guide/xarray.html) | `pyens.xarray`: dims to axes, variables to fields |
| [Running an Ensemble](https://arob5.github.io/PyEns/user_guide/running.html) | Backends, `EnsembleRunner`, structured results |
| [Running on a Grid Engine Cluster](https://arob5.github.io/PyEns/user_guide/gridengine.html) | `GridEngineBackend`: array jobs via `qsub`, failure handling |
| [Reproducibility and Serialization](https://arob5.github.io/PyEns/user_guide/reproducibility.html) | Saving specs, custom codecs, production workflow |

---

## Design goals

- **Zero core dependencies.** `Axis`, `Fixed`, `Grid`, and `EnsembleSpec` use
  only the Python standard library. Backends are optional extras.
- **Explicit structure.** The ensemble's combinatorial structure is a
  first-class object — inspectable, serializable, and validated before any
  model is run.
- **Backend agnostic.** The same spec runs locally or on an HPC cluster
  without modification.
- **Composable.** `EnsembleSpec.freeze()` turns a spec into a callable that
  accepts free fields.
