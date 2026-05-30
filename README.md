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

For HPC execution via [Parsl](https://parsl-project.org):

```bash
pip install pyens[parsl]
```

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
unchanged.

### Save for reproducibility

```python
spec.dump("runs/2026-05-29/spec.json")

# Later — exact reconstruction
spec = EnsembleSpec.load("runs/2026-05-29/spec.json")
```

The JSON file is human-readable, records the PyEns version and timestamp, and
is suitable for committing alongside results in version control.

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
| [Running an Ensemble](https://arob5.github.io/PyEns/user_guide/running.html) | Backends, `EnsembleRunner`, structured results |
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
  accepts free fields — making it straightforward to wrap an ensemble as a
  parameter-to-output map for samplers and optimizers.
