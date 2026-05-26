# The Data Model

This page explains the fundamental abstractions in pyens: **axes**, **fields**, and **specs**. Understanding these three concepts is all you need to describe even very complex ensemble structures.

## Motivation

Suppose you have a model — any function that takes some inputs and produces some outputs:

```python
def my_model(x, noise_level):
    return x ** 2 + noise_level * 0.1
```

Now suppose you want to run this model across a range of `x` values and a range of `noise_level` values. The straightforward approach uses a nested loop:

```python
results = []
for x in [1.0, 2.0, 3.0]:
    for noise in [0.0, 0.5, 1.0]:
        results.append(my_model(x, noise))
```

This works, but it has a serious limitation: **the structure of the result is implicit**. `results` is a flat list of 9 numbers. There is nothing in the list itself that tells you which result corresponds to `x=2.0, noise=0.5`. If you want to know, you have to reconstruct that mapping manually — and errors in that reconstruction are easy to make and hard to detect.

The problem gets worse as the number of inputs grows, or when some inputs are paired rather than crossed, or when inputs are structured objects rather than scalars.

pyens solves this by giving the ensemble structure an explicit, named representation. Instead of writing nested loops, you declare what varies and how, and the library handles the enumeration and labeling.

---

## Axes

An `Axis` represents one dimension of variation in the ensemble. It has a name and a set of **labels** — one for each position along that dimension.

```python
from pyens import Axis

# An axis with explicit string labels
x_axis = Axis("x", labels=[1.0, 2.0, 3.0])

# An axis with implicit integer labels (0, 1, 2, ...)
noise_axis = Axis("noise_level", size=3)
```

The name is used when labeling results so you can identify which output came from which input. The labels are attached to those results as coordinates.

### Why labels matter

When results come back from a large ensemble, you need to be able to slice and query them. Coordinates with meaningful labels — site names, parameter set IDs, date ranges — are far more useful than bare integer indices.

```python
# String labels make results self-documenting
sites = Axis("site", labels=["harvard_forest", "niwot_ridge", "metolius"])

# Integer labels work fine when the values themselves are the identity
members = Axis("member", size=100)
```

### The identity rule

The most important thing to understand about axes is how they determine ensemble structure. In pyens, **two fields that reference the same `Axis` object co-vary along that dimension**. Two fields that reference different `Axis` objects are independent.

This is intentional and precise: identity means Python object identity (`is`), not name equality. Two `Axis` objects with the same name are still considered different dimensions.

```python
ax_a = Axis("x", size=3)
ax_b = Axis("x", size=3)  # same name, different object

ax_a is ax_a  # True  → same dimension
ax_a is ax_b  # False → different dimensions
```

The practical consequence of this rule — what "co-vary" and "independent" mean for the number of runs — is explained in detail in the [EnsembleSpec](#ensemblespec-1) section below.

---

## Fields

A **field** is a description of how one named input to the model is determined for each run. There are two kinds: `Fixed` and `Grid`.

### Fixed

`Fixed` describes an input that takes the same value for every run. It contributes no axis and adds no dimension to the ensemble.

```python
from pyens import Fixed

f = Fixed(42)
f.axes    # ()  — no axes; contributes no dimension
```

`Fixed` is appropriate for any input that serves as a constant background — default parameters, a shared configuration object, a pre-loaded dataset that all runs share.

```python
default_config = {"seed": 0, "precision": "float64"}
config_field = Fixed(default_config)
```

### Grid

`Grid` describes an input that varies. It takes a sequence of values and one or more axes that index into that sequence.

#### Single axis

The simplest case: one value per position along one axis.

```python
from pyens import Axis, Grid

x_axis = Axis("x", labels=[1.0, 2.0, 3.0])
x_field = Grid([1.0, 2.0, 3.0], along=x_axis)
```

The length of the values must match the size of the axis:

```python
x_axis = Axis("x", size=3)
Grid([10, 20, 30], along=x_axis)   # OK — length 3 matches size 3
Grid([10, 20],     along=x_axis)   # Error — length 2 does not match size 3
```

The values can be anything — scalars, dicts, dataframes, model objects. pyens does not inspect the values themselves.

```python
sites = Axis("site", labels=["site_A", "site_B", "site_C"])

# Values can be structured objects
climate_field = Grid(
    [
        {"temperature": 12.0, "precipitation": 800.0},
        {"temperature": -2.0, "precipitation": 500.0},
        {"temperature": 18.0, "precipitation": 1200.0},
    ],
    along=sites,
)
```

#### Multiple axes

A `Grid` can vary along more than one axis simultaneously. In this case, the values are a nested sequence: the outermost level indexes the first axis, the next level indexes the second axis, and so on.

```python
sites   = Axis("site",   labels=["site_A", "site_B"])
members = Axis("member", size=3)

# Values[i][j] is the value at site i, member j
ic_values = [
    [100.0, 120.0, 140.0],  # site_A: members 0, 1, 2
    [200.0, 220.0, 240.0],  # site_B: members 0, 1, 2
]
ic_field = Grid(ic_values, along=[sites, members])
```

A multi-axis `Grid` is useful when an input is **jointly determined** by multiple factors. An initial condition that differs by both site and ensemble member is a natural example: you cannot specify it as a function of site alone or member alone — you need a value for every (site, member) combination.

---

## EnsembleSpec

An `EnsembleSpec` assembles named fields into a complete ensemble. It takes a dictionary mapping input names to `FieldSpec` objects:

```python
from pyens import EnsembleSpec

spec = EnsembleSpec(inputs={
    "x":           Grid([1.0, 2.0, 3.0], along=x_axis),
    "noise_level": Fixed(0.5),
})
```

The spec then provides:

- `spec.n_runs` — the total number of model evaluations.
- `spec.describe()` — a human-readable summary to inspect before running.
- `spec.iter_runs()` — an iterator that yields `(inputs_dict, coordinate_dict)` for each run.

```python
for inputs, coord in spec.iter_runs():
    output = my_model(**inputs)
    print(coord, "→", output)
# {'x': 1.0} → 1.05
# {'x': 2.0} → 4.05
# {'x': 3.0} → 9.05
```

The `inputs` dict is ready to unpack directly into the model. The `coord` dict records the label of each axis for this run — this is what lets you trace every output back to its inputs.

### The zip / product rule

How the spec determines how many runs to produce — and which input values go together — follows a single rule:

:::{admonition} The axis identity rule
:class: important

Fields that share the **same `Axis` object** co-vary along that dimension (zip semantics). Fields on **different `Axis` objects** are varied independently (Cartesian product).
:::

Everything else in the data model follows from this one rule. The sections below work through the cases one at a time.

#### Case 1: One axis, multiple fields — zip

When two `Grid` fields reference the same axis, each run takes one value from each field at the same position along that axis. The fields are *paired*, not crossed.

```python
sites = Axis("site", labels=["site_A", "site_B", "site_C"])

spec = EnsembleSpec(inputs={
    "climate": Grid([c_A, c_B, c_C], along=sites),
    "ic":      Grid([i_A, i_B, i_C], along=sites),  # same axis object
})

spec.n_runs     # 3  (not 9)
print(spec.describe())
# EnsembleSpec: 3 runs
#   Axes (1):
#     Axis('site', labels=['site_A', 'site_B', 'site_C'])
#   Fields (2):
#     climate: Grid [site]
#     ic:      Grid [site]
```

The runs are:

| Run | `climate` | `ic` | coordinate |
|-----|-----------|------|------------|
| 0   | `c_A`     | `i_A` | `site='site_A'` |
| 1   | `c_B`     | `i_B` | `site='site_B'` |
| 2   | `c_C`     | `i_C` | `site='site_C'` |

This is the correct structure for inputs that are *naturally paired* — climate and initial conditions that both belong to a specific site. Crossing them would be nonsensical: you would be pairing site_A's climate with site_B's initial conditions.

#### Case 2: Two independent axes — Cartesian product

When two `Grid` fields reference *different* axes, every combination of their values is a distinct run.

```python
x_axis     = Axis("x",     size=3)
noise_axis = Axis("noise", size=4)

spec = EnsembleSpec(inputs={
    "x":     Grid([1.0, 2.0, 3.0],       along=x_axis),
    "noise": Grid([0.0, 0.1, 0.5, 1.0], along=noise_axis),
})

spec.n_runs    # 12  (3 × 4)
```

The runs cover every combination:

| `x` | `noise` |
|-----|---------|
| 1.0 | 0.0     |
| 1.0 | 0.1     |
| 1.0 | 0.5     |
| 1.0 | 1.0     |
| 2.0 | 0.0     |
| ... | ...     |

This is appropriate when the inputs are genuinely independent — when you have no reason to pair particular values of `x` with particular values of `noise`.

#### Case 3: Mixed — some axes shared, some independent

Axes can be mixed freely. Fields that share an axis co-vary along it; fields on different axes are crossed.

```python
sites   = Axis("site",   labels=["site_A", "site_B"])  # size 2
params  = Axis("params", size=5)

spec = EnsembleSpec(inputs={
    "climate": Grid(climate_list, along=sites),   # paired with ic on the sites axis
    "ic":      Grid(ic_list,      along=sites),   # same sites axis → co-varies with climate
    "theta":   Grid(param_list,   along=params),  # independent axis → crossed
})

spec.n_runs    # 2 × 5 = 10
```

Here, `climate` and `ic` are paired (they share `sites`), but both are crossed with `theta` (different axis). Each of the 10 runs gets a specific site's climate and IC together, combined with one of the 5 parameter sets.

#### Case 4: Multi-axis Grid and shared axes

A multi-axis `Grid` can share *some* axes with other fields and not others. The rule is the same: axes that appear in multiple fields are shared dimensions; new axes added by a `Grid` are crossed with the rest.

```python
sites   = Axis("site",   labels=["site_A", "site_B"])
params  = Axis("params", size=3)

# ic varies by both site and parameter set — shape (2 sites, 3 params)
ic_table = [
    [ia_p0, ia_p1, ia_p2],  # site_A
    [ib_p0, ib_p1, ib_p2],  # site_B
]

spec = EnsembleSpec(inputs={
    "climate": Grid(climate_list, along=sites),
    "ic":      Grid(ic_table,     along=[sites, params]),  # both axes
    "theta":   Grid(param_list,   along=params),           # shares params with ic
})

spec.n_runs    # 2 × 3 = 6  (sites × params — no new axes introduced)
```

This structure is appropriate when an initial condition depends on *both* the site and the parameter set — for example, when parameters define calibrated initial states that differ by location.

---

## Choosing the right structure

The choice between zip and Cartesian product should reflect the physical or logical relationships in your problem.

**Use the same axis (zip) when values are paired by design.** If your climate data and initial conditions were prepared together for each site, they belong to the same site axis. Crossing them would create nonsensical combinations.

**Use different axes (Cartesian product) when inputs are independent.** If you are sweeping a model parameter across a range of values, and that parameter is independent of which site you are running, use a separate axis.

**Use a multi-axis `Grid` when an input is jointly determined.** If an initial condition depends on both the parameter set and the site, use `Grid(values, along=[params, sites])`. This is more explicit than computing the right value inside the model, and keeps the input structure visible in the spec.

A useful check before running: call `spec.describe()` and verify that the axis count and run count match your expectations:

```python
print(spec.describe())
# EnsembleSpec: 6 runs
#   Axes (2):
#     Axis('site', labels=['site_A', 'site_B'])
#     Axis('params', size=3)
#   Fields (3):
#     climate: Grid [site]
#     ic:      Grid [site, params]
#     theta:   Grid [params]
```

If you expected 10 runs and see 6, or see 3 axes instead of 2, the structure of your spec does not match your intent — better to catch that here than after a long run.

---

## Simple entry point: `from_runs`

For straightforward cases — small ensembles, hand-crafted run lists, one-off experiments — the axis algebra may be more than you need. `EnsembleSpec.from_runs` accepts a flat list of input dictionaries, one per run:

```python
spec = EnsembleSpec.from_runs(
    {"x": 1.0, "noise_level": 0.0},
    {"x": 2.0, "noise_level": 0.5},
    {"x": 3.0, "noise_level": 1.0},
)

spec.n_runs    # 3
```

This creates a single anonymous axis of size 3 with integer coordinates (0, 1, 2). The axis algebra (`Axis`, `Grid`) is not involved. All dictionaries must have the same keys.

`from_runs` is a thin wrapper: the result is an ordinary `EnsembleSpec` and can be iterated, inspected with `describe()`, and passed to a runner exactly like any other spec. When you need more structure, you can switch to the full API without changing any downstream code.

---

## Coordinates

Every run in an ensemble has a **coordinate** — a dictionary mapping axis names to labels. Coordinates are what connect outputs back to inputs.

```python
sites  = Axis("site", labels=["site_A", "site_B"])
params = Axis("params", size=3)

spec = EnsembleSpec(inputs={
    "climate": Grid(climate_list, along=sites),
    "theta":   Grid(param_list,   along=params),
})

for inputs, coord in spec.iter_runs():
    print(coord)
# {'site': 'site_A', 'params': 0}
# {'site': 'site_A', 'params': 1}
# {'site': 'site_A', 'params': 2}
# {'site': 'site_B', 'params': 0}
# ...
```

`Fixed` fields do not appear in the coordinate — they have no axis and do not vary between runs.

Coordinates are designed to be used as result keys, index entries, or metadata tags. When you collect outputs from `iter_runs`, attach each output's coordinate to it so you can query or restructure the results later.

```python
results = {}
for inputs, coord in spec.iter_runs():
    key = (coord["site"], coord["params"])
    results[key] = my_model(**inputs)

# Retrieve output for a specific combination
results[("site_A", 2)]
```

---

## Partial application

Once a spec is defined, you can freeze some fields and expose others as free variables. This produces a **`BoundSpec`** — a callable that accepts only the free fields and returns a complete, runnable spec.

```python
# Fix climate and IC; leave parameters open
base_spec = EnsembleSpec(inputs={
    "theta":   Fixed(default_params),            # will be replaced
    "climate": Grid(climate_list, along=sites),
    "ic":      Grid(ic_list,      along=sites),
})

param_map = base_spec.freeze(free=["theta"])
```

You can then call `param_map` with any value for `theta`:

```python
# Single parameter set — fixed for all runs
runnable = param_map(theta=my_params)
runnable.n_runs    # 3 (just the sites axis)

# Grid of parameter sets — crossed with the sites axis
p_axis = Axis("param_set", size=10)
runnable = param_map(theta=Grid(param_list_10, along=p_axis))
runnable.n_runs    # 30 (10 param sets × 3 sites)
```

A plain Python value (like `my_params` above) is automatically wrapped in `Fixed`. A `Grid` or another `FieldSpec` is used directly.

`BoundSpec` is particularly useful as the interface between an ensemble runner and an algorithm that generates inputs programmatically — for example, an optimiser that proposes new parameter sets in each iteration:

```python
for iteration in range(n_iterations):
    proposed = algorithm.suggest(n=batch_size)   # list of parameter dicts
    batch_axis = Axis("proposal", size=batch_size)
    runnable = param_map(theta=Grid(proposed, along=batch_axis))

    outputs = []
    for inputs, coord in runnable.iter_runs():
        outputs.append((coord, my_model(**inputs)))

    algorithm.update(outputs)
```

The spec structure — which climate goes with which site, which IC values are used — is defined once and stays fixed across all iterations. Only the free field changes.
