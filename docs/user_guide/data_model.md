# The Data Model

This page explains the three core abstractions in PyEns — **axes**, **fields**,
and **specs** — from first principles. It also describes how to select individual
runs by coordinate and how PyEns relates to the xarray data model.

---

## What is an ensemble?

At its core, PyEns is about running a function — a model — many times with
different inputs. One model run is one function call. Its inputs are a collection
of **named values**:

```python
result = my_model(parameters=theta, forcing=climate_data, ic=initial_state)
```

Here `parameters`, `forcing`, and `ic` are the input names; `theta`,
`climate_data`, and `initial_state` are their values for this particular run.

:::{note}
Throughout this tutorial, we use toy examples motivated by ecological and climate
modeling applications. The dynamical models used in these fields often have various
inputs: parameters, initial conditions (IC), and forcing data.
:::

An **ensemble** is simply a collection of such function calls, each with the same
input names but potentially different values. The simplest possible representation
is a list of dicts, one dict per run:

```python
runs = [
    {"parameters": theta_1, "forcing": climate_A, "ic": state_1},
    {"parameters": theta_1, "forcing": climate_B, "ic": state_2},
    {"parameters": theta_2, "forcing": climate_A, "ic": state_1},
    {"parameters": theta_2, "forcing": climate_B, "ic": state_2},
]
```

PyEns's `EnsembleSpec.from_runs` wraps this idea directly:

```python
from pyens import EnsembleSpec

spec = EnsembleSpec.from_runs(
    {"parameters": theta_1, "forcing": climate_A, "ic": state_1},
    {"parameters": theta_1, "forcing": climate_B, "ic": state_2},
    {"parameters": theta_2, "forcing": climate_A, "ic": state_1},
    {"parameters": theta_2, "forcing": climate_B, "ic": state_2},
)
```

Once you have a spec, you iterate over it to drive your model:

```python
results = []
for inputs, coord in spec.iter_runs():
    results.append(my_model(**inputs))
```

`iter_runs()` yields `(inputs_dict, coordinate_dict)` pairs. The `inputs` dict
is passed directly to the model. The `coordinate` dict records a label for this
run — more on that shortly.

`from_runs` is a good entry point for small, hand-crafted ensembles. But as
the number of inputs and combinations grows, building the list manually becomes
unwieldy. That is the motivation for PyEns's algebra.

---

## Why a flat list is not enough

Consider adding one more dimension: instead of 2 parameter sets and 2 forcing
conditions, suppose you have 5 parameter sets, 10 forcing conditions, and 8
initial conditions — and you want every combination. You now need 400 runs.
Writing 400 dicts is obviously impractical; a loop is the natural approach:

```python
runs = []
for theta in param_sets:         # 5 values
    for forcing in forcing_list: # 10 values
        for ic in ic_list:       # 8 values
            runs.append({"parameters": theta, "forcing": forcing, "ic": ic})
```

This works for the simple Cartesian product, but it quickly breaks down:

**The inputs are not all independent.** Suppose `forcing` and `ic` are paired:
each forcing condition has a matching initial condition specific to it. Now the
`forcing` and `ic` loops must be zipped, not nested:

```python
for theta in param_sets:
    for forcing, ic in zip(forcing_list, ic_list):  # paired, not crossed
        runs.append(...)
```

Whether to nest or zip is easy to get wrong and nothing in the code will tell
you if you made a mistake. Adding a fourth input that is paired with one axis but
crossed with another compounds the problem.

**Some inputs are jointly determined.** Suppose the initial condition also depends
on which parameter set you're using — perhaps the parameters define a calibrated
initial state specific to each location. Now the IC is a function of *both* the
parameter set and the location:

```python
for p_idx, theta in enumerate(param_sets):
    for f_idx, forcing in enumerate(forcing_list):
        ic = ic_table[p_idx][f_idx]  # 2D lookup
        runs.append(...)
```

The loop structure no longer reflects the conceptual structure. To understand what
the code generates, a reader must mentally simulate its execution.

**The structure is implicit and fragile.** Change the order of `forcing_list` and
the pairings silently change. There is no way to inspect the loop and ask "how
many runs does this produce?" without running it. There is no validation step.

**The whole list is materialised immediately.** With 400 or 40,000 runs, you may
not want to hold all input objects in memory at once.

PyEns addresses all of these by giving the structure a first-class representation.

---

## The PyEns data model

The key insight is that an ensemble has *structure*: different inputs vary along
different dimensions, and some dimensions are shared across inputs. PyEns
represents this structure explicitly with three abstractions:

- An **`Axis`** names one dimension of variation and defines the set of values
  that dimension can take.
- A **`Grid`** attaches values to positions along one or more axes — it is a
  labeled lookup table.
- A **`Fixed`** attaches a single constant value to an input that does not vary.
- An **`EnsembleSpec`** assembles named fields into a complete description of
  the ensemble. It determines the total number of runs and how each input
  is resolved for each run.

The full set of runs is the Cartesian product of all *independent* axes. Two
fields that vary along the *same* axis co-vary rather than being crossed. This
one rule — same axis means co-vary, different axes means cross — is what makes
the algebra expressive.

---

## Axes

An `Axis` names a dimension and defines its **labels** — one per position along
that dimension. Labels can be explicit strings (or other hashable values) or
implicit integer indices:

```python
from pyens import Axis

# Explicit labels: useful for named categories (sites, experiments, species…)
sites = Axis("site", labels=["forest_A", "grassland_B", "wetland_C"])

# Integer labels: useful for anonymous sequences (ensemble members, draws…)
members = Axis("member", size=100)
```

An axis does two things:

1. **Specifies the unique set of values that dimension can take** — either the
   explicit labels you supply, or the integers `0, 1, …, size-1`.
2. **Provides a length constraint** — any field indexed along this axis must
   supply exactly one value per position.

The axis name appears in coordinates (the labels attached to each run's output)
and in diagnostic output from `describe()`. Choose names that are meaningful in
the context of your problem.

### Identity semantics

Two axes are the *same dimension* if and only if they are the same Python object.
Name equality is not sufficient:

```python
ax1 = Axis("site", labels=["A", "B", "C"])
ax2 = Axis("site", labels=["A", "B", "C"])  # identical definition, different object

ax1 is ax2   # False — these are two independent dimensions
```

This is intentional. It means you can have two dimensions named "site" without
them interfering (though this is not recommended). It also means that to make
two fields co-vary along the same
dimension, you must pass the *same* `Axis` object to both. The design makes
sharing explicit and accidental alignment impossible.

---

## Fields

A **field** describes one keyword argument to the model function — a single named
slot in the model's interface such as `parameters` or `forcing`. Every argument in
an `EnsembleSpec` is represented as a field. A field's role in the ensemble is
described by its *field spec* — an object that says whether and how that argument
varies across runs. PyEns provides two field spec types:

- **`Fixed`** — the value is the same for every run. A fixed field contributes no
  axis and adds no dimension to the ensemble.
- **`Grid`** — the value depends on one or more axes. A grid field defines a
  labeled lookup table: given the coordinates of a run, it returns the
  appropriate value.

### Fixed

`Fixed` wraps a value that is the same for every run. It contributes no axis
and adds no dimension to the ensemble.

```python
from pyens import Fixed

config = Fixed({"seed": 42, "precision": "float64"})
config.axes   # () — no axes; does not vary
```

Use `Fixed` for any input that serves as a shared constant — a configuration
object, a pre-loaded dataset, a scalar hyperparameter.

### Grid

`Grid` describes an input that varies. It defines a labeled sequence: each
position along the axis holds one value, and the axis's labels identify those
positions.

#### Positional sequence (ordered by axis)

The simplest form pairs values with axis positions by order: position 0 gets the
first value, position 1 gets the second, and so on.

```python
from pyens import Grid

sites = Axis("site", labels=["forest_A", "grassland_B", "wetland_C"])

# Position 0 → "forest_A", position 1 → "grassland_B", position 2 → "wetland_C"
climate_field = Grid(
    [climate_forest, climate_grassland, climate_wetland],
    along=sites,
)
```

The values can be anything — scalars, dicts, dataframes, model objects. PyEns
treats them as opaque; it only looks them up by position.

#### Label-keyed mapping (explicit, order-independent)

When an axis has explicit labels, you can supply values as a dict mapping labels
to values. This form is self-documenting and immune to ordering errors:

```python
climate_field = Grid(
    {
        "wetland_C":   climate_wetland,    # order does not matter
        "forest_A":    climate_forest,
        "grassland_B": climate_grassland,
    },
    along=sites,
)
```

Both forms produce identical grids. The dict form is recommended whenever
the axis has meaningful string labels, because it makes the assignment explicit
and PyEns will error if a label is missing or misspelled. It is only available
for axes created with `labels=[...]` (not `size=N`).

#### Multi-axis Grid

A `Grid` can vary along more than one axis. Values are then a nested sequence:
the outermost level indexes the first axis, the next level indexes the second, and
so on.

```python
members = Axis("member", size=4)

# Shape (3 sites, 4 members): ic_values[i][j] is the IC at site i, member j
ic_values = [
    [ic_fA_m0, ic_fA_m1, ic_fA_m2, ic_fA_m3],   # forest_A
    [ic_gB_m0, ic_gB_m1, ic_gB_m2, ic_gB_m3],   # grassland_B
    [ic_wC_m0, ic_wC_m1, ic_wC_m2, ic_wC_m3],   # wetland_C
]
ic_field = Grid(ic_values, along=[sites, members])
```

A multi-axis Grid is appropriate when an input is *jointly determined* by several
factors. An initial condition that depends on both the site and the ensemble member
is a natural example: you cannot describe it as a function of site alone or member
alone.

When axes have explicit labels, you can use mappings at any level of the nesting,
independently. The other three combinations for a two-axis grid are:

**Dict of lists** — outer (site) axis labeled, inner (member) axis positional:

```python
ic_field = Grid(
    {
        "wetland_C":   [ic_wC_m0, ic_wC_m1, ic_wC_m2, ic_wC_m3],  # order of
        "forest_A":    [ic_fA_m0, ic_fA_m1, ic_fA_m2, ic_fA_m3],  # keys does
        "grassland_B": [ic_gB_m0, ic_gB_m1, ic_gB_m2, ic_gB_m3],  # not matter
    },
    along=[sites, members],
)
```

**Dict of dicts** — both axes labeled:

```python
named_members = Axis("member", labels=["low", "mid", "high", "high2"])

ic_field = Grid(
    {
        "forest_A":    {"low": ic_fA_low, "mid": ic_fA_mid, "high": ic_fA_high, "high2": ic_fA_h2},
        "grassland_B": {"low": ic_gB_low, "mid": ic_gB_mid, "high": ic_gB_high, "high2": ic_gB_h2},
        "wetland_C":   {"low": ic_wC_low, "mid": ic_wC_mid, "high": ic_wC_high, "high2": ic_wC_h2},
    },
    along=[sites, named_members],
)
```

**List of dicts** — outer axis positional, inner axis labeled:

```python
ic_field = Grid(
    [
        {"low": ic_fA_low, "mid": ic_fA_mid, "high": ic_fA_high, "high2": ic_fA_h2},   # forest_A
        {"low": ic_gB_low, "mid": ic_gB_mid, "high": ic_gB_high, "high2": ic_gB_h2},   # grassland_B
        {"low": ic_wC_low, "mid": ic_wC_mid, "high": ic_wC_high, "high2": ic_wC_h2},   # wetland_C
    ],
    along=[sites, named_members],
)
```

All four forms produce identical results. Use whichever makes the assignment most
readable. The dict form at any level is only available when the corresponding axis
was created with `labels=[...]` (not `size=N`).

---

## EnsembleSpec

`EnsembleSpec` assembles named fields into a complete ensemble:

```python
from pyens import EnsembleSpec

spec = EnsembleSpec(inputs={
    "parameters": Fixed(theta),
    "forcing":    Grid(forcing_list, along=sites),
    "ic":         Grid(ic_list,      along=sites),
})
```

Useful properties and methods for inspection before running:

```python
spec.n_runs       # total number of model evaluations
spec.field_names  # ('parameters', 'forcing', 'ic')
spec.axes         # (Axis('site', labels=[...]),)
print(spec.describe())
# EnsembleSpec: 3 runs
#   Axes (1):
#     Axis('site', labels=['forest_A', 'grassland_B', 'wetland_C'])
#   Fields (3):
#     parameters: Fixed [fixed]
#     forcing:    Grid  [site]
#     ic:         Grid  [site]
```

To iterate over all runs:

```python
for inputs, coord in spec.iter_runs():
    output = my_model(**inputs)
    # coord = {'site': 'forest_A'}, {'site': 'grassland_B'}, …
```

---

## The zip and Cartesian product rule

How the spec produces run combinations depends entirely on which axes the fields
share. The rule:

:::{admonition} The axis identity rule
:class: important

Fields on the **same `Axis` instance** co-vary along that dimension (zip
semantics). Fields on **different `Axis` instances** are varied independently
(Cartesian product).
:::

Working through the cases concretely:

### Shared axis → zip

When two `Grid` fields use the same axis, the ensemble has one run per axis
position. Each run takes one value from each field at the *same* position.

```python
sites = Axis("site", labels=["A", "B", "C"])

spec = EnsembleSpec(inputs={
    "forcing": Grid([f_A, f_B, f_C], along=sites),
    "ic":      Grid([i_A, i_B, i_C], along=sites),  # same axis object
})
spec.n_runs  # 3
```

| Run | `forcing` | `ic`  | coordinate       |
|-----|-----------|-------|------------------|
| 0   | `f_A`     | `i_A` | `site = 'A'`     |
| 1   | `f_B`     | `i_B` | `site = 'B'`     |
| 2   | `f_C`     | `i_C` | `site = 'C'`     |

Each site's forcing is paired with that site's initial condition — the semantics
you want when the inputs are *paired by design*.

### Independent axes → Cartesian product

When two `Grid` fields use different axes, every combination of their values
is a distinct run.

```python
params  = Axis("param_set", size=5)
forcing_ax = Axis("forcing", size=3)

spec = EnsembleSpec(inputs={
    "parameters": Grid(param_list,   along=params),
    "forcing":    Grid(forcing_list, along=forcing_ax),
})
spec.n_runs  # 15 = 5 × 3
```

Use independent axes when the inputs are genuinely independent — when there is no
reason to pair a particular parameter set with a particular forcing condition.

### Mixed: some shared, some independent

Axes can be combined freely. A spec with two shared fields and one independent
field yields runs = (shared axis size) × (independent axis size):

```python
sites  = Axis("site",      labels=["A", "B", "C"])
params = Axis("param_set", size=5)

spec = EnsembleSpec(inputs={
    "forcing":    Grid(forcing_list, along=sites),   # paired with ic
    "ic":         Grid(ic_list,      along=sites),   # same sites axis
    "parameters": Grid(param_list,   along=params),  # independent
})
spec.n_runs  # 15 = 3 × 5
```

Each of the 15 runs gets one site's (forcing, ic) pair together with one
parameter set.

### Multi-axis Grid and shared axes

A multi-axis Grid can share axes with other fields. Shared axes do not introduce
new dimensions; they define how values are jointly indexed.

```python
sites  = Axis("site",      labels=["A", "B"])
params = Axis("param_set", size=3)

# IC depends on both site and param_set
ic_table = [[ic_A_p0, ic_A_p1, ic_A_p2],
            [ic_B_p0, ic_B_p1, ic_B_p2]]

spec = EnsembleSpec(inputs={
    "forcing":    Grid(forcing_list, along=sites),
    "ic":         Grid(ic_table,     along=[sites, params]),  # two shared axes
    "parameters": Grid(param_list,   along=params),
})
spec.n_runs  # 6 = 2 × 3  (no new axes introduced by the multi-axis Grid)
```

The `ic` field is resolved jointly by the site and param_set indices for each run,
giving the correct IC for that combination.

---

## Practical guidance

**Choose zip when inputs are paired by design.** If forcing conditions and initial
conditions were prepared together for each site, they belong on the same axis.
Crossing them creates combinations that have no physical meaning.

**Choose independent axes when inputs are genuinely independent.** A parameter
sweep that is unrelated to which forcing condition is used should use a separate
axis.

**Use a multi-axis Grid when an input is jointly determined.** If an initial
condition depends on both the parameter set and the site, specify it as
`Grid(table, along=[params, sites])` rather than computing the right value inside
the model.

**Validate before running.** Call `spec.describe()` and confirm that the axis
count and run count match your expectations before committing to a long run.

---

## Coordinates and selection

Every run in the ensemble has a **coordinate** — a dict mapping axis names to
the label of that axis for this run. Coordinates are produced automatically by
`iter_runs()` alongside the inputs:

```python
for inputs, coord in spec.iter_runs():
    print(coord)
# {'site': 'A', 'param_set': 0}
# {'site': 'A', 'param_set': 1}
# …
```

`Fixed` fields do not appear in the coordinate (they have no axis and do not
vary between runs).

Coordinates are designed to be used as result keys or metadata tags, so that
every output can be traced back to the inputs that produced it:

```python
results = {}
for inputs, coord in spec.iter_runs():
    key = tuple(coord.items())   # hashable key from coordinate
    results[key] = my_model(**inputs)
```

### Selecting a single run

`EnsembleSpec.sel()` retrieves the inputs for one specific run by label, without
iterating:

```python
inputs = spec.sel(site="B", param_set=2)
# Returns: {'forcing': f_B, 'ic': ic_B_p2, 'parameters': theta_2}
```

All axes must be specified. The return value is the same `inputs` dict that
`iter_runs()` would yield for that coordinate, making `sel()` useful for
spot-checking specs and for retrieving inputs for a specific run of interest.

---

## Partial application

`EnsembleSpec.freeze()` marks certain fields as free variables, returning a
**`BoundSpec`** callable. Calling it with values for the free fields produces a
fully-bound `EnsembleSpec`.

```python
base_spec = EnsembleSpec(inputs={
    "parameters": Fixed(default_theta),        # will be replaced
    "forcing":    Grid(forcing_list, along=sites),
    "ic":         Grid(ic_list,      along=sites),
})

param_map = base_spec.freeze(free=["parameters"])
```

`param_map` is then a reusable callable. Supplying a single value fixes
parameters for all runs:

```python
runnable = param_map(parameters=new_theta)
runnable.n_runs  # 3 (just the sites axis)
```

Supplying a `Grid` adds a new axis and produces a Cartesian product:

```python
p_axis = Axis("param_set", size=10)
runnable = param_map(parameters=Grid(proposed_params, along=p_axis))
runnable.n_runs  # 30 = 10 × 3
```

This pattern is particularly useful when an algorithm — an optimiser, a sampler,
a data assimilation routine — generates new parameter sets on each iteration. The
forcing and IC structure is defined once; only the parameter field changes:

```python
for iteration in range(n_iterations):
    proposed = algorithm.suggest(n=batch_size)
    batch_ax  = Axis("proposal", size=batch_size)
    runnable  = param_map(parameters=Grid(proposed, along=batch_ax))

    outputs = [(coord, my_model(**inputs)) for inputs, coord in runnable.iter_runs()]
    algorithm.update(outputs)
```

---

## Connection to xarray

PyEns's data model is closely analogous to [xarray](https://docs.xarray.dev)'s,
adapted to hold arbitrary Python objects rather than numeric arrays. Readers
familiar with xarray will find the following mapping useful; others can skip this
section.

| xarray concept | PyEns equivalent |
|---|---|
| `Dimension` | `Axis` |
| `Coordinate` (labels along a dimension) | `Axis.labels` |
| `DataArray` (labeled N-D numeric array) | `Grid` (labeled sequence of objects) |
| `Dataset` (collection of DataArrays) | `EnsembleSpec` (collection of fields) |
| Broadcasting / alignment rule | The axis identity rule |
| `.sel(x="a")` | `.sel(x="a")` — same name, analogous semantics |

The conceptual logic is the same: arrays that share a dimension are aligned along
it rather than crossed, and a Dataset collects multiple variables that share
coordinate systems.

**Key differences from xarray:**

*Alignment by object identity, not name.* In xarray, two DataArrays are aligned
along a dimension if they have the same dimension *name* (string). In PyEns,
alignment requires the same `Axis` *object*. This is a deliberate trade-off: name-
based alignment is more ergonomic but silently aligns dimensions that happen to share
a name. Identity-based alignment requires passing the same object around, but makes
sharing completely explicit and unambiguous.

*Arbitrary objects, not numeric arrays.* xarray stores numeric data and supports
arithmetic, broadcasting ufuncs, and statistical operations. PyEns stores arbitrary
Python objects — dicts, dataframes, model objects — and only ever looks them up by
position. No numeric operations are performed on the values themselves.

*No coordinate-based arithmetic.* xarray supports addition, subtraction, and
groupby operations aligned on coordinates. PyEns does not: it only enumerates
input combinations and retrieves values by coordinate.

*Lazy enumeration.* PyEns never materialises the full list of inputs in memory.
`iter_runs()` is a generator that produces one `(inputs, coordinate)` pair at a
time.
