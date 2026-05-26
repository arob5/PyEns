# pyens

**pyens** is a Python library for specifying and executing structured ensemble model runs. It is designed for scientific workflows where the same model must be evaluated at many different combinations of inputs — different parameters, different forcing data, different sites — and where keeping track of which output came from which input combination is important.

The library separates three concerns:

- **Specification** — a concise, validated description of what combinations to run.
- **Execution** — a pluggable backend that runs those combinations locally or on an HPC cluster.
- **Results** — a structured output container that preserves the coordinate labels from the specification.

This documentation covers the core concepts. If you are new to pyens, start with the [Data Model](user_guide/data_model.md) page.

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/data_model
```
