# Documentation and Writing Conventions

This file records project-wide conventions for writing documentation and
user-facing text. Check it before writing or editing any documentation.

---

## Name usage

| Context | Form | Example |
|---|---|---|
| Prose, page titles, headings | **PyEns** | "PyEns is a Python library for…" |
| Python code, imports, package references | `pyens` | `import pyens`, `pip install pyens` |
| File names, directory names | `pyens` | `pyens/spec.py` |

The distinction mirrors established conventions in the scientific Python ecosystem
(NumPy / `numpy`, SciPy / `scipy`, etc.). Never write "pyENS", "PyENS", or "Pyens".

---

## Terminology

**Initial condition (IC):** Many scientific models are *dynamical* — they simulate
how a system evolves over time starting from some initial state. That initial state is
called an **initial condition**, abbreviated **IC**. The concept is domain-agnostic:
it appears in climate models, ecosystem models, epidemiological models, fluid
dynamics simulations, and many others. Whenever documentation examples use a field
named `ic`, introduce the term on first use with a brief explanation.

**Ensemble member:** One realisation in an ensemble — one combination of inputs that
the model is evaluated at. Do not use "sample" (confusable with statistical sampling),
"trial" (confusable with Optuna/Ax), or "job" (an execution concept, not an input
concept).

**Run:** One execution of the model callable. Equivalent to "ensemble member" in most
contexts. Prefer "run" when discussing execution; prefer "member" when discussing the
input structure.

**Axis:** A named dimension of the ensemble input space. Not to be confused with a
plotting axis. Always capitalise as a Python class name (`Axis`) when referring to
the code object; use lowercase when speaking abstractly ("the site axis").

**Field:** One named input to the model. Use lowercase ("the parameters field", "a
fixed field").

---

## Code examples

- Use short, concrete variable names that reflect the domain (`sites`, `members`,
  `params`) rather than abstract names (`ax1`, `v1`).
- When an example uses `ic`, add a parenthetical or footnote on first use:
  "initial condition (IC)".
- Prefer examples that work without external data — use simple Python literals
  (floats, strings, dicts) rather than assuming a dataset is loaded.
- Code blocks in Markdown docs use triple-backtick fences with `python` syntax tag.

---

## Tone and register

- Write for a technically literate reader who is not necessarily a software engineer
  and may not be an ecologist or earth scientist.
- Explain domain-specific terms (IC, site, forcing data) briefly on first use.
- Avoid unnecessary hedging ("it might be useful to…", "you could consider…").
  State things directly.
- Use active voice and second person ("you can…", "call `describe()`") rather than
  passive ("it can be seen that…", "`describe()` may be called").
