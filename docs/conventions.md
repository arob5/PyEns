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

**Field / argument:** A field describes one keyword argument to the model function —
one named slot in the model's interface. Use "field" when discussing the PyEns data
model; "argument" when clarifying what a field represents in Python terms. Never use
"input" in the singular to mean a field. Use lowercase ("the parameters field", "a
fixed field").

**Inputs:** Always means the complete dict of field values for one run — what
``iter_runs()`` yields as its first element and what gets passed to the model as
``my_model(**inputs)``. Never use "inputs" to mean a single field.

---

## Code examples

- Use short, concrete variable names that reflect the domain (`sites`, `members`,
  `params`) rather than abstract names (`ax1`, `v1`).
- Introduce any domain-specific terms (IC, forcing data, ensemble member) with a
  brief parenthetical on first use.
- Prefer examples that work without external data — use simple Python literals
  (floats, strings, dicts) rather than assuming a dataset is loaded.
- Code blocks in Markdown docs use triple-backtick fences with `python` syntax tag.

---

## Tone and register

- Write for a technically literate reader who is not necessarily a software engineer
  and may not be an ecologist or earth scientist.
- If domain-specific examples are used in documentation, explain any domain-specific terms briefly on first use.
- Avoid unnecessary hedging ("it might be useful to…", "you could consider…").
  State things directly.
- Use active voice and second person ("you can…", "call `describe()`") rather than
  passive ("it can be seen that…", "`describe()` may be called").
