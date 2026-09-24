"""Guarantees every backend must honor, run against each backend.

The Grid Engine backend runs against the fake scheduler in
``tests/fake_ge.py``.
"""

from __future__ import annotations

import pytest

from pyens import Axis, EnsembleSpec, Fixed, Grid
from pyens.backends import (
    Backend,
    GridEngineBackend,
    LocalBackend,
    RemoteError,
    SequentialBackend,
)
from pyens.runner import EnsembleRunner
from tests import _models


@pytest.fixture(params=["sequential", "local", "gridengine"])
def backend(request: pytest.FixtureRequest) -> Backend:
    if request.param == "sequential":
        return SequentialBackend()
    if request.param == "local":
        return LocalBackend(n_workers=2)
    fake = request.getfixturevalue("fake_ge")
    return GridEngineBackend(
        walltime="00:05:00", work_dir=fake.work_dir, n_jobs=3, poll_interval=0.05,
    )


def test_basic(backend):
    assert backend.map(_models.add, [{"x": 1, "y": 2}, {"x": 3, "y": 4}]) == [3, 7]


def test_empty(backend):
    assert backend.map(_models.add, []) == []


def test_order_preserved(backend):
    runs = [{"x": i, "y": 0} for i in range(10)]
    assert backend.map(_models.add, runs) == list(range(10))


def test_accepts_generator(backend):
    assert backend.map(_models.add, ({"x": i, "y": 1} for i in range(3))) == [1, 2, 3]


def test_exceptions_stored_in_place(backend):
    runs = [{"x": -1, "y": 0}, {"x": 5, "y": 0}, {"x": -2, "y": 0}]
    results = backend.map(_models.sometimes_fails, runs)
    assert isinstance(results[0], ValueError)
    assert results[1] == 5
    assert isinstance(results[2], ValueError)
    assert "negative x: -2" in str(results[2])


def test_all_fail(backend):
    results = backend.map(_models.always_raises, [{"x": i} for i in range(4)])
    assert all(isinstance(r, RuntimeError) for r in results)


def test_unpicklable_exception_isolated(backend):
    """The pySIPNET SIPNETRunError case: later runs must be unaffected."""
    results = backend.map(_models.kw_only_crash, [{"x": i} for i in range(5)])
    assert results[0] == 0
    assert results[2:] == [2, 3, 4]
    assert isinstance(results[1], (_models.KwOnlyError, RemoteError))
    assert "sipnet exited 1" in str(results[1])


def test_runner_integration(backend):
    sites = Axis("site", labels=["A", "B", "C"])
    spec = EnsembleSpec(inputs={
        "x": Grid([1, 2, 3], along=sites),
        "y": Fixed(10),
    })
    result = EnsembleRunner(_models.add, backend).run(spec)
    assert result.outputs == [11, 12, 13]
    assert result[{"site": "B"}].output == 12
