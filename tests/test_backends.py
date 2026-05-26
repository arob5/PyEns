"""Tests for pyens.backends: SequentialBackend and LocalBackend."""

from __future__ import annotations

import pytest

from pyens.backends import LocalBackend, SequentialBackend


# ---------------------------------------------------------------------------
# Module-level helpers — must be at module scope for LocalBackend pickling.
# ---------------------------------------------------------------------------

def _add(x, y):
    return x + y


def _sometimes_fails(x, y):
    if x < 0:
        raise ValueError(f"negative x: {x}")
    return x + y


def _always_raises(x):
    raise RuntimeError("deliberate failure")


# ---------------------------------------------------------------------------
# SequentialBackend
# ---------------------------------------------------------------------------

class TestSequentialBackend:
    def setup_method(self):
        self.backend = SequentialBackend()

    def test_basic(self):
        runs = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        assert self.backend.map(_add, runs) == [3, 7]

    def test_empty_runs(self):
        assert self.backend.map(_add, []) == []

    def test_order_preserved(self):
        runs = [{"x": i, "y": 0} for i in range(10)]
        results = self.backend.map(_add, runs)
        assert results == list(range(10))

    def test_exception_stored_not_raised(self):
        runs = [{"x": -1, "y": 0}, {"x": 2, "y": 0}]
        results = self.backend.map(_sometimes_fails, runs)
        assert isinstance(results[0], ValueError)
        assert results[1] == 2

    def test_all_fail(self):
        runs = [{"x": i} for i in range(3)]
        results = self.backend.map(_always_raises, runs)
        assert all(isinstance(r, RuntimeError) for r in results)

    def test_exception_does_not_stop_subsequent_runs(self):
        """A failing run does not prevent later runs from executing."""
        runs = [{"x": -1, "y": 0}, {"x": -2, "y": 0}, {"x": 5, "y": 0}]
        results = self.backend.map(_sometimes_fails, runs)
        assert isinstance(results[0], ValueError)
        assert isinstance(results[1], ValueError)
        assert results[2] == 5

    def test_single_run(self):
        assert self.backend.map(_add, [{"x": 7, "y": 3}]) == [10]

    def test_accepts_generator(self):
        """map() should accept any iterable, not just lists."""
        runs = ({"x": i, "y": 1} for i in range(3))
        results = self.backend.map(_add, runs)
        assert results == [1, 2, 3]


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------
#
# NOTE: LocalBackend uses ProcessPoolExecutor. All functions and values must
# be picklable (defined at module level, not as lambdas or closures).
# These tests use module-level helpers for that reason.

class TestLocalBackend:
    def setup_method(self):
        self.backend = LocalBackend(n_workers=2)

    def test_basic(self):
        runs = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        assert self.backend.map(_add, runs) == [3, 7]

    def test_empty_runs(self):
        assert self.backend.map(_add, []) == []

    def test_order_preserved(self):
        """Results must be in submission order regardless of completion order."""
        runs = [{"x": i, "y": 0} for i in range(8)]
        results = self.backend.map(_add, runs)
        assert results == list(range(8))

    def test_exception_stored_not_raised(self):
        runs = [{"x": -1, "y": 0}, {"x": 2, "y": 0}]
        results = self.backend.map(_sometimes_fails, runs)
        assert isinstance(results[0], ValueError)
        assert results[1] == 2

    def test_exception_does_not_stop_subsequent_runs(self):
        runs = [{"x": -1, "y": 0}, {"x": 5, "y": 0}, {"x": -2, "y": 0}]
        results = self.backend.map(_sometimes_fails, runs)
        assert isinstance(results[0], ValueError)
        assert results[1] == 5
        assert isinstance(results[2], ValueError)

    def test_default_workers(self):
        """LocalBackend(n_workers=None) should work without error."""
        backend = LocalBackend()
        runs = [{"x": 1, "y": 1}]
        assert backend.map(_add, runs) == [2]
