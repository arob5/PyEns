"""Tests for pyens.runner: EnsembleRunner."""

from __future__ import annotations

import pytest

from pyens import Axis, EnsembleSpec, Fixed, Grid
from pyens.backends import LocalBackend, SequentialBackend
from pyens.result import EnsembleResult, RunRecord
from pyens.runner import EnsembleRunner


# ---------------------------------------------------------------------------
# Module-level model functions (must be at module scope for LocalBackend).
# ---------------------------------------------------------------------------

def _add(x, y):
    return x + y


def _sometimes_fails(x, y):
    if x == 99:
        raise RuntimeError("deliberate failure")
    return x + y


def _identity(**kwargs):
    return dict(kwargs)


# ---------------------------------------------------------------------------
# Basic correctness — SequentialBackend
# ---------------------------------------------------------------------------

class TestRunnerSequential:
    def setup_method(self):
        self.backend = SequentialBackend()

    def test_returns_ensemble_result(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert isinstance(result, EnsembleResult)

    def test_single_fixed_run(self):
        spec = EnsembleSpec(inputs={"x": Fixed(3), "y": Fixed(4)})
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.n_runs == 1
        assert result.outputs == [7]

    def test_grid_outputs_in_order(self):
        ax = Axis("x", size=4)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2, 3, 4], along=ax),
            "y": Fixed(10),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.outputs == [11, 12, 13, 14]

    def test_cartesian_product_run_count(self):
        ax1 = Axis("a", size=3)
        ax2 = Axis("b", size=4)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2, 3], along=ax1),
            "y": Grid([10, 20, 30, 40], along=ax2),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.n_runs == 12

    def test_coordinates_preserved(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2, 3], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.coordinates[0] == {"site": "A"}
        assert result.coordinates[1] == {"site": "B"}
        assert result.coordinates[2] == {"site": "C"}

    def test_empty_coordinate_for_fixed_only_spec(self):
        spec = EnsembleSpec(inputs={"x": Fixed(5), "y": Fixed(5)})
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.coordinates[0] == {}

    def test_failed_run_stored_not_raised(self):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 99, 3], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_sometimes_fails, self.backend)
        result = runner.run(spec)
        assert result.n_runs == 3
        assert result.n_failed == 1
        assert isinstance(result.failed[0].output, RuntimeError)
        assert result.outputs[0] == 1
        assert result.outputs[2] == 3

    def test_failed_run_coordinate_correct(self):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 99, 3], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_sometimes_fails, self.backend)
        result = runner.run(spec)
        assert result.failed[0].coordinate == {"x": 1}  # integer label, index 1

    def test_getitem_by_coordinate(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={
            "x": Grid([10, 20, 30], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result[{"site": "B"}].output == 20

    def test_all_inputs_forwarded(self):
        """Every field value is passed to the model under its field name."""
        ax = Axis("site", labels=["X"])
        spec = EnsembleSpec(inputs={
            "x": Grid([42], along=ax),
            "y": Fixed(99),
        })
        runner = EnsembleRunner(_identity, self.backend)
        result = runner.run(spec)
        assert result.outputs[0] == {"x": 42, "y": 99}

    def test_zip_semantics_preserved(self):
        """Shared-axis fields co-vary, not cross."""
        ax = Axis("site", labels=["A", "B"])
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2], along=ax),
            "y": Grid([10, 20], along=ax),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.n_runs == 2
        assert result.outputs == [11, 22]


# ---------------------------------------------------------------------------
# LocalBackend integration
# ---------------------------------------------------------------------------

class TestRunnerLocal:
    def setup_method(self):
        self.backend = LocalBackend(n_workers=2)

    def test_basic(self):
        ax = Axis("x", size=4)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2, 3, 4], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.outputs == [1, 2, 3, 4]

    def test_order_preserved(self):
        ax = Axis("x", size=6)
        spec = EnsembleSpec(inputs={
            "x": Grid(list(range(6)), along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_add, self.backend)
        result = runner.run(spec)
        assert result.outputs == list(range(6))

    def test_failed_run_stored_not_raised(self):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 99, 3], along=ax),
            "y": Fixed(0),
        })
        runner = EnsembleRunner(_sometimes_fails, self.backend)
        result = runner.run(spec)
        assert result.n_failed == 1
        assert isinstance(result.failed[0].output, RuntimeError)
