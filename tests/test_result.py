"""Tests for pyens.result: RunRecord and EnsembleResult."""

from __future__ import annotations

import pytest

from pyens.result import EnsembleResult, RunRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(entries: list[tuple]) -> EnsembleResult:
    """Build an EnsembleResult from (coordinate, output) pairs."""
    return EnsembleResult([RunRecord(coordinate=c, output=o) for c, o in entries])


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

class TestRunRecord:
    def test_not_failed_for_value(self):
        rec = RunRecord(coordinate={"site": "A"}, output=42)
        assert not rec.failed

    def test_not_failed_for_none(self):
        rec = RunRecord(coordinate={}, output=None)
        assert not rec.failed

    def test_failed_for_exception(self):
        rec = RunRecord(coordinate={"site": "A"}, output=ValueError("bad"))
        assert rec.failed

    def test_failed_for_base_exception(self):
        rec = RunRecord(coordinate={}, output=RuntimeError("crash"))
        assert rec.failed

    def test_output_accessible(self):
        rec = RunRecord(coordinate={"x": 1}, output="hello")
        assert rec.output == "hello"

    def test_coordinate_accessible(self):
        coord = {"site": "A", "member": 3}
        rec = RunRecord(coordinate=coord, output=0)
        assert rec.coordinate == coord

    def test_frozen(self):
        rec = RunRecord(coordinate={}, output=1)
        with pytest.raises((AttributeError, TypeError)):
            rec.output = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EnsembleResult — counts
# ---------------------------------------------------------------------------

class TestEnsembleResultCounts:
    def setup_method(self):
        err = RuntimeError("fail")
        self.result = _make_result([
            ({"site": "A"}, 10),
            ({"site": "B"}, err),
            ({"site": "C"}, 30),
        ])

    def test_n_runs(self):
        assert self.result.n_runs == 3

    def test_n_failed(self):
        assert self.result.n_failed == 1

    def test_len(self):
        assert len(self.result) == 3

    def test_all_succeed(self):
        r = _make_result([({"x": i}, i * 2) for i in range(5)])
        assert r.n_failed == 0

    def test_all_fail(self):
        r = _make_result([({"x": i}, ValueError()) for i in range(3)])
        assert r.n_failed == 3


# ---------------------------------------------------------------------------
# EnsembleResult — bulk accessors
# ---------------------------------------------------------------------------

class TestEnsembleResultAccessors:
    def setup_method(self):
        self.err = RuntimeError("fail")
        self.result = _make_result([
            ({"site": "A"}, 10),
            ({"site": "B"}, self.err),
            ({"site": "C"}, 30),
        ])

    def test_outputs_in_order(self):
        outs = self.result.outputs
        assert outs[0] == 10
        assert outs[1] is self.err
        assert outs[2] == 30

    def test_outputs_length(self):
        assert len(self.result.outputs) == 3

    def test_coordinates_in_order(self):
        coords = self.result.coordinates
        assert coords[0] == {"site": "A"}
        assert coords[1] == {"site": "B"}
        assert coords[2] == {"site": "C"}

    def test_succeeded_filters(self):
        succeeded = self.result.succeeded
        assert len(succeeded) == 2
        assert all(not r.failed for r in succeeded)

    def test_succeeded_outputs(self):
        outputs = [r.output for r in self.result.succeeded]
        assert outputs == [10, 30]

    def test_failed_filters(self):
        failed = self.result.failed
        assert len(failed) == 1
        assert failed[0].coordinate == {"site": "B"}
        assert failed[0].output is self.err

    def test_empty_result(self):
        r = _make_result([])
        assert r.outputs == []
        assert r.coordinates == []
        assert r.succeeded == []
        assert r.failed == []


# ---------------------------------------------------------------------------
# EnsembleResult — iteration
# ---------------------------------------------------------------------------

class TestEnsembleResultIteration:
    def test_iter_yields_records(self):
        result = _make_result([({"x": i}, i) for i in range(4)])
        records = list(result)
        assert len(records) == 4
        assert all(isinstance(r, RunRecord) for r in records)

    def test_iter_order_matches_outputs(self):
        result = _make_result([({"x": i}, i * 10) for i in range(3)])
        iterated = [r.output for r in result]
        assert iterated == result.outputs


# ---------------------------------------------------------------------------
# EnsembleResult — __getitem__
# ---------------------------------------------------------------------------

class TestEnsembleResultGetItem:
    def setup_method(self):
        self.result = _make_result([
            ({"site": "A", "member": 0}, 100),
            ({"site": "A", "member": 1}, 101),
            ({"site": "B", "member": 0}, 200),
        ])

    def test_lookup_by_coord(self):
        rec = self.result[{"site": "A", "member": 1}]
        assert rec.output == 101

    def test_lookup_first(self):
        rec = self.result[{"site": "A", "member": 0}]
        assert rec.output == 100

    def test_lookup_last(self):
        rec = self.result[{"site": "B", "member": 0}]
        assert rec.output == 200

    def test_unknown_coord_raises_key_error(self):
        with pytest.raises(KeyError):
            self.result[{"site": "Z", "member": 0}]

    def test_empty_coord_lookup(self):
        result = _make_result([({}, 42)])
        rec = result[{}]
        assert rec.output == 42


# ---------------------------------------------------------------------------
# EnsembleResult — repr
# ---------------------------------------------------------------------------

class TestEnsembleResultRepr:
    def test_repr_contains_counts(self):
        result = _make_result([({"x": 0}, 1), ({"x": 1}, ValueError())])
        r = repr(result)
        assert "2" in r
        assert "1" in r
