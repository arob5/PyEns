"""Tests for the batch protocol: pyens.backends._chunk, _batch and the worker."""

from __future__ import annotations

import io
import pickle

import pytest

from pyens.backends import RemoteError
from pyens.backends._batch import MARKER, BatchDir, split_evenly
from pyens.backends._chunk import (
    capture_call,
    decode_record,
    encode_failure,
    iter_decoded,
    read_frames,
    run_chunk,
    write_frame,
)
from pyens.backends._gridengine_worker import main as worker_main
from tests import _models


class TestRecords:
    def test_success_round_trip(self):
        assert decode_record(capture_call(_models.add, 7, {"x": 1, "y": 2})) == (7, 3)

    def test_failure_round_trip_with_cause(self):
        index, err = decode_record(capture_call(_models.always_raises, 3, {"x": 0}))
        assert index == 3
        assert type(err) is RuntimeError
        assert "always_raises" in str(err.__cause__)

    def test_kw_only_exception_becomes_remote_error(self):
        _, err = decode_record(capture_call(_models.kw_only_crash, 0, {"x": 1}))
        assert isinstance(err, RemoteError)
        assert err.attributes["returncode"] == 1

    def test_unreadable_payload_isolated_to_its_run(self):
        record = pickle.dumps((5, True, None, b"not a pickle"))
        index, err = decode_record(record)
        assert index == 5
        assert isinstance(err, RemoteError)

    def test_encode_failure_without_traceback(self):
        _, err = decode_record(encode_failure(0, ValueError("x")))
        assert isinstance(err, ValueError)


class TestFrames:
    def _write(self, tmp_path, records, extra=b""):
        path = tmp_path / "frames"
        buf = io.BytesIO()
        for record in records:
            write_frame(buf, record)
        path.write_bytes(buf.getvalue() + extra)
        return path

    def test_complete_file(self, tmp_path):
        path = self._write(tmp_path, [b"a", b"bcd", b""])
        assert read_frames(path) == ([b"a", b"bcd", b""], True)

    def test_truncated_header(self, tmp_path):
        path = self._write(tmp_path, [b"a"], extra=b"\x00\x00")
        assert read_frames(path) == ([b"a"], False)

    def test_truncated_body(self, tmp_path):
        path = self._write(tmp_path, [b"a"], extra=b"\x00" * 7 + b"\x09abc")
        assert read_frames(path) == ([b"a"], False)

    def test_iter_decoded_skips_corrupt_frames(self):
        good = capture_call(_models.add, 0, {"x": 1, "y": 1})
        assert list(iter_decoded([b"garbage", good])) == [(0, 2)]


class TestRunChunk:
    def test_serial_in_order(self):
        records = []
        run_chunk(_models.add, 10, [{"x": i, "y": 0} for i in range(3)], records.append)
        assert [decode_record(r) for r in records] == [(10, 0), (11, 1), (12, 2)]

    def test_processes(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(pickle.dumps(_models.sometimes_fails))
        records = []
        runs = [{"x": i - 1, "y": 0} for i in range(4)]
        run_chunk(None, 0, runs, records.append, processes=2, model_path=model_path)
        decoded = dict(decode_record(r) for r in records)
        assert isinstance(decoded[0], ValueError)
        assert [decoded[i] for i in (1, 2, 3)] == [0, 1, 2]

    def test_requires_fn_or_model_path(self):
        with pytest.raises(ValueError):
            run_chunk(None, 0, [{}], lambda r: None)
        with pytest.raises(ValueError):
            run_chunk(_models.add, 0, [{}], lambda r: None, processes=2)


class TestBatchDir:
    @pytest.mark.parametrize("n, k, expected", [
        (7, 3, [(0, 3), (3, 5), (5, 7)]),
        (6, 3, [(0, 2), (2, 4), (4, 6)]),
        (2, 5, [(0, 1), (1, 2)]),
        (5, 1, [(0, 5)]),
        (0, 3, []),
    ])
    def test_split_evenly(self, n, k, expected):
        assert split_evenly(n, k) == expected

    def test_create_layout(self, tmp_path):
        batch = BatchDir.create(tmp_path / "work", "eki")
        assert batch.root.parent == tmp_path / "work"
        assert batch.root.name.startswith("eki-")
        for sub in ("inputs", "results", "logs", MARKER):
            assert (batch.root / sub).exists()

    def test_finished_tasks(self, tmp_path):
        batch = BatchDir.create(tmp_path, "x")
        for name in ("task-1.pkl", "task-2.partial", "task-3.error",
                     "task-4.error.tmp", "task-x.pkl", "other.pkl"):
            (batch.results_dir / name).write_bytes(b"")
        assert batch.finished_tasks() == {1, 3}

    def test_remove_requires_marker(self, tmp_path):
        batch = BatchDir.create(tmp_path, "x")
        (batch.root / MARKER).unlink()
        batch.remove()
        assert batch.root.exists()
        (batch.root / MARKER).write_text("")
        batch.remove()
        assert not batch.root.exists()

    def test_inputs_round_trip(self, tmp_path):
        batch = BatchDir.create(tmp_path, "x")
        batch.write_inputs(2, 5, [{"a": 1}, {"a": 2}])
        assert batch.load_inputs(2) == (5, [{"a": 1}, {"a": 2}])


class TestWorker:
    def _batch(self, tmp_path, fn, runs):
        batch = BatchDir.create(tmp_path, "x")
        batch.write_model(fn)
        batch.write_inputs(1, 0, runs)
        batch.write_manifest({"processes": 1, "tasks": [[0, len(runs)]]})
        return batch

    def test_writes_final_result_file(self, tmp_path):
        batch = self._batch(tmp_path, _models.add, [{"x": 1, "y": 2}])
        assert worker_main([str(batch.root), "1"]) == 0
        assert not batch.partial_path(1).exists()
        frames, complete = read_frames(batch.result_path(1))
        assert complete and [decode_record(f) for f in frames] == [(0, 3)]

    def test_setup_failure_writes_error_file(self, tmp_path):
        batch = self._batch(tmp_path, _models.BrokenOnLoad(), [{"x": 1}])
        assert worker_main([str(batch.root), "1"]) == 1
        exc, tb = batch.read_task_error(1)
        assert isinstance(exc, ImportError)
        assert "simulated missing dependency" in tb

    def test_bad_arguments(self, tmp_path):
        assert worker_main([]) == 2
        assert worker_main([str(tmp_path), "one"]) == 2
