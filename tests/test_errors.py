"""Tests for pyens.backends.errors: portable exception capture."""

from __future__ import annotations

import pickle
import threading

from pyens.backends.errors import (
    GridEngineError,
    RemoteError,
    TaskFailedError,
    attach_remote_traceback,
    format_traceback,
    portable_exception,
)
from tests._models import KwOnlyError


def _raised(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as caught:
        return caught


class _UnpicklableAttrError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.lock = threading.Lock()
        self.code = 7


class TestPortableException:
    def test_round_trippable_exception_returned_unchanged(self):
        exc = _raised(ValueError("fine"))
        assert portable_exception(exc) is exc

    def test_kw_only_exception_becomes_remote_error(self):
        exc = _raised(KwOnlyError("boom", returncode=3, stderr="bad param"))
        out = portable_exception(exc)
        assert isinstance(out, RemoteError)
        assert out.type_name == "tests._models.KwOnlyError"
        assert out.message == "boom"
        assert out.attributes == {"returncode": 3, "stderr": "bad param"}
        assert "KwOnlyError: boom" in out.traceback
        assert "_raised" in out.traceback

    def test_unpicklable_attributes_are_dropped(self):
        out = portable_exception(_raised(_UnpicklableAttrError("x")))
        assert isinstance(out, RemoteError)
        assert out.attributes == {"code": 7}

    def test_explicit_traceback_text_used(self):
        out = portable_exception(KwOnlyError("x", returncode=1, stderr=""), "TB")
        assert out.traceback == "TB"

    def test_builtin_type_name_unqualified(self):
        class Local(Exception):
            def __init__(self, *, a):
                super().__init__()
        out = portable_exception(Local(a=1))
        assert out.type_name.endswith("Local")


class TestRemoteError:
    def test_pickle_round_trip(self):
        err = RemoteError("pkg.Err", "msg", "tb", {"a": 1})
        clone = pickle.loads(pickle.dumps(err))
        assert (clone.type_name, clone.message, clone.traceback, clone.attributes) == (
            "pkg.Err", "msg", "tb", {"a": 1}
        )

    def test_str_and_repr(self):
        err = RemoteError("pkg.Err", "msg")
        assert str(err) == "pkg.Err: msg"
        assert repr(err) == "RemoteError('pkg.Err', 'msg')"

    def test_is_exception(self):
        assert isinstance(RemoteError("a", "b"), Exception)


class TestTaskFailedError:
    def test_pickle_round_trip(self):
        err = TaskFailedError("died", "killed", job_id="12", task_id=3, log_path="/l")
        clone = pickle.loads(pickle.dumps(err))
        assert (clone.kind, clone.reason, clone.job_id, clone.task_id, clone.log_path) == (
            "died", "killed", "12", 3, "/l"
        )

    def test_str_mentions_task_job_and_log(self):
        err = TaskFailedError("timeout", "expired", job_id="12", task_id=3, log_path="/l")
        text = str(err)
        assert "task 3 of job 12" in text
        assert "timeout" in text
        assert "/l" in text

    def test_str_without_ids(self):
        assert str(TaskFailedError("died", "gone")) == "task failed (died): gone"


class TestHelpers:
    def test_attach_remote_traceback_sets_cause(self):
        exc = ValueError("x")
        assert attach_remote_traceback(exc, "remote tb") is exc
        assert "remote tb" in str(exc.__cause__)

    def test_attach_remote_traceback_skips_remote_error(self):
        err = RemoteError("a", "b", "tb")
        attach_remote_traceback(err, "other")
        assert err.__cause__ is None

    def test_format_traceback(self):
        assert "ValueError: x" in format_traceback(_raised(ValueError("x")))

    def test_grid_engine_error_is_runtime_error(self):
        assert issubclass(GridEngineError, RuntimeError)
