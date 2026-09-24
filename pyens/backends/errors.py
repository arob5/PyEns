"""Exceptions shared by the execution backends.

A backend that runs the model in another process has to send each run's
exception back to the driver through :mod:`pickle`. Not every exception
survives that trip: an exception class whose ``__init__`` takes required
keyword-only arguments pickles without complaint but fails to unpickle,
because :class:`BaseException` reconstructs itself as ``cls(*self.args)``.
The helpers here detect that case in the worker and substitute a
:class:`RemoteError` that carries the same diagnostic information.
"""

from __future__ import annotations

import pickle
import traceback as _traceback
from typing import Any, Literal

TaskFailureKind = Literal["died", "error_state", "timeout", "worker_error"]


class RemoteError(Exception):
    """Stand-in for a model exception that could not cross a process boundary.

    Backends return a ``RemoteError`` in a run's result slot when the model
    raised an exception that cannot be pickled and unpickled intact. It keeps
    everything needed to diagnose the failure: the original exception's fully
    qualified type name, its message, the formatted traceback from the worker
    process, and any of its instance attributes that are themselves
    picklable.

    Because the original class is not reconstructed, ``isinstance`` checks
    against it fail. Check :attr:`type_name` instead, or make the original
    class picklable (for example by defining ``__reduce__``).

    Args:
        type_name: Fully qualified name of the original exception type, e.g.
            ``"pysipnet.runner.SIPNETRunError"``.
        message: ``str()`` of the original exception.
        traceback: The formatted traceback from the process where the
            exception was raised. Empty if unavailable.
        attributes: The original exception's instance attributes that
            survived a pickle round trip.

    Examples:
        >>> err = RemoteError("mypkg.RunError", "exit status 1",
        ...                   attributes={"returncode": 1})
        >>> str(err)
        'mypkg.RunError: exit status 1'
        >>> err.attributes["returncode"]
        1
    """

    def __init__(
        self,
        type_name: str,
        message: str,
        traceback: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        attributes = dict(attributes or {})
        super().__init__(type_name, message, traceback, attributes)
        self.type_name = type_name
        self.message = message
        self.traceback = traceback
        self.attributes = attributes

    @classmethod
    def from_exception(
        cls, exc: BaseException, traceback: str | None = None
    ) -> RemoteError:
        """Build a ``RemoteError`` describing *exc*.

        Args:
            exc: The exception to describe.
            traceback: Pre-formatted traceback text. If omitted, it is
                formatted from ``exc.__traceback__``.

        Returns:
            A picklable ``RemoteError`` with the type name, message,
            traceback and picklable attributes of *exc*.
        """
        if traceback is None:
            traceback = format_traceback(exc)
        attributes: dict[str, Any] = {}
        for name, value in getattr(exc, "__dict__", {}).items():
            if name.startswith("__"):
                continue
            if _round_trips(value):
                attributes[name] = value
        return cls(_qualified_name(type(exc)), _safe_str(exc), traceback, attributes)

    def __str__(self) -> str:
        return f"{self.type_name}: {self.message}"

    def __repr__(self) -> str:
        return f"RemoteError({self.type_name!r}, {self.message!r})"


class TaskFailedError(Exception):
    """A run that never produced a result because its batch task failed.

    Cluster backends group runs into tasks (for example Grid Engine array
    tasks). When a task stops without reporting a result for some of its
    runs, each of those runs gets a ``TaskFailedError`` in its result slot.
    Runs the task finished before it stopped keep their real results.

    Args:
        kind: What went wrong, one of:

            - ``"died"``: the task left the queue without writing a result
              for the run (killed at its run-time limit, out of memory, node
              failure, cancelled outside PyEns).
            - ``"error_state"``: the scheduler put the task in an error state
              (Grid Engine ``Eqw``), so it never started. PyEns deletes such
              tasks.
            - ``"timeout"``: the backend's overall ``timeout`` expired before
              the task finished.
            - ``"worker_error"``: the worker process started but failed
              before running the model, e.g. it could not import the model's
              module. The cause is attached as ``__cause__``.
        reason: A human-readable explanation, including any scheduler
            accounting details available.
        job_id: The scheduler's job ID, if known.
        task_id: The task number within the job, if known.
        log_path: Path of the task's combined stdout/stderr log, if any.

    Examples:
        >>> err = TaskFailedError("timeout", "backend timeout expired",
        ...                       job_id="123", task_id=4)
        >>> err.kind
        'timeout'
    """

    def __init__(
        self,
        kind: TaskFailureKind,
        reason: str,
        job_id: str | None = None,
        task_id: int | None = None,
        log_path: str | None = None,
    ) -> None:
        super().__init__(kind, reason, job_id, task_id, log_path)
        self.kind = kind
        self.reason = reason
        self.job_id = job_id
        self.task_id = task_id
        self.log_path = log_path

    def __str__(self) -> str:
        where = "task"
        if self.task_id is not None:
            where = f"task {self.task_id}"
        if self.job_id is not None:
            where += f" of job {self.job_id}"
        text = f"{where} failed ({self.kind}): {self.reason}"
        if self.log_path:
            text += f" [log: {self.log_path}]"
        return text

    def __repr__(self) -> str:
        return (
            f"TaskFailedError({self.kind!r}, job_id={self.job_id!r}, "
            f"task_id={self.task_id!r})"
        )


class GridEngineError(RuntimeError):
    """A Grid Engine command failed in a way that prevents running the ensemble.

    Raised by :class:`~pyens.backends.GridEngineBackend` when submission
    fails (for example ``qsub`` rejects a directive), in which case nothing
    is running. The one exception is a ``qsub`` that does not return in
    time: the job may have been queued anyway, so the message says to check
    ``qstat`` and the batch directory is kept. Failures of individual runs
    or tasks are never raised; they are returned in the result slots.
    """


class _RemoteTraceback(Exception):
    """Carries a worker's formatted traceback as an exception's ``__cause__``.

    Mirrors :mod:`concurrent.futures`, so a re-raised remote exception prints
    the traceback from the process where it happened.
    """

    def __init__(self, tb: str) -> None:
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return f'\n"""\n{self.tb}"""'


def portable_exception(
    exc: BaseException, traceback: str | None = None
) -> BaseException:
    """Return *exc* if it survives a pickle round trip, else a ``RemoteError``.

    Call this in a worker process before sending an exception to the driver.

    Args:
        exc: The exception raised by the model.
        traceback: Pre-formatted traceback text for *exc*, used if a
            ``RemoteError`` is built. Formatted from *exc* if omitted.

    Returns:
        *exc* itself if ``pickle.loads(pickle.dumps(exc))`` succeeds and
        yields the same type; otherwise a :class:`RemoteError` describing it.

    Examples:
        >>> class KwOnly(Exception):
        ...     def __init__(self, msg, *, code):
        ...         super().__init__(msg)
        ...         self.code = code
        >>> err = portable_exception(KwOnly("boom", code=3))
        >>> type(err).__name__, err.attributes
        ('RemoteError', {'code': 3})
        >>> portable_exception(ValueError("fine"))
        ValueError('fine')
    """
    try:
        clone = pickle.loads(pickle.dumps(exc, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pass
    else:
        if type(clone) is type(exc):
            return exc
    return RemoteError.from_exception(exc, traceback)


def attach_remote_traceback(exc: BaseException, traceback: str) -> BaseException:
    """Set *exc*'s ``__cause__`` to the worker traceback text, then return it.

    Args:
        exc: An exception received from a worker process.
        traceback: The formatted traceback captured in the worker.

    Returns:
        *exc*, modified in place. ``RemoteError`` instances already carry the
        traceback and are returned unchanged.
    """
    if traceback and not isinstance(exc, RemoteError):
        exc.__cause__ = _RemoteTraceback(traceback)
    return exc


def format_traceback(exc: BaseException) -> str:
    """Format *exc* and its traceback as text, never raising."""
    try:
        return "".join(_traceback.format_exception(exc))
    except Exception:
        return f"{_qualified_name(type(exc))}: <traceback unavailable>\n"


def _qualified_name(cls: type) -> str:
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", cls.__name__)
    if module in (None, "builtins"):
        return qualname
    return f"{module}.{qualname}"


def _safe_str(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return f"<str() of {_qualified_name(type(exc))} failed>"


def _round_trips(value: Any) -> bool:
    try:
        pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return False
    return True
