"""Grid Engine (SGE/OGS/UGE) array-job execution backend."""

from __future__ import annotations

import contextlib
import functools
import getpass
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pyens.backends._batch import FORMAT_VERSION, BatchDir, split_evenly
from pyens.backends._chunk import iter_decoded, read_frames
from pyens.backends.base import Backend
from pyens.backends.errors import (
    GridEngineError,
    TaskFailedError,
    TaskFailureKind,
    attach_remote_traceback,
)

logger = logging.getLogger(__name__)

KeepPolicy = Literal["always", "on_failure", "never"]

# Options the backend writes itself; a directive repeating one would conflict.
_OWNED_OPTIONS = frozenset({
    "-t", "-tc", "-o", "-e", "-j", "-N", "-pe", "-wd", "-cwd", "-S",
    "-sync", "-terse", "-b", "-now",
})
_JOB_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_WALLTIME = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
_COMMAND_TIMEOUT = 120.0

_sleep = time.sleep
_MISSING = object()


@dataclass(frozen=True)
class GridEngineBackend(Backend):
    """Run each ``map`` call as one Grid Engine array job, submitted with ``qsub``.

    The runs are split into contiguous chunks, one per array task. Each task
    runs its chunk serially, or across ``slots`` worker processes, and
    writes results to a batch directory on a filesystem shared with the
    compute nodes. The driver polls that directory and ``qstat`` until
    every task has finished, then returns the results in run order. Queue
    wait is paid once per ``map`` call, so this backend suits a few large
    ensemble evaluations (for example one per EKI iteration) better than
    many small ones.

    The backend is stateless and immutable: every ``map`` call is a separate
    job, and :func:`dataclasses.replace` makes a variant, e.g.
    ``replace(backend, n_jobs=10)``.

    .. important:: **Requirements**

        - ``work_dir`` and the Python interpreter (``python``, by default
          the driver's ``sys.executable``) must be on filesystems mounted on
          the compute nodes at the same paths. Using the driver's own
          virtual environment means the worker imports the same packages.
        - The model callable and every field value must be picklable, as
          for :class:`~pyens.backends.LocalBackend`. The model must be
          importable by module name on the compute node: functions defined
          in a script run as ``__main__`` or in a notebook are rejected.
          Tasks start in the driver's current working directory, so
          relative paths in inputs resolve the same way as on the driver.
        - Jobs start with the environment Grid Engine gives them, not the
          driver's. Forward what the model needs with ``directives`` (e.g.
          ``"-v OMP_NUM_THREADS"``) or ``setup`` lines.

    **Failure semantics.** Nothing is raised for run or task failures; each
    run's slot holds its result or an exception:

    - The model's own exception, with the worker traceback as
      ``__cause__``, or a :class:`~pyens.backends.RemoteError` if it could
      not be unpickled.
    - A :class:`~pyens.backends.TaskFailedError` for runs whose task died
      (for example killed at ``walltime``), went into an error state, did
      not finish before ``timeout``, or could not start. Runs the task
      finished before it stopped keep their results.

    A failed submission raises :class:`~pyens.backends.GridEngineError`.
    On ``KeyboardInterrupt``, ``SIGTERM`` or ``SIGHUP`` (for example when an
    SSH session drops), the job is deleted with ``qdel``, the batch
    directory is kept, and the exception propagates.

    Progress is reported through :mod:`logging` on the
    ``pyens.backends.gridengine`` logger at ``INFO`` level.

    Args:
        walltime: Run-time limit for each array task, as ``"HH:MM:SS"`` or
            a number of seconds. Written as ``-l h_rt=...``. A task still
            running at the limit is killed and its unfinished runs fail.
        work_dir: Directory in which a new batch directory is created for
            each ``map`` call. Must be on a shared filesystem.
        n_jobs: Number of array tasks to split the runs into (fewer if there
            are fewer runs). Exactly one of ``n_jobs`` and ``runs_per_job``
            is required.
        runs_per_job: Maximum number of runs per array task.
        slots: Slots to request per task. With ``slots > 1`` the task
            requests ``-pe <parallel_env> <slots>`` and runs its chunk in
            that many worker processes.
        parallel_env: Name of the shared-memory parallel environment used
            when ``slots > 1`` (``"omp"`` on BU SCC; often ``"smp"``
            elsewhere).
        directives: Extra ``qsub`` options, one per string, written as
            ``#$`` lines in the job script, e.g. ``["-P myproject",
            "-l mem_per_core=4G"]``. Options the backend sets itself
            (``-t``, ``-tc``, ``-o``, ``-e``, ``-j``, ``-N``, ``-pe``,
            ``-wd``, ``-cwd``, ``-S``, ``-sync`` and ``h_rt``) are rejected.
        setup: Shell lines run in each task before the worker starts, e.g.
            ``["module load gcc", "export OMP_NUM_THREADS=1"]``.
        max_concurrent: Maximum number of tasks of one job running at once
            (``-tc``). ``None`` leaves it to the scheduler.
        python: Interpreter that runs the worker on the compute node.
            Defaults to ``sys.executable``.
        job_name: Grid Engine job name (``-N``); also prefixes batch
            directory names.
        poll_interval: Seconds between checks of the batch directory and
            ``qstat``.
        timeout: Wall-clock limit in seconds for a whole ``map`` call,
            including queue wait. When it expires the job is deleted and
            unfinished runs get a ``TaskFailedError`` of kind
            ``"timeout"``. ``None`` waits indefinitely.
        missing_grace: Seconds a task may be absent from ``qstat`` without a
            final result file before it is declared dead. Covers the short
            delay between a task exiting and its file becoming visible.
        keep_batch_dir: When to keep the batch directory after a completed
            ``map`` call: ``"on_failure"`` (default) keeps it if any run
            failed, ``"always"`` keeps it, ``"never"`` deletes it. It is
            always kept when ``map`` is interrupted.

    Raises:
        ValueError: If the configuration is invalid.

    Examples:
        Split an ensemble into 50 array tasks of one slot each::

            from pyens import EnsembleRunner
            from pyens.backends import GridEngineBackend

            backend = GridEngineBackend(
                walltime="01:00:00",
                work_dir="/projectnb/mygroup/me/pyens_batches",
                n_jobs=50,
                directives=["-P mygroup"],
                setup=["export OMP_NUM_THREADS=1"],
            )
            result = EnsembleRunner(my_model, backend).run(spec)

        Four processes per task, and at most 20 tasks running at once::

            backend = GridEngineBackend(
                walltime="02:00:00",
                work_dir="/projectnb/mygroup/me/pyens_batches",
                runs_per_job=2000,
                slots=4,
                max_concurrent=20,
            )
    """

    walltime: str | int
    work_dir: str | os.PathLike[str]
    n_jobs: int | None = None
    runs_per_job: int | None = None
    slots: int = 1
    parallel_env: str = "omp"
    directives: Sequence[str] = ()
    setup: Sequence[str] = ()
    max_concurrent: int | None = None
    python: str | os.PathLike[str] | None = None
    job_name: str = "pyens"
    poll_interval: float = 30.0
    timeout: float | None = None
    missing_grace: float = 60.0
    keep_batch_dir: KeepPolicy = "on_failure"

    def __post_init__(self) -> None:
        set_ = functools.partial(object.__setattr__, self)
        if (self.n_jobs is None) == (self.runs_per_job is None):
            raise ValueError("pass exactly one of n_jobs and runs_per_job")
        for name in ("n_jobs", "runs_per_job", "max_concurrent"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 1):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not _is_int(self.slots) or self.slots < 1:
            raise ValueError(f"slots must be a positive integer, got {self.slots!r}")
        set_("walltime", _normalize_walltime(self.walltime))

        work_dir = os.path.abspath(os.fspath(self.work_dir))
        if any(c.isspace() for c in work_dir):
            raise ValueError(f"work_dir must not contain whitespace: {work_dir!r}")
        set_("work_dir", work_dir)

        if not self.parallel_env or any(c.isspace() for c in self.parallel_env):
            raise ValueError(f"invalid parallel_env {self.parallel_env!r}")
        if not _JOB_NAME.match(self.job_name):
            raise ValueError(
                f"job_name must start with a letter or underscore and contain only "
                f"letters, digits, '_', '.' and '-', got {self.job_name!r}"
            )
        if isinstance(self.directives, str) or isinstance(self.setup, str):
            raise ValueError("directives and setup must be sequences of strings")
        set_("directives", tuple(self.directives))
        set_("setup", tuple(self.setup))
        for directive in self.directives:
            _check_directive(directive)
        for line in self.setup:
            if not isinstance(line, str):
                raise ValueError(f"setup lines must be strings, got {line!r}")
        if self.python is not None:
            set_("python", os.fspath(self.python))

        if not self.poll_interval > 0:
            raise ValueError("poll_interval must be positive")
        if self.timeout is not None and not self.timeout > 0:
            raise ValueError("timeout must be positive or None")
        if self.missing_grace < 0:
            raise ValueError("missing_grace must be non-negative")
        if self.keep_batch_dir not in ("always", "on_failure", "never"):
            raise ValueError(
                f"keep_batch_dir must be 'always', 'on_failure' or 'never', "
                f"got {self.keep_batch_dir!r}"
            )

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def map(
        self,
        fn: Callable[..., Any],
        runs: Iterable[dict[str, Any]],
    ) -> list[Any]:
        """Run ``fn(**inputs)`` for each element of *runs* as one array job.

        Blocks until every task has finished, failed, or the ``timeout``
        expired.

        Args:
            fn: The model callable. Must be picklable and importable by
                module name on the compute nodes.
            runs: Input dicts, one per run.

        Returns:
            Results in the order of *runs*. A failed run's slot holds its
            exception, a :class:`~pyens.backends.RemoteError`, or a
            :class:`~pyens.backends.TaskFailedError`.

        Raises:
            ValueError: If *fn* is defined in ``__main__``.
            GridEngineError: If the job could not be submitted.
            pickle.PicklingError: (or ``TypeError``/``AttributeError``) if
                *fn* or an input cannot be pickled. Raised before submission.
        """
        runs_list = list(runs)
        if not runs_list:
            return []
        _reject_main_module(fn)
        n = len(runs_list)
        if self.n_jobs is not None:
            ranges = split_evenly(n, self.n_jobs)
        else:
            ranges = split_evenly(n, -(-n // self.runs_per_job))  # type: ignore[operator]

        batch = BatchDir.create(self.work_dir, self.job_name)
        try:
            self._write_batch(batch, fn, runs_list, ranges)
        except BaseException:
            batch.remove()
            raise
        try:
            job_id = _SCHEDULER.submit(batch.script_path)
        except Exception:
            batch.remove()
            raise
        except BaseException:
            # qsub may have queued the job before the interrupt arrived.
            # Removing the directory would leave its tasks unable to start
            # and stuck in an error state, so keep it and let them run.
            logger.warning(
                "Interrupted during qsub; a job named %s may have been submitted. "
                "Check `qstat -u $USER`. Batch directory kept: %s",
                self.job_name, batch.root,
            )
            raise

        try:
            batch.job_id_path.write_text(job_id + "\n")
            logger.info(
                "Submitted Grid Engine job %s: %d runs in %d tasks. Batch directory: "
                "%s. To cancel: qdel %s",
                job_id, n, len(ranges), batch.root, job_id,
            )
            with _signals_raise_system_exit():
                failed = self._wait(batch, job_id, len(ranges))
        except BaseException:
            logger.warning(
                "Interrupted; deleting Grid Engine job %s. Batch directory kept: %s",
                job_id, batch.root,
            )
            _SCHEDULER.cancel(job_id)
            raise

        results = self._collect(batch, job_id, ranges, failed, n)
        n_failed = sum(isinstance(r, BaseException) for r in results)
        keep = self.keep_batch_dir == "always" or (
            self.keep_batch_dir == "on_failure" and n_failed > 0
        )
        if keep:
            logger.info("Job %s: %d of %d runs failed. Batch directory kept: %s",
                        job_id, n_failed, n, batch.root)
        else:
            batch.remove()
        return results

    # ------------------------------------------------------------------
    # Steps of map()
    # ------------------------------------------------------------------

    def _write_batch(
        self,
        batch: BatchDir,
        fn: Callable[..., Any],
        runs: list[dict[str, Any]],
        ranges: list[tuple[int, int]],
    ) -> None:
        try:
            batch.write_model(fn)
        except Exception as exc:
            exc.add_note("GridEngineBackend could not pickle the model callable.")
            raise
        for task, (start, stop) in enumerate(ranges, start=1):
            try:
                batch.write_inputs(task, start, runs[start:stop])
            except Exception as exc:
                exc.add_note(
                    f"GridEngineBackend could not pickle the inputs of runs "
                    f"{start}..{stop - 1}."
                )
                raise
        batch.write_manifest({
            "format_version": FORMAT_VERSION,
            "n_runs": len(runs),
            "tasks": ranges,
            "processes": self.slots,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": self._python(),
        })
        batch.script_path.write_text(self.render_script(batch, len(ranges)))

    def render_script(self, batch: BatchDir, n_tasks: int) -> str:
        """Return the job script submitted for *batch* with *n_tasks* tasks.

        Exposed for inspection and testing; ``map`` calls it for you.

        Args:
            batch: The batch the script runs.
            n_tasks: Number of array tasks.

        Returns:
            The script text.
        """
        cwd = os.getcwd()
        if any(c.isspace() for c in cwd):
            raise ValueError(
                f"the current working directory must not contain whitespace "
                f"(tasks start there): {cwd!r}"
            )
        lines = [
            "#!/bin/bash",
            "# Generated by pyens GridEngineBackend.",
            "#$ -S /bin/bash",
            f"#$ -N {self.job_name}",
            f"#$ -t 1-{n_tasks}",
        ]
        if self.max_concurrent is not None:
            lines.append(f"#$ -tc {self.max_concurrent}")
        lines.append(f"#$ -l h_rt={self.walltime}")
        if self.slots > 1:
            lines.append(f"#$ -pe {self.parallel_env} {self.slots}")
        lines += [
            f"#$ -wd {cwd}",
            "#$ -j y",
            f"#$ -o {batch.logs_dir}/task-$TASK_ID.log",
        ]
        lines += [f"#$ {d}" for d in self.directives]
        if self.setup:
            lines += ["", *self.setup]
        lines += [
            "",
            f"exec {shlex.quote(self._python())} -m pyens.backends._gridengine_worker "
            f'{shlex.quote(str(batch.root))} "$SGE_TASK_ID"',
            "",
        ]
        return "\n".join(lines)

    def _wait(
        self, batch: BatchDir, job_id: str, n_tasks: int
    ) -> dict[int, tuple[TaskFailureKind, str]]:
        """Poll until every task has a result or has failed.

        Returns:
            Tasks declared failed while waiting, with the kind and reason.
        """
        all_tasks = set(range(1, n_tasks + 1))
        failed: dict[int, tuple[TaskFailureKind, str]] = {}
        absent_since: dict[int, float] = {}
        grace = max(self.missing_grace, 2 * self.poll_interval)
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        last_report: tuple[int, ...] | None = None
        qstat_down = False

        while True:
            finished = batch.finished_tasks()
            outstanding = all_tasks - finished - failed.keys()
            if not outstanding:
                return failed
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                _SCHEDULER.cancel(job_id)
                # A task may have finished while the job was being deleted.
                outstanding -= batch.finished_tasks()
                for task in outstanding:
                    failed[task] = (
                        "timeout",
                        f"the backend timeout of {self.timeout:g} s expired "
                        f"before the task finished",
                    )
                logger.warning("Job %s: timeout expired; deleted %d unfinished tasks",
                               job_id, len(outstanding))
                return failed

            states = _SCHEDULER.status(job_id)
            running = pending = 0
            if states is None:
                if not qstat_down:
                    logger.warning("qstat failed; relying on result files until it recovers")
                qstat_down = True
            else:
                qstat_down = False
                in_error = []
                for task in sorted(outstanding):
                    state = states.get(task)
                    if state is None:
                        first_absent = absent_since.setdefault(task, now)
                        if now - first_absent >= grace:
                            failed[task] = (
                                "died",
                                "the task left the queue without writing its final "
                                "result file",
                            )
                    else:
                        absent_since.pop(task, None)
                        if "E" in state:
                            in_error.append(task)
                        elif "r" in state or "t" in state:
                            running += 1
                        else:
                            pending += 1
                if in_error:
                    detail = _SCHEDULER.error_reason(job_id)
                    for task in in_error:
                        failed[task] = (
                            "error_state",
                            f"Grid Engine put the task in an error state and it was "
                            f"deleted{': ' + detail if detail else ''}",
                        )
                    _SCHEDULER.cancel(job_id, in_error)

            report = (len(finished), running, pending, len(failed))
            if report != last_report:
                logger.info(
                    "Job %s: %d/%d tasks finished, %d running, %d pending, %d failed",
                    job_id, len(finished), n_tasks, running, pending, len(failed),
                )
                last_report = report
            _sleep(self.poll_interval)

    def _collect(
        self,
        batch: BatchDir,
        job_id: str,
        ranges: list[tuple[int, int]],
        failed: dict[int, tuple[TaskFailureKind, str]],
        n_runs: int,
    ) -> list[Any]:
        results: list[Any] = [_MISSING] * n_runs
        accounting: dict[int, dict[str, str]] | None = None
        for task, (start, stop) in enumerate(ranges, start=1):
            log_path = str(batch.log_path(task))
            final = batch.result_path(task)
            if not final.exists() and batch.error_path(task).exists():
                err = self._worker_error(batch, job_id, task, log_path)
                for i in range(start, stop):
                    results[i] = err
                continue

            source = final if final.exists() else batch.partial_path(task)
            if source.exists():
                frames, _ = read_frames(source)
                for index, output in iter_decoded(frames):
                    if start <= index < stop:
                        results[index] = output
            missing = [i for i in range(start, stop) if results[i] is _MISSING]
            if not missing:
                continue

            kind, reason = failed.get(task, (
                "died", "the task finished without a result for these runs",
            ))
            if kind == "died":
                if accounting is None:
                    accounting = _SCHEDULER.accounting(job_id)
                reason += _describe_accounting(accounting.get(task), self.walltime)
            n_done = (stop - start) - len(missing)
            if n_done:
                reason += (f"; {n_done} of {stop - start} runs in this task "
                           f"finished before it stopped")
            for i in missing:
                results[i] = TaskFailedError(kind, reason, job_id, task, log_path)
        return results

    def _worker_error(
        self, batch: BatchDir, job_id: str, task: int, log_path: str
    ) -> TaskFailedError:
        try:
            exc, tb = batch.read_task_error(task)
        except Exception as read_exc:
            return TaskFailedError(
                "worker_error",
                f"the worker failed before running the model, and its error "
                f"file could not be read ({read_exc}); see the log",
                job_id, task, log_path,
            )
        err = TaskFailedError(
            "worker_error",
            f"the worker failed before running the model: {exc}",
            job_id, task, log_path,
        )
        err.__cause__ = attach_remote_traceback(exc, tb)
        return err

    def _python(self) -> str:
        return os.fspath(self.python) if self.python is not None else sys.executable


# ----------------------------------------------------------------------
# Grid Engine commands
# ----------------------------------------------------------------------


class _GridEngine:
    """Thin wrapper around ``qsub``, ``qstat``, ``qdel`` and ``qacct``."""

    def submit(self, script: os.PathLike[str]) -> str:
        try:
            proc = _run(["qsub", "-terse", os.fspath(script)])
        except FileNotFoundError as exc:
            raise GridEngineError("qsub was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise GridEngineError(
                f"qsub did not return within {_COMMAND_TIMEOUT:g} s; check "
                f"`qstat -u $USER` for a job that may have been submitted anyway"
            ) from exc
        if proc.returncode != 0:
            raise GridEngineError(
                f"qsub failed with exit status {proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        job_id = parse_submit_output(proc.stdout)
        if job_id is None:
            raise GridEngineError(f"could not parse a job ID from qsub output {proc.stdout!r}")
        return job_id

    def status(self, job_id: str) -> dict[int, str] | None:
        try:
            proc = _run(["qstat", "-xml", "-g", "d", "-u", getpass.getuser()])
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return parse_qstat_xml(proc.stdout, job_id)

    def error_reason(self, job_id: str) -> str:
        try:
            proc = _run(["qstat", "-j", job_id])
        except (OSError, subprocess.TimeoutExpired):
            return ""
        reasons = [line.strip() for line in proc.stdout.splitlines()
                   if line.lower().startswith("error reason")]
        return "; ".join(reasons)

    def cancel(self, job_id: str, tasks: Sequence[int] | None = None) -> None:
        cmd = ["qdel", job_id]
        if tasks:
            cmd += ["-t", ",".join(str(t) for t in sorted(tasks))]
        try:
            proc = _run(cmd)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("qdel %s failed (%s); delete the job manually", job_id, exc)
            return
        if proc.returncode != 0:
            logger.debug("qdel %s exited %d: %s", job_id, proc.returncode,
                         (proc.stderr or proc.stdout).strip())

    def accounting(self, job_id: str) -> dict[int, dict[str, str]]:
        try:
            proc = _run(["qacct", "-j", job_id])
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if proc.returncode != 0:
            return {}
        return parse_qacct(proc.stdout)


_SCHEDULER = _GridEngine()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=_COMMAND_TIMEOUT, check=False,
    )


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def parse_submit_output(text: str) -> str | None:
    """Extract the job ID from ``qsub -terse`` output.

    Examples:
        >>> parse_submit_output("12345.1-10:1\\n")
        '12345'
        >>> parse_submit_output("warning: something\\n678\\n")
        '678'
    """
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)(?:\.\S*)?\s*$", line)
        if match:
            return match.group(1)
    return None


def parse_task_ids(text: str) -> list[int]:
    """Expand a Grid Engine task list such as ``"1-9:2,12"``.

    Examples:
        >>> parse_task_ids("1-5:2,8")
        [1, 3, 5, 8]
        >>> parse_task_ids("4")
        [4]
    """
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$", part)
        if not match:
            continue
        first = int(match.group(1))
        last = int(match.group(2) or first)
        step = int(match.group(3) or 1)
        ids.extend(range(first, last + 1, step))
    return ids


def parse_qstat_xml(text: str, job_id: str) -> dict[int, str] | None:
    """Map task number to state for *job_id* from ``qstat -xml`` output.

    Tasks not listed are not in the queue (finished or never existed).

    Returns:
        ``{task: state}``, e.g. ``{1: "r", 2: "qw", 3: "Eqw"}``, or ``None``
        if the XML cannot be parsed.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    states: dict[int, str] = {}
    for job in root.iter("job_list"):
        if (job.findtext("JB_job_number") or "").strip() != job_id:
            continue
        state = (job.findtext("state") or "").strip()
        tasks = job.findtext("tasks")
        for task in parse_task_ids(tasks or ""):
            states[task] = state
    return states


def parse_qacct(text: str) -> dict[int, dict[str, str]]:
    """Map task number to its accounting fields from ``qacct -j`` output.

    If a task appears more than once (it was rescheduled), the last record
    wins.
    """
    records: dict[int, dict[str, str]] = {}
    current: dict[str, str] = {}

    def flush() -> None:
        task = current.get("taskid", "")
        if task.isdigit():
            records[int(task)] = dict(current)

    for line in text.splitlines():
        if line.startswith("====="):
            flush()
            current = {}
            continue
        key, _, value = line.strip().partition(" ")
        if key:
            current[key] = value.strip()
    flush()
    return records


def _describe_accounting(record: dict[str, str] | None, walltime: str) -> str:
    if record is None:
        return " (no qacct record found)"
    fields = [f"{k}={record[k]}" for k in
              ("exit_status", "failed", "ru_wallclock", "maxvmem") if k in record]
    text = f" (qacct: {', '.join(fields)})" if fields else ""
    exit_status = record.get("exit_status", "").split()[:1]
    failed_code = record.get("failed", "").split()[:1]
    if exit_status in (["137"], ["9"]) or failed_code == ["37"]:
        text += (f"; killed with SIGKILL, typically for exceeding h_rt={walltime} "
                 f"or a memory limit")
    return text


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_walltime(value: str | int) -> str:
    if _is_int(value):
        if value < 1:  # type: ignore[operator]
            raise ValueError("walltime must be at least one second")
        hours, rest = divmod(int(value), 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if isinstance(value, str) and _WALLTIME.match(value.strip()):
        return value.strip()
    raise ValueError(f"walltime must be 'HH:MM:SS' or a number of seconds, got {value!r}")


def _check_directive(directive: Any) -> None:
    if not isinstance(directive, str):
        raise ValueError(f"directives must be strings, got {directive!r}")
    if "\n" in directive or "\r" in directive:
        raise ValueError(f"a directive must be a single line: {directive!r}")
    tokens = directive.split()
    if not tokens or not tokens[0].startswith("-"):
        raise ValueError(
            f"a directive must be a qsub option starting with '-', e.g. "
            f"'-P myproject', got {directive!r}"
        )
    option = tokens[0]
    if option in _OWNED_OPTIONS:
        raise ValueError(
            f"directive {directive!r} sets {option}, which GridEngineBackend "
            f"sets itself; use its arguments instead"
        )
    if option in ("-l", "-hard", "-soft") and re.search(r"\b[hs]_rt\s*=", directive):
        raise ValueError(
            f"directive {directive!r} sets a run-time limit; use walltime= instead"
        )


def _reject_main_module(fn: Callable[..., Any]) -> None:
    candidates: list[Any] = [fn, type(fn)]
    if isinstance(fn, functools.partial):
        candidates += [fn.func, type(fn.func)]
    owner = getattr(fn, "__self__", None)
    if owner is not None:
        candidates.append(type(owner))
    for obj in candidates:
        if getattr(obj, "__module__", None) == "__main__":
            name = getattr(obj, "__qualname__", repr(obj))
            raise ValueError(
                f"the model {name!r} is defined in __main__ (a script or notebook), "
                f"so worker processes on the compute nodes cannot import it. Move "
                f"it into an importable module and import it from there."
            )


@contextlib.contextmanager
def _signals_raise_system_exit() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP into ``SystemExit`` so cleanup code runs.

    Only signals whose handler is the default are changed, so an ignored
    SIGHUP (under ``nohup``) or a caller's own handler is left alone. Does
    nothing outside the main thread, where handlers cannot be installed.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous: dict[int, Any] = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None or signal.getsignal(sig) is not signal.SIG_DFL:
            continue
        previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)
