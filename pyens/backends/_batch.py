"""On-disk layout of one batch: the files exchanged between driver and tasks.

A batch is one ``map`` call submitted as one array job. Its directory lives
under the backend's ``work_dir`` on a filesystem shared by the driver and the
compute nodes::

    <job_name>-<YYYYmmdd-HHMMSS>-<8 hex>/
        PYENS_BATCH            marker; only directories with it are deleted
        manifest.json          format version, run count, task ranges, ...
        model.pkl              the pickled model callable
        job.sh                 the submitted job script (re-runnable by hand)
        job_id                 the scheduler job ID, once submitted
        inputs/task-<t>.pkl    pickled (start_index, [inputs, ...]) for task t
        results/task-<t>.partial   records written so far by a running task
        results/task-<t>.pkl       the same file, renamed when the task finished
        results/task-<t>.error     pickled (exception, traceback) if the worker
                                   failed before running the model
        logs/task-<t>.log      the task's combined stdout and stderr

Task numbers ``t`` start at 1, matching Grid Engine's ``SGE_TASK_ID``. Task
``t`` runs ensemble indices ``manifest["tasks"][t - 1]`` = ``[start, stop)``.
Result files contain records framed as described in
:mod:`pyens.backends._chunk`.
"""

from __future__ import annotations

import json
import os
import pickle
import secrets
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
MARKER = "PYENS_BATCH"
_PROTOCOL = pickle.HIGHEST_PROTOCOL


def split_evenly(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``min(k, n)`` contiguous ``[start, stop)`` ranges.

    Range sizes differ by at most one, larger ranges first.

    Examples:
        >>> split_evenly(7, 3)
        [(0, 3), (3, 5), (5, 7)]
        >>> split_evenly(2, 5)
        [(0, 1), (1, 2)]
    """
    if n <= 0:
        return []
    k = max(1, min(k, n))
    base, extra = divmod(n, k)
    ranges = []
    start = 0
    for i in range(k):
        stop = start + base + (1 if i < extra else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


class BatchDir:
    """Paths and file operations for one batch directory.

    Args:
        root: The batch directory.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, work_dir: str | os.PathLike[str], prefix: str) -> BatchDir:
        """Create a new, uniquely named batch directory under *work_dir*."""
        base = Path(work_dir)
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = base / f"{prefix}-{stamp}-{secrets.token_hex(4)}"
        root.mkdir()
        for sub in ("inputs", "results", "logs"):
            (root / sub).mkdir()
        (root / MARKER).write_text(f"pyens batch format {FORMAT_VERSION}\n")
        return cls(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def model_path(self) -> Path:
        return self.root / "model.pkl"

    @property
    def script_path(self) -> Path:
        return self.root / "job.sh"

    @property
    def job_id_path(self) -> Path:
        return self.root / "job_id"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def inputs_path(self, task: int) -> Path:
        return self.root / "inputs" / f"task-{task}.pkl"

    def partial_path(self, task: int) -> Path:
        return self.results_dir / f"task-{task}.partial"

    def result_path(self, task: int) -> Path:
        return self.results_dir / f"task-{task}.pkl"

    def error_path(self, task: int) -> Path:
        return self.results_dir / f"task-{task}.error"

    def log_path(self, task: int) -> Path:
        return self.logs_dir / f"task-{task}.log"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def write_model(self, fn: Callable[..., Any]) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(fn, f, protocol=_PROTOCOL)

    def load_model(self) -> Callable[..., Any]:
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def write_inputs(
        self, task: int, start: int, inputs: Sequence[dict[str, Any]]
    ) -> None:
        # One pickle per task, so field values shared by every run (Fixed
        # fields) are stored once per task through the pickle memo.
        with open(self.inputs_path(task), "wb") as f:
            pickle.dump((start, list(inputs)), f, protocol=_PROTOCOL)

    def load_inputs(self, task: int) -> tuple[int, list[dict[str, Any]]]:
        with open(self.inputs_path(task), "rb") as f:
            return pickle.load(f)

    def write_task_error(self, task: int, exc: BaseException, traceback: str) -> None:
        tmp = self.error_path(task).with_suffix(".error.tmp")
        with open(tmp, "wb") as f:
            pickle.dump((exc, traceback), f, protocol=_PROTOCOL)
        os.replace(tmp, self.error_path(task))

    def read_task_error(self, task: int) -> tuple[BaseException, str]:
        with open(self.error_path(task), "rb") as f:
            return pickle.load(f)

    def finished_tasks(self) -> set[int]:
        """Task numbers that have a final result file or an error file."""
        done: set[int] = set()
        try:
            names = os.listdir(self.results_dir)
        except FileNotFoundError:
            return done
        for name in names:
            stem, _, suffix = name.partition(".")
            if suffix in ("pkl", "error") and stem.startswith("task-"):
                try:
                    done.add(int(stem[5:]))
                except ValueError:
                    continue
        return done

    def remove(self) -> None:
        """Delete the batch directory, but only if it carries the marker."""
        if (self.root / MARKER).is_file():
            shutil.rmtree(self.root, ignore_errors=True)
