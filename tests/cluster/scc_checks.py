"""Manual checks of GridEngineBackend on a real Grid Engine cluster.

These submit real jobs, so they are not collected by pytest and do nothing
without ``--submit``. Each check submits exactly one small array job
running trivial workloads (arithmetic and sleeps), never retries, and
deletes its job on any failure. Run from the repository root::

    python -m tests.cluster.scc_checks s1 --work-dir /path/on/shared/fs \\
        --directive "-P myproject" --submit

Without ``--submit`` the job script is printed and nothing is submitted.

Checks (tasks x slots, h_rt, worst-case core-hours = tasks*slots*h_rt):

    s1  smoke: 30 x 0.1 s runs                  3 x 1, 00:05:00, 0.25
    s2  failure semantics                       4 x 1, 00:03:00, 0.20
    s3  Ctrl-C deletes the job                  3 x 1, 00:05:00, 0.25
    s4  -pe with 2 processes, -tc 1             2 x 2, 00:05:00, 0.33
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pyens.backends import GridEngineBackend, RemoteError, TaskFailedError
from pyens.backends._batch import BatchDir
from pyens.backends.gridengine import parse_qstat_xml
from tests import _models

USER = getpass.getuser()


def _qstat_jobs() -> dict[str, str]:
    """Job ID -> job name for the current user's queued or running jobs."""
    out = subprocess.run(["qstat", "-u", USER], capture_output=True, text=True,
                         check=True).stdout
    jobs = {}
    for line in out.splitlines()[2:]:
        fields = line.split()
        if fields and fields[0].isdigit():
            jobs[fields[0]] = fields[2]
    return jobs


def _job_ids_in(work_dir: Path) -> list[str]:
    ids = []
    for path in sorted(work_dir.glob("*/job_id")):
        ids.append(path.read_text().strip())
    return ids


class Watcher(threading.Thread):
    """Sample qstat while a check runs; save raw output for later inspection."""

    def __init__(self, work_dir: Path, out_dir: Path) -> None:
        super().__init__(daemon=True)
        self.work_dir = work_dir
        self.out_dir = out_dir
        self.stop = threading.Event()
        self.samples: list[dict[int, str]] = []
        self.job_id: str | None = None

    def run(self) -> None:
        n = 0
        while not self.stop.is_set():
            ids = _job_ids_in(self.work_dir)
            if ids and self.job_id is None:
                self.job_id = ids[-1]
                detail = subprocess.run(["qstat", "-j", self.job_id],
                                        capture_output=True, text=True).stdout
                (self.out_dir / "qstat_j.txt").write_text(detail)
            if self.job_id is not None:
                xml = subprocess.run(["qstat", "-xml", "-g", "d", "-u", USER],
                                     capture_output=True, text=True).stdout
                states = parse_qstat_xml(xml, self.job_id)
                if states:
                    self.samples.append(states)
                    if n < 5:
                        (self.out_dir / f"qstat_{n}.xml").write_text(xml)
                        n += 1
            self.stop.wait(5)


def check_s1(backend: GridEngineBackend) -> list[str]:
    backend = dataclasses.replace(backend, n_jobs=3, walltime="00:05:00")
    runs = [{"x": i, "y": 1000, "delay": 0.1} for i in range(30)]
    results = backend.map(_models.slow_add, runs)
    problems = []
    if results != [1000 + i for i in range(30)]:
        problems.append(f"unexpected results: {results!r}")
    return problems


def check_s2(backend: GridEngineBackend) -> list[str]:
    backend = dataclasses.replace(backend, n_jobs=4, walltime="00:03:00")
    runs = [
        {"mode": "ok", "x": 0}, {"mode": "ok", "x": 1},              # task 1
        {"mode": "kwonly", "x": 2}, {"mode": "unpicklable", "x": 3},  # task 2
        {"mode": "ok", "x": 4}, {"mode": "exit", "x": 5},             # task 3
        {"mode": "ok", "x": 6}, {"mode": "sleep", "x": 600},          # task 4
    ]
    results = backend.map(_models.by_mode, runs)
    for i, r in enumerate(results):
        print(f"  run {i}: {r!r}\n          {r}" if isinstance(r, BaseException)
              else f"  run {i}: {r!r}")
    problems = []
    if [results[i] for i in (0, 1, 4, 6)] != [0, 1, 4, 6]:
        problems.append("successful runs are wrong")
    if not (isinstance(results[2], RemoteError)
            and results[2].attributes.get("returncode") == 2):
        problems.append(f"run 2 should be a RemoteError with attributes: {results[2]!r}")
    if not isinstance(results[3], RemoteError):
        problems.append(f"run 3 should be a RemoteError: {results[3]!r}")
    for i in (5, 7):
        if not (isinstance(results[i], TaskFailedError) and results[i].kind == "died"):
            problems.append(f"run {i} should be TaskFailedError(died): {results[i]!r}")
    if isinstance(results[7], TaskFailedError) and "qacct" not in results[7].reason:
        problems.append("run 7 reason lacks qacct details")
    return problems


def check_s3(backend: GridEngineBackend, watcher: Watcher) -> list[str]:
    backend = dataclasses.replace(backend, n_jobs=3, walltime="00:05:00")
    runs = [{"x": i, "y": 0, "delay": 120.0} for i in range(3)]

    def interrupt_once_running() -> None:
        while not any("r" in s for sample in watcher.samples for s in sample.values()):
            time.sleep(2)
        time.sleep(10)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt_once_running, daemon=True).start()
    try:
        backend.map(_models.slow_add, runs)
    except KeyboardInterrupt:
        pass
    else:
        return ["map returned instead of raising KeyboardInterrupt"]
    time.sleep(15)
    left = [j for j in _job_ids_in(backend_work_dir(backend)) if j in _qstat_jobs()]
    return [f"jobs still queued after interrupt: {left}"] if left else []


def check_s4(backend: GridEngineBackend, watcher: Watcher) -> list[str]:
    backend = dataclasses.replace(
        backend, n_jobs=2, walltime="00:05:00", slots=2, max_concurrent=1)
    runs = [{"x": i, "delay": 1.0} for i in range(40)]
    results = backend.map(_models.host_pid, runs)
    problems = []
    if [r[0] for r in results if isinstance(r, tuple)] != list(range(40)):
        problems.append(f"unexpected results: {results!r}")
        return problems
    for task, chunk in ((1, results[:20]), (2, results[20:])):
        pids = {pid for _, _, pid in chunk}
        hosts = {host for _, host, _ in chunk}
        print(f"  task {task}: hosts {sorted(hosts)}, {len(pids)} worker processes")
        if len(pids) != 2:
            problems.append(f"task {task} used {len(pids)} processes, expected 2")
    most = max((sum("r" in s for s in sample.values()) for sample in watcher.samples),
               default=0)
    print(f"  most tasks seen running at once: {most}")
    if most > 1:
        problems.append("-tc 1 was not respected")
    return problems


def backend_work_dir(backend: GridEngineBackend) -> Path:
    return Path(os.fspath(backend.work_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["s1", "s2", "s3", "s4"])
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--directive", action="append", default=[])
    parser.add_argument("--setup", action="append", default=[])
    parser.add_argument("--submit", action="store_true",
                        help="actually submit the job (otherwise print the script)")
    args = parser.parse_args()

    work_dir = args.work_dir / args.check
    backend = GridEngineBackend(
        walltime="00:05:00", work_dir=work_dir, n_jobs=1,
        directives=args.directive, setup=args.setup,
        job_name=f"pyenstest-{args.check}", poll_interval=10.0, missing_grace=60.0,
        timeout=900.0, keep_batch_dir="always",
    )
    if not args.submit:
        preview = {"s1": (3, "00:05:00", 1), "s2": (4, "00:03:00", 1),
                   "s3": (3, "00:05:00", 1), "s4": (2, "00:05:00", 2)}[args.check]
        n_tasks, walltime, slots = preview
        variant = dataclasses.replace(backend, walltime=walltime, slots=slots,
                                      max_concurrent=1 if args.check == "s4" else None)
        batch = BatchDir.create(Path(os.environ.get("TMPDIR", "/tmp")), "preview")
        print(variant.render_script(batch, n_tasks))
        batch.remove()
        print("Dry run: nothing submitted. Add --submit to run this check.")
        return 0

    before = _qstat_jobs()
    print(f"qstat -u {USER} before: {before or 'no jobs'}")
    clashing = [j for j, name in before.items() if name.startswith("pyenstest")]
    if clashing:
        print(f"Refusing to run: pyenstest jobs already queued: {clashing}")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = work_dir / "observations"
    out_dir.mkdir(parents=True, exist_ok=True)
    watcher = Watcher(work_dir, out_dir)
    watcher.start()
    started = time.time()
    problems: list[str] = []
    try:
        if args.check == "s1":
            problems = check_s1(backend)
        elif args.check == "s2":
            problems = check_s2(backend)
        elif args.check == "s3":
            problems = check_s3(backend, watcher)
        else:
            problems = check_s4(backend, watcher)
    except BaseException as exc:
        problems = [f"check raised {exc!r}"]
    finally:
        watcher.stop.set()
        watcher.join(timeout=10)
        ours = _job_ids_in(work_dir)
        still = [j for j in ours if j in _qstat_jobs()]
        for job_id in still:
            print(f"Deleting leftover job {job_id}")
            subprocess.run(["qdel", job_id], check=False)
        after = _qstat_jobs()
        print(f"qstat -u {USER} after: {after or 'no jobs'}")
        (out_dir / "samples.json").write_text(json.dumps(watcher.samples))

    print(f"Elapsed {time.time() - started:.0f} s; job IDs: {_job_ids_in(work_dir)}")
    if problems:
        print("FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
