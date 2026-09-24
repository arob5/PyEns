"""A minimal fake of Grid Engine's qsub/qstat/qdel/qacct for local tests.

Invoked as ``python fake_ge.py <command> [args...]`` through small wrapper
scripts named ``qsub``, ``qstat``, ``qdel`` and ``qacct`` on ``PATH``. All
state lives under ``$FAKE_GE_STATE``:

    next_id                 job ID counter
    faults.json             fault injection, read when each command runs
    calls.log               one JSON line per command invocation
    jobs/<id>/job.json      parsed submission
    jobs/<id>/submitted.sh  copy of the submitted script
    jobs/<id>/task-<t>.state    "qw", "r" or "Eqw" while the task is queued
    jobs/<id>/task-<t>.pid      process group of a running task
    jobs/<id>/task-<t>.exit     accounting record once the task has ended
    jobs/<id>/cancel-<t>        written by qdel so the daemon never starts t

``qsub`` starts a detached daemon (``_run``) per job that launches tasks
(respecting ``-tc``) as ``bash job.sh`` with ``SGE_TASK_ID`` set and a
minimal environment, as on a cluster with ``INHERIT_ENV=false``.

Faults (keys of ``faults.json``):

    qsub_fail: str          qsub exits 1 with this message
    qstat_fail: bool        qstat exits 1
    eqw_tasks: [int]        these tasks go to "Eqw" and never run
    pending_forever: bool   no task ever starts
    kill_after: {t: secs}   SIGKILL task t after secs, recorded as an h_rt kill
    qstat_hide: [int]       these tasks never appear in qstat output
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("FAKE_GE_STATE", "/nonexistent"))


def _faults() -> dict:
    path = STATE / "faults.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _log_call(argv: list[str]) -> None:
    with open(STATE / "calls.log", "a") as f:
        f.write(json.dumps(argv) + "\n")


def _job_dir(job_id: str) -> Path:
    return STATE / "jobs" / str(job_id)


# ----------------------------------------------------------------------
# qsub
# ----------------------------------------------------------------------


def qsub(args: list[str]) -> int:
    faults = _faults()
    if faults.get("qsub_fail"):
        print(f"Unable to run job: {faults['qsub_fail']}", file=sys.stderr)
        return 1
    script = Path(args[-1])
    opts: dict[str, list[str]] = {}
    for line in script.read_text().splitlines():
        if line.startswith("#$"):
            tokens = shlex.split(line[2:])
            opts.setdefault(tokens[0], []).append(" ".join(tokens[1:]))
    first, last = opts["-t"][0].split("-")
    n_tasks = int(last)
    assert int(first) == 1
    pe = opts.get("-pe", [""])[0].split()
    counter = STATE / "next_id"
    job_id = int(counter.read_text()) if counter.exists() else 1000
    counter.write_text(str(job_id + 1))
    forwarded = {}
    for spec in opts.get("-v", []):
        for item in spec.split(","):
            name, eq, value = item.partition("=")
            forwarded[name] = value if eq else os.environ.get(name, "")
    job = {
        "id": str(job_id),
        "script": str(script),
        "n_tasks": n_tasks,
        "tc": int(opts["-tc"][0]) if "-tc" in opts else None,
        "out": opts["-o"][0],
        "wd": opts["-wd"][0],
        "name": opts["-N"][0],
        "slots": int(pe[1]) if pe else 1,
        "env": forwarded,
        "options": opts,
    }
    jd = _job_dir(str(job_id))
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps(job))
    (jd / "submitted.sh").write_text(script.read_text())
    for t in range(1, n_tasks + 1):
        (jd / f"task-{t}.state").write_text("qw")
    daemon = subprocess.Popen(
        [sys.executable, __file__, "_run", str(job_id)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(STATE / "pids", "a") as f:
        f.write(f"{daemon.pid}\n")
    print(f"{job_id}.1-{n_tasks}:1")
    return 0


def _run(job_id: str) -> int:
    jd = _job_dir(job_id)
    job = json.loads((jd / "job.json").read_text())
    faults = _faults()
    eqw = set(faults.get("eqw_tasks", []))
    kill_after = {int(k): float(v) for k, v in faults.get("kill_after", {}).items()}
    queue = list(range(1, job["n_tasks"] + 1))
    running: dict[int, tuple[subprocess.Popen, float]] = {}
    limit = job["tc"] or 16
    while True:
        for t, (proc, started) in list(running.items()):
            rc = proc.poll()
            killed = t in kill_after and time.monotonic() - started > kill_after[t]
            if rc is None and killed:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                _finish(jd, t, 137, "37  : qmaster enforced h_rt, h_cpu, or h_vmem limit",
                        started)
                del running[t]
            elif rc is not None:
                status = 128 - rc if rc < 0 else rc
                _finish(jd, t, status, "0", started)
                del running[t]
        if not faults.get("pending_forever"):
            while queue and len(running) < limit:
                t = queue.pop(0)
                if (jd / f"cancel-{t}").exists():
                    continue
                if t in eqw:
                    (jd / f"task-{t}.state").write_text("Eqw")
                    continue
                running[t] = (_start(job, jd, t), time.monotonic())
        if not running and (not queue or faults.get("pending_forever")):
            return 0
        time.sleep(0.02)


def _start(job: dict, jd: Path, t: int) -> subprocess.Popen:
    log = (job["out"].replace("$TASK_ID", str(t))
           .replace("$JOB_ID", job["id"]).replace("$JOB_NAME", job["name"]))
    tmp = jd / f"tmp-{t}"
    tmp.mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/"),
        "SGE_TASK_ID": str(t),
        "JOB_ID": job["id"],
        "JOB_NAME": job["name"],
        "NSLOTS": str(job["slots"]),
        "TMPDIR": str(tmp),
        **job["env"],
    }
    with open(log, "ab") as out:
        proc = subprocess.Popen(
            ["bash", job["script"]], cwd=job["wd"], env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (jd / f"task-{t}.pid").write_text(str(proc.pid))
    (jd / f"task-{t}.state").write_text("r")
    return proc


def _finish(jd: Path, t: int, exit_status: int, failed: str, started: float) -> None:
    record = {
        "taskid": str(t),
        "exit_status": str(exit_status),
        "failed": failed,
        "ru_wallclock": f"{time.monotonic() - started:.0f}s",
    }
    (jd / f"task-{t}.exit").write_text(json.dumps(record))
    for suffix in ("state", "pid"):
        (jd / f"task-{t}.{suffix}").unlink(missing_ok=True)


# ----------------------------------------------------------------------
# qstat / qdel / qacct
# ----------------------------------------------------------------------


def qstat(args: list[str]) -> int:
    faults = _faults()
    if faults.get("qstat_fail"):
        print("error: failed receiving gdi request", file=sys.stderr)
        return 1
    if "-j" in args:
        job_id = args[args.index("-j") + 1]
        print(f"job_number: {job_id}")
        print("error reason    4:      can't make directory /nonexistent")
        return 0
    hidden = set(faults.get("qstat_hide", []))
    running, pending = [], []
    for jd in sorted((STATE / "jobs").glob("*")) if (STATE / "jobs").exists() else []:
        for sf in jd.glob("task-*.state"):
            t = int(sf.name.split("-")[1].split(".")[0])
            if t in hidden:
                continue
            try:
                state = sf.read_text()
            except FileNotFoundError:
                continue
            entry = (
                f"    <job_list state=\"{'running' if state == 'r' else 'pending'}\">\n"
                f"      <JB_job_number>{jd.name}</JB_job_number>\n"
                f"      <JB_name>fake</JB_name>\n"
                f"      <state>{state}</state>\n"
                f"      <tasks>{t}</tasks>\n"
                f"    </job_list>\n"
            )
            (running if state == "r" else pending).append(entry)
    if "-xml" not in args:
        print("job-ID  prior   name       user         state submit/start at     queue")
        print("-" * 70)
        for entry in running + pending:
            number = entry.split("<JB_job_number>")[1].split("<")[0]
            state = entry.split("<state>")[1].split("<")[0]
            print(f"{number} 0.5 fake arober {state} 01/01/2026 00:00:00 all.q")
        return 0
    print("<?xml version='1.0'?>\n<job_info>\n  <queue_info>\n" + "".join(running)
          + "  </queue_info>\n  <job_info>\n" + "".join(pending)
          + "  </job_info>\n</job_info>")
    return 0


def qdel(args: list[str]) -> int:
    job_id = args[0]
    jd = _job_dir(job_id)
    if not jd.exists():
        print(f"denied: job \"{job_id}\" does not exist", file=sys.stderr)
        return 1
    job = json.loads((jd / "job.json").read_text())
    if "-t" in args:
        tasks = [int(t) for t in args[args.index("-t") + 1].split(",")]
    else:
        tasks = list(range(1, job["n_tasks"] + 1))
    for t in tasks:
        (jd / f"cancel-{t}").write_text("")
        pid_file = jd / f"task-{t}.pid"
        if pid_file.exists():
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        (jd / f"task-{t}.state").unlink(missing_ok=True)
    print(f"arober has registered the job {job_id} for deletion")
    return 0


def qacct(args: list[str]) -> int:
    job_id = args[args.index("-j") + 1]
    records = sorted(_job_dir(job_id).glob("task-*.exit"))
    if not records:
        print(f"error: job id {job_id} not found", file=sys.stderr)
        return 1
    for path in records:
        record = json.loads(path.read_text())
        print("=" * 62)
        print(f"jobnumber    {job_id}")
        for key, value in record.items():
            print(f"{key:<12} {value}")
    return 0


def main() -> int:
    command, args = sys.argv[1], sys.argv[2:]
    if command != "_run":
        _log_call([command, *args])
    return {"qsub": qsub, "qstat": qstat, "qdel": qdel, "qacct": qacct,
            "_run": lambda a: _run(a[0])}[command](args)


if __name__ == "__main__":
    sys.exit(main())
