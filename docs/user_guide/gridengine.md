# Running on a Grid Engine Cluster

[Running an Ensemble](running.md) introduced backends that run on the machine
you are using. This page covers `GridEngineBackend`, which runs an ensemble on
a cluster managed by Grid Engine (SGE, Open Grid Scheduler, Univa/Altair Grid
Engine), the scheduler behind `qsub`, `qstat` and `qdel` on clusters such as
Boston University's SCC.

`GridEngineBackend` needs no extra dependencies. Your spec and model code do
not change: you swap the backend.

---

## Quick start

```python
import logging

from pyens import EnsembleRunner
from pyens.backends import GridEngineBackend

logging.basicConfig(level=logging.INFO)   # show submission and progress messages

backend = GridEngineBackend(
    walltime="01:00:00",                                # per array task
    work_dir="/projectnb/mygroup/me/pyens_batches",     # on a shared filesystem
    n_jobs=50,                                          # split the runs into 50 tasks
    directives=["-P mygroup"],                          # any extra qsub options
    setup=["export OMP_NUM_THREADS=1"],                 # shell lines run before each task
)

runner = EnsembleRunner(my_model, backend)
result = runner.run(spec)
```

`runner.run(spec)` submits one job, waits for it to finish, and returns an
`EnsembleResult` exactly as `LocalBackend` would.

---

## How it works

Each call to `map` (each `runner.run(spec)`) becomes **one Grid Engine array
job** (a single `qsub` submission made of numbered, independently scheduled
*tasks*):

1. PyEns splits the runs into contiguous chunks, one per array task. With
   `n_jobs=K` there are `K` chunks of nearly equal size. With
   `runs_per_job=R` each chunk has at most `R` runs.
2. It creates a **batch directory** under `work_dir`, writes the pickled model
   and each task's inputs into it, and submits a job script with
   `qsub -t 1-K`.
3. Each task starts `python -m pyens.backends._gridengine_worker`, loads its
   chunk, and runs it serially (or across `slots` worker processes). It
   appends each run's result to a file in the batch directory as soon as the
   run finishes.
4. The driver (your Python process) polls the batch directory and `qstat`
   every `poll_interval` seconds until every task has finished or failed, then
   returns the results in run order.

Your runs spread over up to `K` scheduler slots. Tasks queue and start
independently, so `K` tasks behave like `K` separate jobs, but you make only
one submission.

:::{note}
You pay the queue wait once per `map` call. That suits a few large ensemble
evaluations, such as one per iteration of ensemble Kalman inversion (EKI), better than
many small ones. For workloads that call the model thousands of times with a
handful of runs each (for example Metropolis–Hastings), don't submit a job
per call: run your whole driver inside one batch job with `LocalBackend`
(see *Where to run the driver* below).
:::

---

## Choosing the number of tasks and the walltime

Three settings trade serial run time per task against time spent waiting in
the queue:

- **`n_jobs`** (or **`runs_per_job`**): more tasks means more parallelism but
  more scheduling. A useful rule of thumb is to have at least as many tasks
  as slots you can realistically get at once. When there are more tasks than
  free slots, tasks start as slots free up, which also balances the load
  across fast and slow nodes.
- **`slots`**: with `slots=N`, each task requests `-pe omp N` and runs its
  chunk in `N` worker processes. This packs more runs onto a node per
  scheduling decision.
- **`walltime`**: the run-time limit of each task (`-l h_rt`). Grid Engine
  kills a task that exceeds it, and the runs it hadn't finished fail.

Estimate the walltime from the time per run:

```text
runs per task  = n_runs / n_jobs
task run time  ≈ runs per task × seconds per run / slots
walltime       = task run time × safety factor (2 is reasonable)
```

For example, 800,000 runs at 0.35 s each with `n_jobs=200` and `slots=4`
gives 4,000 runs per task and about 350 s (6 minutes) per task, so
`walltime="00:15:00"` leaves plenty of margin.

The backend is immutable. Use {func}`dataclasses.replace` for an evaluation of
a different size:

```python
from dataclasses import replace

small = replace(backend, n_jobs=5, walltime="00:10:00")
```

---

## Requirements for the model and the environment

Array tasks run in fresh processes on compute nodes, so:

- **`work_dir` must be on a filesystem shared by the driver and the compute
  nodes** (on SCC, `/projectnb` or `/restricted/projectnb`, not `/tmp` or a
  node-local `$TMPDIR`).
- **The worker uses the same Python interpreter as the driver**
  (`sys.executable`), so it sees the same installed packages. That
  interpreter and its virtual environment must also be on a shared
  filesystem. Pass `python=` to use a different interpreter.
- **The model and every field value must be picklable**, as for
  `LocalBackend`, and the model must be **importable by module name**.
  Functions defined in a script run directly (`python my_script.py`) or in a
  notebook live in `__main__`, which workers cannot import, so
  `GridEngineBackend` rejects them. Put the model in a module and import it.
- **Tasks start in the driver's current working directory**, so relative
  paths in your inputs resolve the same way as on the driver.
- **Tasks don't inherit your shell environment.** Forward variables with a
  directive (`"-v OMP_NUM_THREADS"`, or `"-V"` for everything) or set them in
  `setup` lines. `setup` is also the place for `module load` commands.

Pickling errors, and a model defined in `__main__`, are reported **before**
anything is submitted.

### Directives

`directives` takes raw `qsub` options, one per string, written as `#$` lines
in the job script:

```python
directives=["-P mygroup", "-l mem_per_core=4G", "-v OMP_NUM_THREADS"]
```

PyEns writes some options itself and rejects directives that would conflict
with them: `-t`, `-tc`, `-o`, `-e`, `-j`, `-N`, `-pe`, `-wd`, `-cwd`, `-S`,
`-sync` and `h_rt`. Use `n_jobs`, `max_concurrent`, `slots`, `job_name` and
`walltime` instead.

Keep group- or site-specific settings (project names, special queues,
environment variables) in your own code, for example as a small function in
your project that builds the backend.

---

## Failed runs

As with every backend, failures never stop the ensemble. Each run's slot in
the result holds its output or an exception. Check `record.failed`, and look at
the exception's type to tell the cases apart:

| What is in the slot | Meaning |
|---|---|
| The model's own exception (e.g. `ValueError`) | The model raised it. Its `__cause__` holds the traceback from the compute node. |
| `RemoteError` | The model raised an exception that could not be sent back intact (typically because its class needs keyword-only constructor arguments). `type_name`, `message`, `traceback` and `attributes` hold what was recoverable. |
| `TaskFailedError` | The run never finished because its array task failed. `kind` says why; `reason`, `task_id`, `job_id` and `log_path` give details. |

`TaskFailedError.kind` is one of:

- `"died"`: the task left the queue without finishing these runs, for example
  because it exceeded its walltime or memory limit, its node failed, or someone ran
  `qdel` on it. `reason` includes `qacct` accounting details when available.
- `"error_state"`: Grid Engine put the task into an error state (`Eqw`), so it
  never ran. PyEns deletes such tasks rather than waiting forever.
- `"timeout"`: the backend's `timeout` expired first.
- `"worker_error"`: the task started but failed before running the model,
  typically because a module could not be imported on the compute node. The
  underlying exception is the `__cause__`.

**Runs a task finished before it failed keep their real results.** If a task
is killed partway through its chunk, only its unfinished runs get a
`TaskFailedError`.

```python
from pyens.backends import RemoteError, TaskFailedError

for record in result.failed:
    err = record.output
    if isinstance(err, TaskFailedError):
        print(f"{record.coordinate}: task {err.task_id} {err.kind}, see {err.log_path}")
    elif isinstance(err, RemoteError):
        print(f"{record.coordinate}: {err.type_name}: {err.message}")
    else:
        print(f"{record.coordinate}: {err!r}")
```

PyEns never resubmits failed tasks. A task that dies once, from a walltime
that is too short or a broken environment, would usually die again. Inspect
the failures and decide.

A failure to submit at all (`qsub` rejects the job) raises
`GridEngineError`, because in that case nothing is running.

---

## Where to run the driver

The driver process stays alive for the whole `map` call: submitting,
polling, and collecting results. You have two options:

- **As a batch job itself (recommended for long runs).** Submit your driver
  script as an ordinary one-slot job with a generous walltime. Compute nodes
  on SCC can submit jobs, so the driver's array jobs are submitted from
  there. The driver survives your laptop sleeping or your SSH connection
  dropping.
- **On a login node**, inside `tmux` or `screen`. Polling costs almost no
  CPU, but pickling large ensembles and your algorithm's own computation
  may not be appropriate on a shared login node.

If the driver receives `Ctrl-C`, `SIGTERM`, `SIGHUP` (an SSH session
closing), `SIGUSR1` or `SIGUSR2`, PyEns deletes the array job with `qdel`,
keeps the batch directory, and re-raises the exception.

Grid Engine kills a job that reaches its `h_rt` limit, or is deleted with
`qdel`, with `SIGKILL`, which no program can intercept. When the driver
itself runs as a batch job, give it a warning first so it can clean up:

- submit the driver with `-notify`, so Grid Engine sends `SIGUSR2` shortly
  before killing it, or
- request a soft limit a little below the hard one, for example
  `-l h_rt=48:00:00,s_rt=47:50:00`, so it receives `SIGUSR1` ten minutes
  early.

If the driver is killed without warning, its array tasks keep running until
they finish or reach their own walltime. The submission message logged at
`INFO` level includes the `qdel` command to cancel them.

---

## Progress, timeouts and cleanup

`GridEngineBackend` logs to the `pyens.backends.gridengine` logger. At
`INFO` level you see the job ID and batch directory at submission, then a line
whenever the counts of finished, running, pending and failed tasks change.
Enable it with `logging.basicConfig(level=logging.INFO)` or by configuring
that logger.

`timeout` (seconds) bounds a whole `map` call, including queue wait. When it
expires, PyEns deletes the job, keeps whatever runs had finished, and gives
the rest a `TaskFailedError` with `kind="timeout"`.

`keep_batch_dir` controls cleanup after `map` completes:

- `"on_failure"` (default) deletes the batch directory if every run
  succeeded and keeps it otherwise.
- `"always"` keeps it.
- `"never"` deletes it.

The directory is always kept when `map` is interrupted. PyEns only ever
deletes directories it created.

### The batch directory

Each batch directory is self-describing, which is useful when a run goes
wrong:

```text
pyens-20260923-141502-1a2b3c4d/
    job.sh                 the submitted script; you can resubmit it by hand
    job_id                 the Grid Engine job ID
    manifest.json          run count, task ranges, processes per task
    model.pkl              the pickled model
    inputs/task-<t>.pkl    the runs of task t
    results/task-<t>.pkl   results of a finished task
    results/task-<t>.partial   results so far of a running or dead task
    results/task-<t>.error     why a task's worker failed to start
    logs/task-<t>.log      stdout and stderr of task t
```

---

## Example: BU SCC

```python
from pyens.backends import GridEngineBackend

def scc_backend(n_jobs: int, walltime: str) -> GridEngineBackend:
    return GridEngineBackend(
        walltime=walltime,
        work_dir="/projectnb/mygroup/me/pyens_batches",
        n_jobs=n_jobs,
        directives=["-P mygroup"],
        setup=["export OMP_NUM_THREADS=1"],
        max_concurrent=200,
    )
```

On SCC, remember:

- Your home directory has a small quota. Grid Engine writes job logs to your
  home directory unless told otherwise; `GridEngineBackend` always writes
  them into the batch directory.
- `h_rt` defaults to 12 hours on SCC if unset. `GridEngineBackend` requires
  an explicit `walltime`, so a short evaluation doesn't reserve 12-hour
  slots.

---

## Troubleshooting

**Every run fails with `TaskFailedError(kind="worker_error")`.** The worker
could not start on the compute node. Read the task log (`err.log_path`) and
`err.__cause__`. Common causes are a virtual environment that isn't on a shared
filesystem and a model module that isn't installed in it.

**Every run of a task fails with `kind="died"`, and the reason mentions
`exit_status=137`.** Grid Engine killed the task, usually for exceeding
`walltime` or its memory limit. Increase `walltime` or `n_jobs`, or request
more memory with a directive.

**The job stays pending for a long time.** Check `qstat -j <job_id>` for the
scheduler's reasons. Fewer slots per task, a shorter `walltime`, or
`max_concurrent` may help. Set `timeout` so the driver gives up
automatically.

**Runs fail with `RemoteError` instead of the exception class you expected.**
The exception class cannot be unpickled, usually because its `__init__`
requires keyword-only arguments. The information is preserved in
`RemoteError.attributes`. To get the original class back, give it a
`__reduce__` method that rebuilds it.

---

## Limitations

- **One queue wait per `map` call.** A backend with persistent workers (for
  example built on Parsl) would suit many small evaluations better. See
  [issue #5](https://github.com/arob5/PyEns/issues/5).
- **Grid Engine only.** PBS/Torque and Slurm use different array-job syntax
  and are not supported yet.
- **No automatic retries**, by design (see *Failed runs* above).
