"""Entry point run by each Grid Engine array task.

Invoked by the job script as::

    python -m pyens.backends._gridengine_worker <batch_dir> <task_id>

It loads the model and this task's inputs from the batch directory, runs
them (serially, or across ``manifest["processes"]`` worker processes), and
appends one record per run to ``results/task-<t>.partial`` as each run
finishes. When all runs are done the file is renamed to
``results/task-<t>.pkl``, which is how the driver knows the task completed.

If setup fails before any run starts (the model's module cannot be
imported, for example), the exception is written to
``results/task-<t>.error`` and the process exits with status 1.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from pyens.backends._batch import BatchDir
from pyens.backends._chunk import run_chunk, write_frame
from pyens.backends.errors import format_traceback, portable_exception


def main(argv: Sequence[str] | None = None) -> int:
    """Run one array task. Returns the process exit status.

    Args:
        argv: ``[batch_dir, task_id]``. Defaults to ``sys.argv[1:]``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m pyens.backends._gridengine_worker "
              "<batch_dir> <task_id>", file=sys.stderr)
        return 2
    batch = BatchDir(args[0])
    try:
        task = int(args[1])
    except ValueError:
        print(f"pyens worker: invalid task id {args[1]!r}", file=sys.stderr)
        return 2

    try:
        manifest = batch.read_manifest()
        processes = int(manifest.get("processes", 1))
        start, inputs = batch.load_inputs(task)
        fn = batch.load_model()
    except Exception as exc:
        tb = format_traceback(exc)
        print(f"pyens worker: setup failed for task {task}:\n{tb}", file=sys.stderr)
        batch.write_task_error(task, portable_exception(exc, tb), tb)
        return 1

    print(f"pyens worker: task {task}: {len(inputs)} runs from index {start}, "
          f"{processes} process(es), pid {os.getpid()}", file=sys.stderr, flush=True)
    partial = batch.partial_path(task)
    with open(partial, "wb") as stream:
        run_chunk(
            fn,
            start,
            inputs,
            lambda record: write_frame(stream, record),
            processes=processes,
            model_path=batch.model_path,
        )
    os.replace(partial, batch.result_path(task))
    print(f"pyens worker: task {task}: done", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
