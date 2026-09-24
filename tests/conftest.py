"""Shared pytest fixtures."""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_GE = Path(__file__).resolve().parent / "fake_ge.py"


class FakeGridEngine:
    """Handle on the fake scheduler installed by the ``fake_ge`` fixture."""

    def __init__(self, state: Path, work_dir: Path) -> None:
        self.state = state
        self.work_dir = work_dir

    def set_faults(self, **faults: object) -> None:
        """Replace the fault configuration (see ``tests/fake_ge.py``)."""
        (self.state / "faults.json").write_text(json.dumps(faults))

    def calls(self, command: str | None = None) -> list[list[str]]:
        """Every command invocation so far, optionally only one command's."""
        log = self.state / "calls.log"
        if not log.exists():
            return []
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return [c for c in calls if command is None or c[0] == command]

    def scripts(self) -> list[str]:
        """Text of every submitted job script, in submission order."""
        return [p.read_text() for p in sorted(self.state.glob("jobs/*/submitted.sh"))]

    def queued(self) -> list[Path]:
        """State files of tasks still queued or running."""
        return sorted(self.state.glob("jobs/*/task-*.state"))

    def batch_dirs(self) -> list[Path]:
        return sorted(p for p in self.work_dir.iterdir() if p.is_dir())

    def cleanup(self) -> None:
        for pid_file in self.state.glob("jobs/*/task-*.pid"):
            _kill_group(pid_file.read_text())
        pids = self.state / "pids"
        if pids.exists():
            for pid in pids.read_text().split():
                _kill_group(pid)


def _kill_group(pid: str) -> None:
    try:
        os.killpg(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError):
        pass


@pytest.fixture
def fake_ge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGridEngine]:
    """Put fake ``qsub``/``qstat``/``qdel``/``qacct`` on ``PATH``.

    Also changes to the repository root, because tasks start in the driver's
    working directory and must be able to import ``tests._models``.
    """
    bin_dir = tmp_path / "bin"
    state = tmp_path / "ge_state"
    work_dir = tmp_path / "batches"
    for d in (bin_dir, state, work_dir):
        d.mkdir()
    for command in ("qsub", "qstat", "qdel", "qacct"):
        wrapper = bin_dir / command
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_GE}" {command} "$@"\n'
        )
        wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GE_STATE", str(state))
    monkeypatch.chdir(REPO_ROOT)
    fake = FakeGridEngine(state, work_dir)
    yield fake
    fake.cleanup()
