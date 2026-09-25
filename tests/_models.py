"""Module-level model callables and exceptions shared by the backend tests.

Backends that run the model in another process pickle the callable by
reference, so these must live in an importable module rather than in a test
file's local scope.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any


class KwOnlyError(RuntimeError):
    """Mimics pySIPNET's ``SIPNETRunError``: pickles, but cannot unpickle."""

    def __init__(self, message: str, *, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class Unpicklable:
    """A return value that cannot be pickled."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


def add(x: Any, y: Any) -> Any:
    return x + y


def sometimes_fails(x: int, y: int) -> int:
    if x < 0:
        raise ValueError(f"negative x: {x}")
    return x + y


def always_raises(x: int) -> int:
    raise RuntimeError("deliberate failure")


def kw_only_crash(x: int) -> int:
    if x == 1:
        raise KwOnlyError("sipnet exited 1", returncode=1, stderr="missing param")
    return x


def unpicklable_output(x: int) -> Any:
    if x == 1:
        return Unpicklable()
    return x


def slow_add(x: int, y: int, delay: float = 0.0) -> int:
    time.sleep(delay)
    return x + y


def die_at(x: int, die: int) -> int:
    """Kill the whole worker process (not just raise) when ``x == die``."""
    if x == die:
        os._exit(3)
    return x


def pid_of(x: int) -> int:
    return os.getpid()


class BrokenOnLoad:
    """A model that pickles fine but raises when a worker unpickles it."""

    def __call__(self, x: int) -> int:
        return x

    def __reduce__(self) -> Any:
        return (_raise_on_load, ())


def _raise_on_load() -> None:
    raise ImportError("simulated missing dependency on the compute node")


def return_kw_only_error(x: int) -> Any:
    """Return (not raise) an exception that cannot be unpickled on the driver."""
    return KwOnlyError("returned", returncode=x, stderr="")


def getenv_pair() -> tuple[str | None, str | None]:
    return os.environ.get("PYENS_TEST_VAR"), os.environ.get("PYENS_SETUP_VAR")


def by_mode(mode: str, x: int) -> Any:
    """Behave according to *mode*; used by the cluster checks."""
    if mode == "ok":
        return x
    if mode == "kwonly":
        raise KwOnlyError("simulated SIPNET failure", returncode=2, stderr="bad param")
    if mode == "unpicklable":
        return Unpicklable()
    if mode == "exit":
        os._exit(5)
    if mode == "sleep":
        time.sleep(x)
        return x
    raise ValueError(f"unknown mode {mode!r}")


def host_pid(x: int, delay: float) -> tuple[int, str, int]:
    """Sleep, then report which host and process ran the run."""
    time.sleep(delay)
    return x, os.uname().nodename, os.getpid()


def exits_at(x: int, at: int) -> int:
    if x == at:
        raise SystemExit(4)
    return x


def site_series_total(series: Any, scale: float) -> tuple[Any, float]:
    """Return a DataArray slice's ``site`` label and its scaled sum."""
    return series["site"].item(), float(series.sum()) * scale
