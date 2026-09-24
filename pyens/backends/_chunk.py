"""Run a contiguous chunk of an ensemble and encode each outcome as a record.

This is the worker-side core shared by batch backends. A *record* is the
outcome of one run, encoded as a pickled 4-tuple of primitives::

    (index, ok, traceback_text, payload_bytes)

``payload_bytes`` is itself a pickle: the model's return value when ``ok`` is
true, otherwise a picklable exception (see
:func:`~pyens.backends.errors.portable_exception`). Nesting the payload
means the outer tuple can always be unpickled, so the driver learns which
run a record belongs to even when the payload itself cannot be unpickled.

Records are written to a file as *frames*: an 8-byte big-endian length
followed by that many bytes. A worker that dies partway leaves a file whose
complete frames are still readable; :func:`read_frames` stops at the first
truncated one.
"""

from __future__ import annotations

import os
import pickle
import struct
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, BinaryIO

from pyens.backends.errors import (
    RemoteError,
    attach_remote_traceback,
    format_traceback,
    portable_exception,
)

_HEADER = struct.Struct(">Q")
_PROTOCOL = pickle.HIGHEST_PROTOCOL


def encode_success(index: int, value: Any) -> bytes:
    """Encode a successful run, or an error record if *value* cannot be pickled.

    Args:
        index: Position of the run in the full ensemble.
        value: The model's return value.

    Returns:
        The encoded record.
    """
    try:
        payload = pickle.dumps(value, protocol=_PROTOCOL)
    except Exception as exc:
        tb = format_traceback(exc)
        err = RemoteError(
            RemoteError.from_exception(exc).type_name,
            f"model returned a {type(value).__module__}.{type(value).__qualname__} "
            f"that cannot be pickled: {exc}",
            tb,
        )
        return encode_failure(index, err, tb)
    return pickle.dumps((index, True, None, payload), protocol=_PROTOCOL)


def encode_failure(index: int, exc: BaseException, traceback: str | None = None) -> bytes:
    """Encode a failed run, making the exception portable first.

    Args:
        index: Position of the run in the full ensemble.
        exc: The exception raised for the run.
        traceback: Pre-formatted traceback text; formatted from *exc* if
            omitted.

    Returns:
        The encoded record.
    """
    if traceback is None:
        traceback = format_traceback(exc)
    portable = portable_exception(exc, traceback)
    payload = pickle.dumps(portable, protocol=_PROTOCOL)
    return pickle.dumps((index, False, traceback, payload), protocol=_PROTOCOL)


def capture_call(fn: Callable[..., Any], index: int, inputs: dict[str, Any]) -> bytes:
    """Call ``fn(**inputs)`` and encode the outcome as a record.

    Exceptions derived from :class:`Exception`, and :class:`SystemExit` (a
    model calling ``sys.exit``), are recorded as the run's failure instead
    of ending the task.

    Args:
        fn: The model callable.
        index: Position of the run in the full ensemble.
        inputs: Keyword arguments for this run.

    Returns:
        The encoded record.
    """
    try:
        value = fn(**inputs)
    except (Exception, SystemExit) as exc:
        return encode_failure(index, exc)
    return encode_success(index, value)


def decode_record(record: bytes) -> tuple[int, Any]:
    """Decode a record into ``(index, output)``.

    Exceptions get the worker's traceback attached as ``__cause__``. A
    payload that cannot be unpickled here (for example because a class is
    missing on the driver) becomes a :class:`RemoteError` for that run only.

    Args:
        record: Bytes produced by :func:`encode_success`,
            :func:`encode_failure` or :func:`capture_call`.

    Returns:
        The run's index and its output (a value or an exception instance).
    """
    index, ok, traceback, payload = pickle.loads(record)
    try:
        value = pickle.loads(payload)
    except Exception as exc:
        what = "output" if ok else "exception"
        detail = f"\nWorker traceback:\n{traceback}" if traceback else ""
        return index, RemoteError(
            RemoteError.from_exception(exc).type_name,
            f"the run's {what} could not be unpickled on the driver: {exc}",
            format_traceback(exc) + detail,
        )
    if not ok and isinstance(value, BaseException):
        attach_remote_traceback(value, traceback or "")
    return index, value


def write_frame(stream: BinaryIO, record: bytes) -> None:
    """Append one length-prefixed frame to *stream* and flush it."""
    stream.write(_HEADER.pack(len(record)))
    stream.write(record)
    stream.flush()


def read_frames(source: str | os.PathLike[str] | bytes) -> tuple[list[bytes], bool]:
    """Read every complete frame from a file, or from its contents.

    Args:
        source: A file written with :func:`write_frame`, or its bytes.

    Returns:
        ``(frames, complete)``: the frames in file order, and ``False`` if the
        data ended in the middle of a frame (the writer died mid-write).
    """
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    frames: list[bytes] = []
    pos = 0
    while pos < len(data):
        if pos + _HEADER.size > len(data):
            return frames, False
        (length,) = _HEADER.unpack_from(data, pos)
        start = pos + _HEADER.size
        if start + length > len(data):
            return frames, False
        frames.append(data[start:start + length])
        pos = start + length
    return frames, True


def run_chunk(
    fn: Callable[..., Any] | None,
    start: int,
    inputs: Sequence[dict[str, Any]],
    write: Callable[[bytes], None],
    *,
    processes: int = 1,
    model_path: str | os.PathLike[str] | None = None,
) -> None:
    """Run ``inputs`` and pass each encoded record to *write* as it completes.

    With ``processes == 1`` the runs execute serially in this process, in
    order. With more, they run in a :class:`ProcessPoolExecutor` whose
    workers load the model once from *model_path*, and records are written
    in completion order (each carries its index).

    If a pool worker dies hard (for example ``os._exit`` or a segfault), the
    pool breaks and every run not yet finished is recorded with the
    resulting ``BrokenProcessPool`` exception.

    Args:
        fn: The model callable (used when ``processes == 1``).
        start: Ensemble index of ``inputs[0]``.
        inputs: The runs of this chunk.
        write: Called with each encoded record.
        processes: Number of worker processes.
        model_path: Pickled model file, required when ``processes > 1``.
    """
    if processes <= 1:
        if fn is None:
            raise ValueError("fn is required when processes == 1")
        for offset, run_inputs in enumerate(inputs):
            write(capture_call(fn, start + offset, run_inputs))
        return
    if model_path is None:
        raise ValueError("model_path is required when processes > 1")
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=_load_child_model,
        initargs=(os.fspath(model_path),),
    ) as pool:
        futures = {
            pool.submit(_child_call, start + offset, run_inputs): start + offset
            for offset, run_inputs in enumerate(inputs)
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = encode_failure(futures[future], exc)
            write(record)


def iter_decoded(frames: Sequence[bytes]) -> Iterator[tuple[int, Any]]:
    """Decode frames, skipping any whose outer tuple is unreadable."""
    for frame in frames:
        try:
            yield decode_record(frame)
        except Exception:
            continue


_CHILD_MODEL: Callable[..., Any] | None = None


def _load_child_model(model_path: str) -> None:
    global _CHILD_MODEL
    with open(model_path, "rb") as f:
        _CHILD_MODEL = pickle.load(f)


def _child_call(index: int, inputs: dict[str, Any]) -> bytes:
    assert _CHILD_MODEL is not None
    return capture_call(_CHILD_MODEL, index, inputs)
