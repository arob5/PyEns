"""Serialization utilities for EnsembleSpec and PartialSpec.

Specs are serialized to JSON with a metadata envelope, allowing them to be
saved to disk for reproducibility logging and exact reconstruction.

Value encoding
--------------
JSON-native types (``None``, ``bool``, ``int``, ``float``, ``str``, ``list``,
and ``dict`` with string keys and JSON-native values) are stored directly.
Complex types are wrapped in a tagged dict::

    {"__type__": "<tag>", ...type-specific fields...}

Built-in tags:

- ``"path"``    -- :class:`pathlib.Path`
- ``"ndarray"`` -- :class:`numpy.ndarray` (requires numpy; stored as
  dtype + shape + flattened data list)
- ``"dict"``    -- a plain ``dict`` whose keys include ``"__type__"``
  (escaped to avoid collision)
- ``"unknown"`` -- any type with no registered codec

The ``"unknown"`` tag stores a ``repr`` string and the Python class name.
It survives a round-trip for *logging* purposes only. :func:`spec_from_dict`
raises :class:`SerializationError` for unknown-tagged values when
``strict=True`` (the default).

Custom codecs
-------------
Register a :class:`ValueCodec` with :func:`register_codec` to teach PyEns how
to serialise and reconstruct your own types::

    class MyCodec:
        type_tag = "my_type"
        def can_encode(self, value): return isinstance(value, MyClass)
        def encode(self, value): return {"data": value.to_dict()}
        def decode(self, data): return MyClass.from_dict(data["data"])

    from pyens.serialize import register_codec
    register_codec(MyCodec())

PartialSpec serialization
-------------------------
``PartialSpec`` objects are serialized with a ``"FreeField"`` placeholder for
each free field and a ``"free_fields"`` list in the envelope. Calling
:func:`spec_from_dict` on such a dict returns a ``PartialSpec``.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pyens.axis import Axis
from pyens.fields import FieldSpec, Fixed, Grid
from pyens.spec import EnsembleSpec, PartialSpec

try:
    from importlib.metadata import version as _pkg_version
    _PYENS_VERSION = _pkg_version("pyens")
except Exception:
    _PYENS_VERSION = "unknown"


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class SerializationError(Exception):
    """Raised when a spec cannot be serialised or deserialised.

    Common causes:

    - A field value has no registered :class:`ValueCodec` and strict mode is
      enabled.
    - A JSON file produced by :func:`dump_spec` has been edited in an
      incompatible way.
    - An axis name collision is detected during deserialization.
    """


# ---------------------------------------------------------------------------
# ValueCodec protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ValueCodec(Protocol):
    """Protocol for extending PyEns serialization to custom value types.

    Implement this protocol and register your codec with
    :func:`register_codec`. The ``type_tag`` appears in the JSON as
    ``{"__type__": type_tag, ...}`` and must be unique across all registered
    codecs. It must not clash with built-in tags: ``"path"``, ``"ndarray"``,
    ``"dict"``, ``"unknown"``.

    Attributes:
        type_tag: Unique string identifier stored in the serialized JSON.

    Examples:
        >>> class MyCodec:
        ...     type_tag = "my_type"
        ...     def can_encode(self, v): return isinstance(v, MyClass)
        ...     def encode(self, v): return {"data": v.to_dict()}
        ...     def decode(self, d): return MyClass.from_dict(d["data"])
        >>> register_codec(MyCodec())
    """

    type_tag: str

    def can_encode(self, value: Any) -> bool:
        """Return ``True`` if this codec can encode *value*."""
        ...

    def encode(self, value: Any) -> dict[str, Any]:
        """Encode *value* to a JSON-serialisable dict.

        Args:
            value: The value to encode.

        Returns:
            A JSON-serialisable dict. Must not include the ``"__type__"`` key;
            the serializer adds it automatically.
        """
        ...

    def decode(self, data: dict[str, Any]) -> Any:
        """Reconstruct a value from the encoded dict.

        Args:
            data: The dict returned by :meth:`encode` (without ``"__type__"``).

        Returns:
            The reconstructed value.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in codecs
# ---------------------------------------------------------------------------

class _PathCodec:
    type_tag = "path"

    def can_encode(self, value: Any) -> bool:
        return isinstance(value, Path)

    def encode(self, value: Path) -> dict[str, Any]:
        return {"value": str(value)}

    def decode(self, data: dict[str, Any]) -> Path:
        return Path(data["value"])


class _NumpyCodec:
    type_tag = "ndarray"

    def can_encode(self, value: Any) -> bool:
        try:
            import numpy as np  # noqa: PLC0415
            return isinstance(value, np.ndarray)
        except ImportError:
            return False

    def encode(self, value: Any) -> dict[str, Any]:
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.flatten().tolist(),
        }

    def decode(self, data: dict[str, Any]) -> Any:
        import numpy as np  # noqa: PLC0415
        return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])


_BUILTIN_CODECS: list[Any] = [_PathCodec(), _NumpyCodec()]

# User-registered codecs; prepended so they take priority over built-ins.
_user_codecs: list[Any] = []


def register_codec(codec: ValueCodec) -> None:
    """Register a custom value codec.

    Codecs registered later take priority over earlier ones and over all
    built-in codecs.

    Args:
        codec: An object implementing the :class:`ValueCodec` protocol.

    Examples:
        >>> register_codec(MyCodec())
    """
    _user_codecs.insert(0, codec)


def _all_codecs() -> list[Any]:
    return _user_codecs + _BUILTIN_CODECS


# ---------------------------------------------------------------------------
# Value encoding / decoding
# ---------------------------------------------------------------------------

def _encode_value(value: Any) -> Any:
    """Encode a Python value to a JSON-serialisable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]

    if isinstance(value, dict):
        encoded = {str(k): _encode_value(v) for k, v in value.items()}
        # Escape dicts that happen to contain the sentinel key so the decoder
        # can distinguish them from tagged objects.
        if "__type__" in value:
            return {"__type__": "dict", "data": encoded}
        return encoded

    for codec in _all_codecs():
        if codec.can_encode(value):
            return {"__type__": codec.type_tag, **codec.encode(value)}

    return {
        "__type__": "unknown",
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _decode_value(data: Any, *, strict: bool) -> Any:
    """Decode a value that was produced by :func:`_encode_value`.

    Args:
        data: The encoded representation.
        strict: If ``True``, raise :class:`SerializationError` for values
            tagged ``"unknown"``. If ``False``, return the stored repr string.
    """
    if data is None or isinstance(data, (bool, int, float, str)):
        return data

    if isinstance(data, list):
        return [_decode_value(v, strict=strict) for v in data]

    if isinstance(data, dict):
        if "__type__" not in data:
            return {k: _decode_value(v, strict=strict) for k, v in data.items()}

        type_tag = data["__type__"]
        payload = {k: v for k, v in data.items() if k != "__type__"}

        if type_tag == "dict":
            return {k: _decode_value(v, strict=strict) for k, v in payload["data"].items()}

        if type_tag == "unknown":
            if strict:
                raise SerializationError(
                    f"Cannot reconstruct value of type '{data.get('class', '?')}'. "
                    f"Register a ValueCodec for this type, or pass strict=False to "
                    f"retrieve the stored repr string instead."
                )
            return data.get("repr", "<unresolvable>")

        for codec in _all_codecs():
            if codec.type_tag == type_tag:
                return codec.decode(payload)

        raise SerializationError(
            f"No codec registered for type tag '{type_tag}'. "
            f"Register a ValueCodec for this tag or update PyEns."
        )

    return data


# ---------------------------------------------------------------------------
# Axis encoding / decoding
# ---------------------------------------------------------------------------

def _encode_axis(axis: Axis) -> dict[str, Any]:
    if axis._labels is not None:
        return {"name": axis.name, "labels": list(axis._labels)}
    return {"name": axis.name, "size": axis.size}


def _decode_axis(data: dict[str, Any]) -> Axis:
    name = data["name"]
    if "labels" in data:
        return Axis(name, labels=data["labels"])
    return Axis(name, size=data["size"])


# ---------------------------------------------------------------------------
# Field encoding / decoding
# ---------------------------------------------------------------------------

def _encode_field(field: FieldSpec) -> dict[str, Any]:
    if isinstance(field, Fixed):
        return {"field_type": "Fixed", "value": _encode_value(field._value)}
    if isinstance(field, Grid):
        return {
            "field_type": "Grid",
            "along": [ax.name for ax in field._axes],
            "values": _encode_value(field._values),
        }
    raise SerializationError(
        f"Cannot serialize field of type {type(field).__name__}. "
        f"Only Fixed and Grid are supported."
    )


def _decode_field(
    data: dict[str, Any],
    axes_by_name: dict[str, Axis],
    *,
    strict: bool,
) -> FieldSpec:
    field_type = data.get("field_type")

    if field_type == "Fixed":
        return Fixed(_decode_value(data["value"], strict=strict))

    if field_type == "Grid":
        axis_names: list[str] = data["along"]
        missing = [n for n in axis_names if n not in axes_by_name]
        if missing:
            raise SerializationError(
                f"Grid field references unknown axis name(s): {missing}. "
                f"Axes defined in spec: {list(axes_by_name.keys())}."
            )
        along = [axes_by_name[n] for n in axis_names]
        values = _decode_value(data["values"], strict=strict)
        return Grid(values, along=along)

    raise SerializationError(
        f"Unknown field_type: {field_type!r}. Expected 'Fixed' or 'Grid'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spec_to_dict(spec: EnsembleSpec | PartialSpec) -> dict[str, Any]:
    """Convert a spec to a JSON-serialisable dict.

    The returned dict contains a metadata envelope (PyEns version, Python
    version, creation timestamp) plus the full spec structure. Values that
    are JSON-native are stored directly; complex types are tagged for
    reconstruction. See the module docstring for the encoding rules.

    Args:
        spec: An :class:`~pyens.EnsembleSpec` or :class:`~pyens.PartialSpec`.

    Returns:
        A plain Python dict suitable for :func:`json.dumps` or
        :func:`dump_spec`.

    Examples:
        >>> import json
        >>> d = spec_to_dict(my_spec)
        >>> print(json.dumps(d, indent=2))
    """
    is_partial = isinstance(spec, PartialSpec)

    if is_partial:
        seen: dict[Axis, None] = {}
        for field in spec._base.values():
            for ax in field.axes:
                seen[ax] = None
        axes: tuple[Axis, ...] = tuple(seen.keys())
        base_fields: dict[str, FieldSpec] = spec._base
        free_names: set[str] = spec._free
    else:
        axes = spec.axes
        base_fields = spec._inputs
        free_names = set()

    encoded_fields: dict[str, Any] = {}
    for name, field in base_fields.items():
        encoded_fields[name] = _encode_field(field)
    for name in free_names:
        encoded_fields[name] = {"field_type": "FreeField"}

    envelope: dict[str, Any] = {
        "__pyens_spec__": True,
        "__spec_type__": "PartialSpec" if is_partial else "EnsembleSpec",
        "__pyens_version__": _PYENS_VERSION,
        "__created_at__": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "__python_version__": sys.version,
        "axes": [_encode_axis(ax) for ax in axes],
        "fields": encoded_fields,
    }
    if is_partial:
        envelope["free_fields"] = sorted(free_names)
    return envelope


def spec_from_dict(
    d: dict[str, Any],
    *,
    strict: bool = True,
) -> EnsembleSpec | PartialSpec:
    """Reconstruct a spec from a dict produced by :func:`spec_to_dict`.

    Args:
        d: A dict as returned by :func:`spec_to_dict` or loaded from a JSON
            file written by :func:`dump_spec`.
        strict: If ``True`` (default), raise :class:`SerializationError` when
            a value tagged as ``"unknown"`` is encountered — these cannot be
            fully reconstructed. If ``False``, substitute the stored repr
            string, allowing partial reconstruction for inspection.

    Returns:
        An :class:`~pyens.EnsembleSpec` or :class:`~pyens.PartialSpec`,
        depending on the ``__spec_type__`` field in *d*.

    Raises:
        SerializationError: If the dict is malformed, references an unknown
            codec tag, or (when ``strict=True``) contains ``"unknown"``-tagged
            values.

    Examples:
        >>> spec = spec_from_dict(d)
        >>> spec.n_runs
        6
    """
    if not d.get("__pyens_spec__"):
        raise SerializationError(
            "Dict does not appear to be a PyEns spec. "
            "Expected '__pyens_spec__' to be True."
        )

    spec_type = d.get("__spec_type__", "EnsembleSpec")

    axes_by_name: dict[str, Axis] = {}
    for axis_data in d.get("axes", []):
        ax = _decode_axis(axis_data)
        axes_by_name[ax.name] = ax

    fields_data: dict[str, dict] = d.get("fields", {})
    free_names: set[str] = set(d.get("free_fields", []))

    if spec_type == "PartialSpec":
        base_inputs: dict[str, FieldSpec] = {}
        for name, field_data in fields_data.items():
            if field_data.get("field_type") == "FreeField":
                continue
            base_inputs[name] = _decode_field(field_data, axes_by_name, strict=strict)
        return PartialSpec(base_inputs, free_field_names=free_names)

    inputs: dict[str, FieldSpec] = {}
    for name, field_data in fields_data.items():
        inputs[name] = _decode_field(field_data, axes_by_name, strict=strict)
    return EnsembleSpec(inputs=inputs)


def dump_spec(
    spec: EnsembleSpec | PartialSpec,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Write a spec to a JSON file.

    The file is written in UTF-8. Use :func:`load_spec` to read it back.

    Args:
        spec: The spec to serialise.
        path: Destination file path. The parent directory must already exist.
        indent: JSON indentation level (default ``2``). Pass ``None`` for
            compact single-line output.

    Raises:
        SerializationError: If any field value cannot be encoded and no
            matching :class:`ValueCodec` is registered.

    Examples:
        >>> dump_spec(my_spec, "runs/ensemble_spec.json")
    """
    d = spec_to_dict(spec)
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def load_spec(
    path: str | Path,
    *,
    strict: bool = True,
) -> EnsembleSpec | PartialSpec:
    """Read a spec from a JSON file written by :func:`dump_spec`.

    Args:
        path: Path to the JSON file.
        strict: Passed to :func:`spec_from_dict`. If ``True`` (default),
            raises :class:`SerializationError` for values that cannot be
            reconstructed.

    Returns:
        The deserialised :class:`~pyens.EnsembleSpec` or
        :class:`~pyens.PartialSpec`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        SerializationError: If the file cannot be parsed or a value cannot
            be reconstructed (when ``strict=True``).

    Examples:
        >>> spec = load_spec("runs/ensemble_spec.json")
    """
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    return spec_from_dict(d, strict=strict)
