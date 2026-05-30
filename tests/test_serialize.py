"""Tests for pyens.serialize."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pyens import Axis, EnsembleSpec, Fixed, Grid, PartialSpec
from pyens.serialize import (
    SerializationError,
    ValueCodec,
    dump_spec,
    load_spec,
    register_codec,
    spec_from_dict,
    spec_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect(spec: EnsembleSpec) -> list[tuple[dict, dict]]:
    return list(spec.iter_runs())


def roundtrip(spec: EnsembleSpec | PartialSpec) -> EnsembleSpec | PartialSpec:
    return spec_from_dict(spec_to_dict(spec))


# ---------------------------------------------------------------------------
# Metadata envelope
# ---------------------------------------------------------------------------

class TestMetadataEnvelope:
    def test_pyens_spec_flag(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        d = spec_to_dict(spec)
        assert d["__pyens_spec__"] is True

    def test_spec_type_ensemble(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        d = spec_to_dict(spec)
        assert d["__spec_type__"] == "EnsembleSpec"

    def test_spec_type_partial(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        partial = spec.freeze(free=["x"])
        d = spec_to_dict(partial)
        assert d["__spec_type__"] == "PartialSpec"

    def test_created_at_present(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        d = spec_to_dict(spec)
        assert "__created_at__" in d
        assert "T" in d["__created_at__"]  # ISO 8601 datetime

    def test_python_version_present(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        d = spec_to_dict(spec)
        assert "__python_version__" in d

    def test_pyens_version_present(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        d = spec_to_dict(spec)
        assert "__pyens_version__" in d


# ---------------------------------------------------------------------------
# Axis round-trips
# ---------------------------------------------------------------------------

class TestAxisRoundtrip:
    def test_size_axis(self):
        ax = Axis("member", size=5)
        spec = EnsembleSpec(inputs={"x": Grid(list(range(5)), along=ax)})
        recovered = roundtrip(spec)
        assert recovered.axes[0].name == "member"
        assert recovered.axes[0].size == 5
        assert recovered.axes[0]._labels is None

    def test_labeled_axis(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={"x": Grid([1, 2, 3], along=ax)})
        recovered = roundtrip(spec)
        assert recovered.axes[0].labels == ("A", "B", "C")

    def test_multiple_axes(self):
        sites = Axis("site", labels=["A", "B"])
        members = Axis("member", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2], along=sites),
            "y": Grid([10, 20, 30], along=members),
        })
        recovered = roundtrip(spec)
        names = {ax.name for ax in recovered.axes}
        assert names == {"site", "member"}

    def test_shared_axis_serialised_once(self):
        """Two fields on the same axis → only one axis entry in JSON."""
        ax = Axis("site", labels=["A", "B"])
        spec = EnsembleSpec(inputs={
            "climate": Grid(["c1", "c2"], along=ax),
            "ic":      Grid(["i1", "i2"], along=ax),
        })
        d = spec_to_dict(spec)
        assert len(d["axes"]) == 1


# ---------------------------------------------------------------------------
# Fixed field round-trips
# ---------------------------------------------------------------------------

class TestFixedRoundtrip:
    def test_scalar_int(self):
        spec = EnsembleSpec(inputs={"k": Fixed(42)})
        recovered = roundtrip(spec)
        runs = collect(recovered)
        assert runs[0][0]["k"] == 42

    def test_scalar_float(self):
        spec = EnsembleSpec(inputs={"k": Fixed(3.14)})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["k"] == pytest.approx(3.14)

    def test_scalar_string(self):
        spec = EnsembleSpec(inputs={"k": Fixed("hello")})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["k"] == "hello"

    def test_scalar_none(self):
        spec = EnsembleSpec(inputs={"k": Fixed(None)})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["k"] is None

    def test_scalar_bool(self):
        spec = EnsembleSpec(inputs={"k": Fixed(True)})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["k"] is True

    def test_dict_value(self):
        val = {"aMax": 12.0, "k": 0.5, "label": "base"}
        spec = EnsembleSpec(inputs={"params": Fixed(val)})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["params"] == val

    def test_list_value(self):
        spec = EnsembleSpec(inputs={"v": Fixed([1, 2, 3])})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["v"] == [1, 2, 3]

    def test_nested_dict(self):
        val = {"outer": {"inner": 99}}
        spec = EnsembleSpec(inputs={"cfg": Fixed(val)})
        recovered = roundtrip(spec)
        assert collect(recovered)[0][0]["cfg"] == val


# ---------------------------------------------------------------------------
# Grid field round-trips
# ---------------------------------------------------------------------------

class TestGridRoundtrip:
    def test_single_axis_scalars(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={"x": Grid([10, 20, 30], along=ax)})
        recovered = roundtrip(spec)
        values = [r[0]["x"] for r in collect(recovered)]
        assert values == [10, 20, 30]

    def test_single_axis_dicts(self):
        ax = Axis("run", size=2)
        val = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        spec = EnsembleSpec(inputs={"params": Grid(val, along=ax)})
        recovered = roundtrip(spec)
        values = [r[0]["params"] for r in collect(recovered)]
        assert values == val

    def test_multi_axis(self):
        sites = Axis("site", labels=["A", "B"])
        members = Axis("member", size=2)
        values = [[1, 2], [3, 4]]
        spec = EnsembleSpec(inputs={"x": Grid(values, along=[sites, members])})
        recovered = roundtrip(spec)
        result = [r[0]["x"] for r in collect(recovered)]
        assert result == [1, 2, 3, 4]

    def test_run_count_preserved(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={"x": Grid([1, 2, 3], along=ax)})
        recovered = roundtrip(spec)
        assert recovered.n_runs == 3

    def test_coordinate_labels_preserved(self):
        ax = Axis("site", labels=["harvard", "niwot"])
        spec = EnsembleSpec(inputs={"x": Grid([1, 2], along=ax)})
        recovered = roundtrip(spec)
        coords = [r[1] for r in collect(recovered)]
        assert coords[0] == {"site": "harvard"}
        assert coords[1] == {"site": "niwot"}


# ---------------------------------------------------------------------------
# Mixed Fixed + Grid
# ---------------------------------------------------------------------------

class TestMixedFieldRoundtrip:
    def test_fixed_and_grid(self):
        ax = Axis("site", labels=["A", "B"])
        spec = EnsembleSpec(inputs={
            "params":  Fixed({"k": 0.1}),
            "climate": Grid(["dry", "wet"], along=ax),
        })
        recovered = roundtrip(spec)
        assert recovered.n_runs == 2
        runs = collect(recovered)
        assert all(r[0]["params"] == {"k": 0.1} for r in runs)
        assert [r[0]["climate"] for r in runs] == ["dry", "wet"]


# ---------------------------------------------------------------------------
# Path values
# ---------------------------------------------------------------------------

class TestPathValues:
    def test_path_roundtrip(self):
        path = Path("/data/climate/harvard_forest.nc")
        spec = EnsembleSpec(inputs={"data": Fixed(path)})
        recovered = roundtrip(spec)
        val = collect(recovered)[0][0]["data"]
        assert isinstance(val, Path)
        assert val == path

    def test_path_in_grid(self):
        ax = Axis("site", labels=["hf", "nr"])
        paths = [Path("/data/hf.nc"), Path("/data/nr.nc")]
        spec = EnsembleSpec(inputs={"data": Grid(paths, along=ax)})
        recovered = roundtrip(spec)
        values = [r[0]["data"] for r in collect(recovered)]
        assert values == paths
        assert all(isinstance(v, Path) for v in values)

    def test_path_in_json_is_readable(self):
        """The path tag is human-readable in the JSON."""
        spec = EnsembleSpec(inputs={"f": Fixed(Path("/some/file.nc"))})
        d = spec_to_dict(spec)
        raw = json.dumps(d)
        assert "/some/file.nc" in raw


# ---------------------------------------------------------------------------
# Unknown types
# ---------------------------------------------------------------------------

class TestUnknownTypes:
    def test_unknown_stored_as_repr(self):
        class Opaque:
            def __repr__(self): return "Opaque()"

        spec = EnsembleSpec(inputs={"x": Fixed(Opaque())})
        d = spec_to_dict(spec)
        field_val = d["fields"]["x"]["value"]
        assert field_val["__type__"] == "unknown"
        assert "Opaque" in field_val["repr"]

    def test_strict_raises_on_unknown(self):
        class Opaque:
            def __repr__(self): return "Opaque()"

        spec = EnsembleSpec(inputs={"x": Fixed(Opaque())})
        d = spec_to_dict(spec)
        with pytest.raises(SerializationError, match="Cannot reconstruct"):
            spec_from_dict(d, strict=True)

    def test_non_strict_returns_repr_string(self):
        class Opaque:
            def __repr__(self): return "Opaque(42)"

        spec = EnsembleSpec(inputs={"x": Fixed(Opaque())})
        d = spec_to_dict(spec)
        recovered = spec_from_dict(d, strict=False)
        val = collect(recovered)[0][0]["x"]
        assert isinstance(val, str)
        assert "Opaque" in val


# ---------------------------------------------------------------------------
# Custom codecs
# ---------------------------------------------------------------------------

class TestCustomCodec:
    def test_custom_codec_roundtrip(self):
        class Point:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y
            def __eq__(self, other: object) -> bool:
                return isinstance(other, Point) and self.x == other.x and self.y == other.y

        class PointCodec:
            type_tag = "point_test"
            def can_encode(self, v: object) -> bool:
                return isinstance(v, Point)
            def encode(self, v: Point) -> dict:
                return {"x": v.x, "y": v.y}
            def decode(self, d: dict) -> Point:
                return Point(d["x"], d["y"])

        register_codec(PointCodec())

        spec = EnsembleSpec(inputs={"loc": Fixed(Point(1.0, 2.0))})
        recovered = roundtrip(spec)
        val = collect(recovered)[0][0]["loc"]
        assert isinstance(val, Point)
        assert val == Point(1.0, 2.0)


# ---------------------------------------------------------------------------
# PartialSpec round-trips
# ---------------------------------------------------------------------------

class TestPartialSpecRoundtrip:
    def test_spec_type_is_partial(self):
        ax = Axis("site", size=2)
        spec = EnsembleSpec(inputs={
            "params":  Fixed("base"),
            "climate": Grid(["dry", "wet"], along=ax),
        })
        partial = spec.freeze(free=["params"])
        d = spec_to_dict(partial)
        assert d["__spec_type__"] == "PartialSpec"

    def test_free_fields_recorded(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        partial = spec.freeze(free=["x"])
        d = spec_to_dict(partial)
        assert "x" in d["free_fields"]
        assert d["fields"]["x"]["field_type"] == "FreeField"

    def test_roundtrip_returns_partial_spec(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        partial = spec.freeze(free=["x"])
        recovered = roundtrip(partial)
        assert isinstance(recovered, PartialSpec)

    def test_free_field_names_preserved(self):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        partial = spec.freeze(free=["x"])
        recovered = roundtrip(partial)
        assert "x" in recovered.free_field_names

    def test_partial_spec_callable_after_roundtrip(self):
        ax = Axis("site", size=2)
        spec = EnsembleSpec(inputs={
            "params":  Fixed("base"),
            "climate": Grid(["dry", "wet"], along=ax),
        })
        partial = spec.freeze(free=["params"])
        recovered: PartialSpec = roundtrip(partial)  # type: ignore[assignment]
        runnable = recovered(params="new_base")
        assert runnable.n_runs == 2


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_dump_and_load_roundtrip(self, tmp_path: Path):
        ax = Axis("site", labels=["A", "B"])
        spec = EnsembleSpec(inputs={
            "params":  Fixed({"k": 1.0}),
            "climate": Grid(["dry", "wet"], along=ax),
        })
        path = tmp_path / "spec.json"
        dump_spec(spec, path)
        recovered = load_spec(path)
        assert isinstance(recovered, EnsembleSpec)
        assert recovered.n_runs == 2
        runs = collect(recovered)
        assert runs[0][0]["params"] == {"k": 1.0}
        assert [r[0]["climate"] for r in runs] == ["dry", "wet"]

    def test_written_file_is_valid_json(self, tmp_path: Path):
        spec = EnsembleSpec(inputs={"x": Fixed(99)})
        path = tmp_path / "spec.json"
        dump_spec(spec, path)
        with path.open() as f:
            parsed = json.load(f)
        assert parsed["__pyens_spec__"] is True

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_spec(tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        with pytest.raises(Exception):
            load_spec(path)

    def test_load_non_spec_dict_raises(self, tmp_path: Path):
        path = tmp_path / "plain.json"
        path.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(SerializationError, match="does not appear to be"):
            load_spec(path)

    def test_partial_spec_file_roundtrip(self, tmp_path: Path):
        ax = Axis("site", size=3)
        spec = EnsembleSpec(inputs={
            "params":  Fixed("base"),
            "climate": Grid(["a", "b", "c"], along=ax),
        })
        partial = spec.freeze(free=["params"])
        path = tmp_path / "partial.json"
        dump_spec(partial, path)
        recovered = load_spec(path)
        assert isinstance(recovered, PartialSpec)
        assert "params" in recovered.free_field_names


# ---------------------------------------------------------------------------
# Convenience methods: spec.dump() / EnsembleSpec.load() / PartialSpec.load()
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_ensemble_spec_dump_and_load(self, tmp_path: Path):
        ax = Axis("site", labels=["A", "B"])
        spec = EnsembleSpec(inputs={
            "params":  Fixed({"k": 1.0}),
            "climate": Grid(["dry", "wet"], along=ax),
        })
        path = tmp_path / "spec.json"
        spec.dump(path)
        recovered = EnsembleSpec.load(path)
        assert isinstance(recovered, EnsembleSpec)
        assert recovered.n_runs == spec.n_runs

    def test_partial_spec_dump_and_load(self, tmp_path: Path):
        ax = Axis("site", size=2)
        spec = EnsembleSpec(inputs={
            "params":  Fixed("base"),
            "climate": Grid(["a", "b"], along=ax),
        })
        partial = spec.freeze(free=["params"])
        path = tmp_path / "partial.json"
        partial.dump(path)
        recovered = PartialSpec.load(path)
        assert isinstance(recovered, PartialSpec)
        assert "params" in recovered.free_field_names

    def test_ensemble_spec_load_rejects_partial(self, tmp_path: Path):
        spec = EnsembleSpec(inputs={"x": Fixed(1), "y": Fixed(2)})
        partial = spec.freeze(free=["x"])
        path = tmp_path / "partial.json"
        partial.dump(path)
        with pytest.raises(TypeError, match="PartialSpec"):
            EnsembleSpec.load(path)

    def test_partial_spec_load_rejects_ensemble(self, tmp_path: Path):
        spec = EnsembleSpec(inputs={"x": Fixed(1)})
        path = tmp_path / "spec.json"
        spec.dump(path)
        with pytest.raises(TypeError, match="EnsembleSpec"):
            PartialSpec.load(path)

    def test_dump_produces_same_file_as_dump_spec(self, tmp_path: Path):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={"v": Grid([1, 2, 3], along=ax)})
        path_method = tmp_path / "method.json"
        path_func = tmp_path / "func.json"
        spec.dump(path_method)
        dump_spec(spec, path_func)
        # Both files should have identical structure (timestamps will differ)
        import json
        d_method = json.load(path_method.open())
        d_func = json.load(path_func.open())
        assert d_method["axes"] == d_func["axes"]
        assert d_method["fields"] == d_func["fields"]
        assert d_method["__spec_type__"] == d_func["__spec_type__"]


# ---------------------------------------------------------------------------
# Correctness: recovered spec produces same runs as original
# ---------------------------------------------------------------------------

class TestRunEquivalence:
    def test_same_runs_simple(self):
        ax = Axis("site", labels=["A", "B", "C"])
        spec = EnsembleSpec(inputs={
            "params":  Fixed({"k": 0.3}),
            "climate": Grid(["c1", "c2", "c3"], along=ax),
        })
        original_runs = collect(spec)
        recovered_runs = collect(roundtrip(spec))
        assert original_runs == recovered_runs

    def test_same_runs_cartesian(self):
        sites = Axis("site", labels=["A", "B"])
        members = Axis("member", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([1, 2], along=sites),
            "y": Grid([10, 20, 30], along=members),
        })
        original_runs = collect(spec)
        recovered_runs = collect(roundtrip(spec))
        assert original_runs == recovered_runs

    def test_from_runs_roundtrip(self):
        spec = EnsembleSpec.from_runs(
            {"x": 1.0, "y": "a"},
            {"x": 2.0, "y": "b"},
            {"x": 3.0, "y": "c"},
        )
        original_runs = collect(spec)
        recovered_runs = collect(roundtrip(spec))
        assert original_runs == recovered_runs
