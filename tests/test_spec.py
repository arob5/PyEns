"""Tests for pyens.spec: EnsembleSpec and PartialSpec."""

from __future__ import annotations

import pytest

from pyens import Axis, EnsembleSpec, Fixed, Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect(spec: EnsembleSpec) -> list[tuple[dict, dict]]:
    """Collect all (inputs, coordinate) pairs from a spec."""
    return list(spec.iter_runs())


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestEnsembleSpecConstruction:
    def test_empty_inputs_rejected(self):
        with pytest.raises(ValueError, match="at least one field"):
            EnsembleSpec(inputs={})

    def test_single_fixed_field(self):
        spec = EnsembleSpec(inputs={"x": Fixed(42)})
        assert spec.n_runs == 1
        assert spec.field_names == ("x",)
        assert spec.axes == ()

    def test_single_grid(self):
        ax = Axis("a", size=5)
        spec = EnsembleSpec(inputs={"x": Grid(list(range(5)), along=ax)})
        assert spec.n_runs == 5
        assert ax in spec.axes


# ---------------------------------------------------------------------------
# Run counts and coordinate structure
# ---------------------------------------------------------------------------

class TestRunCounts:
    def test_fixed_contributes_no_runs_beyond_1(self):
        spec = EnsembleSpec(inputs={"k": Fixed("v")})
        assert spec.n_runs == 1

    def test_single_axis_n_runs(self):
        ax = Axis("site", size=4)
        spec = EnsembleSpec(inputs={"c": Grid(["a", "b", "c", "d"], along=ax)})
        assert spec.n_runs == 4

    def test_two_different_axes_cartesian_product(self):
        """Different Axis instances → Cartesian product."""
        ax1 = Axis("a", size=3)
        ax2 = Axis("b", size=4)
        spec = EnsembleSpec(inputs={
            "f1": Grid(list(range(3)), along=ax1),
            "f2": Grid(list(range(4)), along=ax2),
        })
        assert spec.n_runs == 12

    def test_same_axis_instance_zip_semantics(self):
        """Same Axis instance → zip (not product)."""
        ax = Axis("site", size=3)
        spec = EnsembleSpec(inputs={
            "climate": Grid(["c1", "c2", "c3"], along=ax),
            "ic":      Grid(["i1", "i2", "i3"], along=ax),
        })
        assert spec.n_runs == 3

    def test_structurally_equal_axes_zip_semantics(self):
        """Two distinct but structurally equal Axis objects → zip (not product)."""
        ax1 = Axis("site", size=3)
        ax2 = Axis("site", size=3)
        assert ax1 is not ax2
        spec = EnsembleSpec(inputs={
            "climate": Grid(["c1", "c2", "c3"], along=ax1),
            "ic":      Grid(["i1", "i2", "i3"], along=ax2),
        })
        assert spec.n_runs == 3

    def test_conflicting_axis_names_raises(self):
        """Same axis name but different structure → ValueError."""
        ax1 = Axis("site", size=3)
        ax2 = Axis("site", size=4)
        with pytest.raises(ValueError, match="two axes named 'site'"):
            EnsembleSpec(inputs={
                "climate": Grid(["c1", "c2", "c3"], along=ax1),
                "ic":      Grid(["i1", "i2", "i3", "i4"], along=ax2),
            })

    def test_three_independent_axes(self):
        ax1 = Axis("p", size=2)
        ax2 = Axis("s", size=3)
        ax3 = Axis("m", size=5)
        spec = EnsembleSpec(inputs={
            "params":  Grid(["p0", "p1"], along=ax1),
            "climate": Grid(["c0", "c1", "c2"], along=ax2),
            "ic":      Grid(list(range(5)), along=ax3),
        })
        assert spec.n_runs == 2 * 3 * 5

    def test_fixed_plus_grid(self):
        ax = Axis("x", size=4)
        spec = EnsembleSpec(inputs={
            "fixed":  Fixed(99),
            "varied": Grid([10, 20, 30, 40], along=ax),
        })
        assert spec.n_runs == 4


# ---------------------------------------------------------------------------
# iter_runs: input values
# ---------------------------------------------------------------------------

class TestIterRuns:
    def test_fixed_value_same_every_run(self):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={
            "const": Fixed("hello"),
            "v":     Grid(["a", "b", "c"], along=ax),
        })
        runs = collect(spec)
        assert all(r[0]["const"] == "hello" for r in runs)

    def test_grid_values_in_order(self):
        ax = Axis("x", size=3)
        spec = EnsembleSpec(inputs={"v": Grid([10, 20, 30], along=ax)})
        values = [r[0]["v"] for r in collect(spec)]
        assert values == [10, 20, 30]

    def test_two_aligned_grids(self):
        ax = Axis("site", labels=["s1", "s2"])
        spec = EnsembleSpec(inputs={
            "climate": Grid(["c1", "c2"], along=ax),
            "ic":      Grid(["i1", "i2"], along=ax),
        })
        runs = collect(spec)
        assert len(runs) == 2
        # s1 run: climate=c1, ic=i1
        assert runs[0][0] == {"climate": "c1", "ic": "i1"}
        # s2 run: climate=c2, ic=i2
        assert runs[1][0] == {"climate": "c2", "ic": "i2"}

    def test_cartesian_product_values(self):
        ax1 = Axis("a", size=2)
        ax2 = Axis("b", size=3)
        spec = EnsembleSpec(inputs={
            "x": Grid([0, 1], along=ax1),
            "y": Grid([10, 20, 30], along=ax2),
        })
        runs = collect(spec)
        inputs_list = [r[0] for r in runs]
        # All combinations of x ∈ {0,1} and y ∈ {10,20,30}
        expected = [
            {"x": 0, "y": 10}, {"x": 0, "y": 20}, {"x": 0, "y": 30},
            {"x": 1, "y": 10}, {"x": 1, "y": 20}, {"x": 1, "y": 30},
        ]
        assert inputs_list == expected

    def test_multi_axis_grid_value_at(self):
        sites = Axis("site", labels=["s1", "s2"])
        members = Axis("member", size=2)
        values = [["s1m0", "s1m1"], ["s2m0", "s2m1"]]
        spec = EnsembleSpec(inputs={
            "ic": Grid(values, along=[sites, members]),
        })
        runs = collect(spec)
        ic_values = [r[0]["ic"] for r in runs]
        assert ic_values == ["s1m0", "s1m1", "s2m0", "s2m1"]


# ---------------------------------------------------------------------------
# iter_runs: coordinates
# ---------------------------------------------------------------------------

class TestCoordinates:
    def test_coordinate_uses_axis_name(self):
        ax = Axis("site", labels=["harvard", "niwot"])
        spec = EnsembleSpec(inputs={"c": Grid(["c1", "c2"], along=ax)})
        coords = [r[1] for r in collect(spec)]
        assert coords[0] == {"site": "harvard"}
        assert coords[1] == {"site": "niwot"}

    def test_coordinate_integer_labels(self):
        ax = Axis("member", size=3)
        spec = EnsembleSpec(inputs={"v": Grid([0, 1, 2], along=ax)})
        coords = [r[1] for r in collect(spec)]
        assert coords == [{"member": 0}, {"member": 1}, {"member": 2}]

    def test_fixed_field_not_in_coordinate(self):
        ax = Axis("x", size=2)
        spec = EnsembleSpec(inputs={"f": Fixed(99), "g": Grid([1, 2], along=ax)})
        for _, coord in collect(spec):
            assert "f" not in coord
            assert "x" in coord

    def test_no_axes_gives_empty_coordinate(self):
        spec = EnsembleSpec(inputs={"k": Fixed("v")})
        runs = collect(spec)
        assert len(runs) == 1
        assert runs[0][1] == {}


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_describe_contains_n_runs(self):
        ax = Axis("site", size=5)
        spec = EnsembleSpec(inputs={"c": Grid(list(range(5)), along=ax)})
        desc = spec.describe()
        assert "5 runs" in desc

    def test_describe_lists_axes(self):
        ax = Axis("myaxis", size=3)
        spec = EnsembleSpec(inputs={"v": Grid([1, 2, 3], along=ax)})
        desc = spec.describe()
        assert "myaxis" in desc

    def test_describe_lists_field_names(self):
        ax = Axis("x", size=2)
        spec = EnsembleSpec(inputs={"alpha": Fixed(1), "beta": Grid([1, 2], along=ax)})
        desc = spec.describe()
        assert "alpha" in desc
        assert "beta" in desc

    def test_singular_run(self):
        spec = EnsembleSpec(inputs={"k": Fixed(0)})
        assert "1 run" in spec.describe()
        assert "1 runs" not in spec.describe()


# ---------------------------------------------------------------------------
# from_runs()
# ---------------------------------------------------------------------------

class TestFromRuns:
    def test_basic(self):
        spec = EnsembleSpec.from_runs(
            {"x": 1, "y": "a"},
            {"x": 2, "y": "b"},
            {"x": 3, "y": "c"},
        )
        assert spec.n_runs == 3

    def test_values_correct(self):
        spec = EnsembleSpec.from_runs({"v": 10}, {"v": 20})
        values = [r[0]["v"] for r in collect(spec)]
        assert values == [10, 20]

    def test_coordinate_is_integer(self):
        spec = EnsembleSpec.from_runs({"v": "a"}, {"v": "b"})
        coords = [r[1] for r in collect(spec)]
        assert coords[0]["run"] == 0
        assert coords[1]["run"] == 1

    def test_no_runs_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EnsembleSpec.from_runs()

    def test_inconsistent_keys_rejected(self):
        with pytest.raises(ValueError, match="keys"):
            EnsembleSpec.from_runs({"x": 1}, {"y": 2})


# ---------------------------------------------------------------------------
# freeze() and PartialSpec
# ---------------------------------------------------------------------------

class TestFreeze:
    def setup_method(self):
        self.ax = Axis("site", size=3)
        self.param_ax = Axis("param", size=2)
        self.spec = EnsembleSpec(inputs={
            "parameters": Fixed("base"),
            "climate":    Grid(["c0", "c1", "c2"], along=self.ax),
        })

    def test_freeze_returns_bound_spec(self):
        from pyens import PartialSpec
        bound = self.spec.freeze(free=["parameters"])
        assert isinstance(bound, PartialSpec)

    def test_bound_spec_free_names(self):
        bound = self.spec.freeze(free=["parameters"])
        assert "parameters" in bound.free_field_names

    def test_freeze_unknown_field_raises(self):
        with pytest.raises(ValueError, match="unknown field"):
            self.spec.freeze(free=["nonexistent"])

    def test_bound_spec_call_with_fixed(self):
        bound = self.spec.freeze(free=["parameters"])
        runnable = bound(parameters="new_params")
        # climate axis (size 3) × no param axis (Fixed) → 3 runs
        assert runnable.n_runs == 3
        params = [r[0]["parameters"] for r in collect(runnable)]
        assert all(p == "new_params" for p in params)

    def test_bound_spec_call_with_grid_expands_runs(self):
        bound = self.spec.freeze(free=["parameters"])
        runnable = bound(parameters=Grid(["p0", "p1"], along=self.param_ax))
        # site (3) × param (2) = 6 runs
        assert runnable.n_runs == 6

    def test_bound_spec_missing_free_raises(self):
        bound = self.spec.freeze(free=["parameters"])
        with pytest.raises(ValueError, match="missing"):
            bound()  # forgot to supply 'parameters'

    def test_bound_spec_extra_field_raises(self):
        bound = self.spec.freeze(free=["parameters"])
        with pytest.raises(ValueError, match="unexpected"):
            bound(parameters="p", unknown_field="oops")

    def test_freeze_multiple_fields(self):
        spec = EnsembleSpec(inputs={
            "a": Fixed(1),
            "b": Fixed(2),
            "c": Grid([10, 20], along=Axis("x", size=2)),
        })
        bound = spec.freeze(free=["a", "b"])
        runnable = bound(a="new_a", b="new_b")
        assert runnable.n_runs == 2
        for inputs, _ in collect(runnable):
            assert inputs["a"] == "new_a"
            assert inputs["b"] == "new_b"


# ---------------------------------------------------------------------------
# sel()
# ---------------------------------------------------------------------------

class TestSel:
    def setup_method(self):
        self.sites   = Axis("site",   labels=["A", "B", "C"])
        self.members = Axis("member", size=3)
        self.spec = EnsembleSpec(inputs={
            "climate": Grid(["c_A", "c_B", "c_C"], along=self.sites),
            "ic":      Grid([10, 20, 30],           along=self.members),
            "params":  Fixed("base"),
        })

    def test_returns_correct_inputs(self):
        result = self.spec.sel(site="B", member=2)
        assert result["climate"] == "c_B"
        assert result["ic"] == 30
        assert result["params"] == "base"

    def test_first_coordinate(self):
        result = self.spec.sel(site="A", member=0)
        assert result["climate"] == "c_A"
        assert result["ic"] == 10

    def test_fixed_field_included(self):
        result = self.spec.sel(site="A", member=0)
        assert "params" in result
        assert result["params"] == "base"

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="unknown axis"):
            self.spec.sel(site="A", member=0, unknown="x")

    def test_missing_axis_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            self.spec.sel(site="A")  # member not specified

    def test_bad_label_raises(self):
        with pytest.raises(ValueError, match="not found in axis"):
            self.spec.sel(site="Z", member=0)

    def test_integer_label_lookup(self):
        """Integer-labeled axes are selected with integer label values."""
        result = self.spec.sel(site="A", member=1)
        assert result["ic"] == 20

    def test_no_axes_spec_sel_empty(self):
        """A spec with only Fixed fields has no axes; sel() with no args works."""
        spec = EnsembleSpec(inputs={"x": Fixed(99)})
        result = spec.sel()
        assert result == {"x": 99}

    def test_matches_iter_runs(self):
        """sel() should return the same inputs as the matching iter_runs() pair."""
        for inputs, coord in self.spec.iter_runs():
            selected = self.spec.sel(**coord)
            assert selected == inputs
