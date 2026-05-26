"""Tests for pyens.fields: Fixed and Grid."""

from __future__ import annotations

import pytest

from pyens import Axis, Fixed, Grid


class TestFixed:
    def test_returns_value(self):
        f = Fixed(99)
        assert f.value_at({}) == 99

    def test_axes_empty(self):
        assert Fixed("anything").axes == ()

    def test_value_at_ignores_coords(self):
        ax = Axis("x", size=3)
        f = Fixed("constant")
        assert f.value_at({ax: 0}) == "constant"
        assert f.value_at({ax: 2}) == "constant"

    def test_repr(self):
        assert "Fixed" in repr(Fixed(42))


class TestGridSingleAxis:
    def setup_method(self):
        self.ax = Axis("site", labels=["s1", "s2", "s3"])
        self.g = Grid(["a", "b", "c"], along=self.ax)

    def test_axes(self):
        assert self.g.axes == (self.ax,)

    def test_value_at(self):
        assert self.g.value_at({self.ax: 0}) == "a"
        assert self.g.value_at({self.ax: 2}) == "c"

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="length 2"):
            Grid(["x", "y"], along=self.ax)  # axis size is 3

    def test_non_sequence_values(self):
        with pytest.raises(ValueError, match="support len()"):
            Grid(42, along=self.ax)

    def test_repr(self):
        r = repr(self.g)
        assert "Grid" in r
        assert "site" in r


class TestGridMultiAxis:
    def setup_method(self):
        self.sites = Axis("site", labels=["s1", "s2"])
        self.members = Axis("member", size=3)
        # Shape: (2 sites) × (3 members) — nested list
        self.values = [
            ["s1m0", "s1m1", "s1m2"],
            ["s2m0", "s2m1", "s2m2"],
        ]
        self.g = Grid(self.values, along=[self.sites, self.members])

    def test_axes(self):
        assert self.g.axes == (self.sites, self.members)

    def test_value_at(self):
        assert self.g.value_at({self.sites: 0, self.members: 0}) == "s1m0"
        assert self.g.value_at({self.sites: 1, self.members: 2}) == "s2m2"

    def test_top_level_shape_mismatch(self):
        with pytest.raises(ValueError, match="length 1"):
            # Only 1 row for 2-site axis
            Grid([["a", "b", "c"]], along=[self.sites, self.members])

    def test_repr(self):
        r = repr(self.g)
        assert "Grid" in r
        assert "site" in r
        assert "member" in r


class TestGridRequiresAxes:
    def test_empty_along_list(self):
        with pytest.raises(ValueError, match="at least one Axis"):
            Grid(["a", "b"], along=[])


class TestGridDictValues:
    """Tests for label-keyed dict construction of Grid."""

    def setup_method(self):
        self.sites = Axis("site", labels=["A", "B", "C"])

    def test_dict_produces_correct_values(self):
        g = Grid({"A": 10, "B": 20, "C": 30}, along=self.sites)
        assert g.value_at({self.sites: 0}) == 10
        assert g.value_at({self.sites: 1}) == 20
        assert g.value_at({self.sites: 2}) == 30

    def test_dict_order_independent(self):
        """Keys in any order should yield values ordered by axis labels."""
        g = Grid({"C": "third", "A": "first", "B": "second"}, along=self.sites)
        assert g.value_at({self.sites: 0}) == "first"
        assert g.value_at({self.sites: 1}) == "second"
        assert g.value_at({self.sites: 2}) == "third"

    def test_dict_equivalent_to_positional(self):
        """Dict and positional construction produce identical grids."""
        g_pos  = Grid(["x", "y", "z"], along=self.sites)
        g_dict = Grid({"A": "x", "B": "y", "C": "z"}, along=self.sites)
        for i in range(3):
            assert g_pos.value_at({self.sites: i}) == g_dict.value_at({self.sites: i})

    def test_dict_missing_label_raises(self):
        with pytest.raises(ValueError, match="missing values for labels"):
            Grid({"A": 1, "B": 2}, along=self.sites)  # C is missing

    def test_dict_extra_label_raises(self):
        with pytest.raises(ValueError, match="unexpected labels"):
            Grid({"A": 1, "B": 2, "C": 3, "D": 4}, along=self.sites)

    def test_dict_integer_labeled_axis_raises(self):
        """Dict form requires explicit labels, not integer-only axis."""
        int_axis = Axis("member", size=3)
        with pytest.raises(ValueError, match="integer-indexed only"):
            Grid({0: "a", 1: "b", 2: "c"}, along=int_axis)

    def test_dict_multi_axis_raises(self):
        """Dict form is not supported for multi-axis grids."""
        other = Axis("other", labels=["x", "y"])
        with pytest.raises(ValueError, match="single-axis"):
            Grid({"A": 1, "B": 2, "C": 3}, along=[self.sites, other])

    def test_dict_axes_property(self):
        g = Grid({"A": 1, "B": 2, "C": 3}, along=self.sites)
        assert g.axes == (self.sites,)
