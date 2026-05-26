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

    def test_dict_scalar_inner_values_raise(self):
        """Scalar inner values at a non-last axis level raise ValueError."""
        other = Axis("other", labels=["x", "y"])
        with pytest.raises(ValueError, match="support len()"):
            Grid({"A": 1, "B": 2, "C": 3}, along=[self.sites, other])

    def test_dict_axes_property(self):
        g = Grid({"A": 1, "B": 2, "C": 3}, along=self.sites)
        assert g.axes == (self.sites,)


class TestGridMultiAxisDict:
    """Tests for mixed sequence/mapping multi-axis Grid construction."""

    def setup_method(self):
        self.sites = Axis("site", labels=["A", "B"])
        self.members_int = Axis("member", size=3)
        self.members_str = Axis("member", labels=["m0", "m1", "m2"])

    # ------------------------------------------------------------------
    # Dict of lists (outer labeled, inner positional)
    # ------------------------------------------------------------------

    def test_dict_of_lists_values(self):
        g = Grid(
            {"A": [10, 20, 30], "B": [40, 50, 60]},
            along=[self.sites, self.members_int],
        )
        assert g.value_at({self.sites: 0, self.members_int: 0}) == 10
        assert g.value_at({self.sites: 0, self.members_int: 2}) == 30
        assert g.value_at({self.sites: 1, self.members_int: 0}) == 40
        assert g.value_at({self.sites: 1, self.members_int: 2}) == 60

    def test_dict_of_lists_order_independent(self):
        """Outer dict key order does not affect the positional result."""
        g = Grid(
            {"B": [40, 50, 60], "A": [10, 20, 30]},  # reversed outer order
            along=[self.sites, self.members_int],
        )
        assert g.value_at({self.sites: 0, self.members_int: 0}) == 10  # A
        assert g.value_at({self.sites: 1, self.members_int: 0}) == 40  # B

    def test_dict_of_lists_axes(self):
        g = Grid(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            along=[self.sites, self.members_int],
        )
        assert g.axes == (self.sites, self.members_int)

    # ------------------------------------------------------------------
    # List of dicts (outer positional, inner labeled)
    # ------------------------------------------------------------------

    def test_list_of_dicts_values(self):
        g = Grid(
            [{"m0": 10, "m1": 20, "m2": 30}, {"m0": 40, "m1": 50, "m2": 60}],
            along=[self.sites, self.members_str],
        )
        assert g.value_at({self.sites: 0, self.members_str: 0}) == 10
        assert g.value_at({self.sites: 0, self.members_str: 2}) == 30
        assert g.value_at({self.sites: 1, self.members_str: 0}) == 40
        assert g.value_at({self.sites: 1, self.members_str: 2}) == 60

    def test_list_of_dicts_inner_order_independent(self):
        """Inner dict key order does not affect the positional result."""
        g = Grid(
            [{"m2": 30, "m0": 10, "m1": 20}, {"m0": 40, "m2": 60, "m1": 50}],
            along=[self.sites, self.members_str],
        )
        assert g.value_at({self.sites: 0, self.members_str: 0}) == 10  # m0
        assert g.value_at({self.sites: 0, self.members_str: 2}) == 30  # m2

    # ------------------------------------------------------------------
    # Dict of dicts (both axes labeled)
    # ------------------------------------------------------------------

    def test_dict_of_dicts_values(self):
        g = Grid(
            {
                "A": {"m0": 10, "m1": 20, "m2": 30},
                "B": {"m0": 40, "m1": 50, "m2": 60},
            },
            along=[self.sites, self.members_str],
        )
        assert g.value_at({self.sites: 0, self.members_str: 0}) == 10
        assert g.value_at({self.sites: 0, self.members_str: 2}) == 30
        assert g.value_at({self.sites: 1, self.members_str: 0}) == 40
        assert g.value_at({self.sites: 1, self.members_str: 2}) == 60

    def test_dict_of_dicts_both_orders_independent(self):
        """Neither outer nor inner dict key order affects the result."""
        g = Grid(
            {
                "B": {"m2": 60, "m0": 40, "m1": 50},
                "A": {"m1": 20, "m2": 30, "m0": 10},
            },
            along=[self.sites, self.members_str],
        )
        assert g.value_at({self.sites: 0, self.members_str: 0}) == 10  # A/m0
        assert g.value_at({self.sites: 1, self.members_str: 2}) == 60  # B/m2

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_dict_at_integer_inner_axis_raises(self):
        """Inner mapping on an integer-labeled axis raises ValueError."""
        with pytest.raises(ValueError, match="integer-indexed only"):
            Grid(
                [{"m0": 1, "m1": 2, "m2": 3}, {"m0": 4, "m1": 5, "m2": 6}],
                along=[self.sites, self.members_int],
            )

    def test_dict_of_lists_missing_inner_label_raises(self):
        """Missing a label at the inner level raises ValueError."""
        with pytest.raises(ValueError, match="missing values for labels"):
            Grid(
                {"A": {"m0": 1, "m1": 2}, "B": {"m0": 4, "m1": 5}},  # m2 missing
                along=[self.sites, self.members_str],
            )

    def test_dict_of_lists_shape_mismatch_inner_raises(self):
        """Wrong inner list length raises ValueError."""
        with pytest.raises(ValueError, match="length 2"):
            Grid(
                {"A": [10, 20], "B": [40, 50]},  # member axis size is 3
                along=[self.sites, self.members_int],
            )
