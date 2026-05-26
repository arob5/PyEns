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
