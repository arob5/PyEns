"""Tests for pyens.axis.Axis."""

from __future__ import annotations

import pytest

from pyens import Axis


class TestAxisCreation:
    def test_labels(self):
        ax = Axis("site", labels=["s1", "s2", "s3"])
        assert ax.size == 3
        assert ax.labels == ("s1", "s2", "s3")
        assert ax.name == "site"

    def test_size(self):
        ax = Axis("member", size=10)
        assert ax.size == 10
        assert ax.labels == tuple(range(10))

    def test_labels_converted_to_tuple(self):
        ax = Axis("site", labels=["a", "b"])
        assert isinstance(ax.labels, tuple)

    def test_requires_labels_or_size(self):
        with pytest.raises(ValueError, match="one of 'labels' or 'size'"):
            Axis("x")

    def test_rejects_both_labels_and_size(self):
        with pytest.raises(ValueError, match="not both"):
            Axis("x", labels=["a"], size=1)

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="unique"):
            Axis("x", labels=["a", "a", "b"])

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="size must be >= 1"):
            Axis("x", size=0)


class TestAxisLabelAt:
    def test_explicit_labels(self):
        ax = Axis("site", labels=["harvard", "niwot"])
        assert ax.label_at(0) == "harvard"
        assert ax.label_at(1) == "niwot"

    def test_integer_labels(self):
        ax = Axis("member", size=5)
        assert ax.label_at(0) == 0
        assert ax.label_at(4) == 4

    def test_out_of_range(self):
        ax = Axis("x", size=3)
        with pytest.raises(IndexError, match="out of range"):
            ax.label_at(3)

    def test_negative_index(self):
        ax = Axis("x", size=3)
        with pytest.raises(IndexError, match="out of range"):
            ax.label_at(-1)


class TestAxisIdentity:
    def test_same_instance_is_equal(self):
        ax = Axis("site", size=3)
        assert ax is ax

    def test_different_instances_not_equal(self):
        """Two Axis objects with the same name are NOT the same dimension."""
        ax1 = Axis("site", size=3)
        ax2 = Axis("site", size=3)
        assert ax1 is not ax2
        assert ax1 != ax2

    def test_usable_as_dict_key(self):
        ax1 = Axis("site", size=2)
        ax2 = Axis("site", size=2)
        d = {ax1: "first", ax2: "second"}
        assert d[ax1] == "first"
        assert d[ax2] == "second"


class TestAxisRepr:
    def test_repr_with_labels(self):
        ax = Axis("site", labels=["s1", "s2"])
        assert "site" in repr(ax)
        assert "s1" in repr(ax)

    def test_repr_with_size(self):
        ax = Axis("member", size=100)
        assert "member" in repr(ax)
        assert "100" in repr(ax)

    def test_len(self):
        ax = Axis("x", size=7)
        assert len(ax) == 7
