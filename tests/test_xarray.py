"""Tests for pyens.xarray: building fields from xarray objects."""

from __future__ import annotations

import datetime
import doctest
import importlib
import pickle
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")

import pyens.xarray as pyens_xarray  # noqa: E402
from pyens import (  # noqa: E402
    Axis,
    EnsembleRunner,
    EnsembleSpec,
    Fixed,
    Grid,
    LocalBackend,
    SequentialBackend,
)
from pyens.xarray import (  # noqa: E402
    axes_of,
    dataset_as_field,
    field_from_dataarray,
    fields_from_dataset,
)
from tests import _models  # noqa: E402

SITES = ["harvard_forest", "niwot_ridge", "bartlett"]
MEMBERS = [10, 20]


def _identity(**inputs):
    return inputs


def _table() -> xr.Dataset:
    """A (member, site) table: one variable on both dims, one on site only."""
    return xr.Dataset(
        {
            "leaf_area": (("member", "site"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "soil_depth": ("site", [0.5, 0.8, 1.1]),
        },
        coords={
            "member": MEMBERS,
            "site": SITES,
            "lon": ("site", [-72.2, -105.5, -71.3]),
            "source": "reanalysis",
        },
    )


def _forcing() -> xr.Dataset:
    """Forcing over (site, time), plus a CO2 series shared by every site."""
    time = pd.date_range("2020-01-01", periods=4, freq="D")
    return xr.Dataset(
        {
            "tair": (("site", "time"), np.arange(12.0).reshape(3, 4)),
            "precip": (("site", "time"), np.arange(12.0).reshape(3, 4) / 10),
            "co2": ("time", [410.0, 411.0, 412.0, 413.0]),
        },
        coords={"site": SITES, "time": time},
    )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


class TestAxesOf:
    def test_one_axis_per_dim(self):
        axes = axes_of(_table())
        assert axes == {
            "member": Axis("member", labels=MEMBERS),
            "site": Axis("site", labels=SITES),
        }

    def test_labels_are_plain_python_values(self):
        axes = axes_of(_table())
        assert all(type(label) is int for label in axes["member"].labels)
        assert all(type(label) is str for label in axes["site"].labels)

    def test_dataarray(self):
        axes = axes_of(_table()["leaf_area"])
        assert list(axes) == ["member", "site"]

    def test_dim_without_coordinate_is_positional(self):
        array = xr.DataArray(np.zeros((3, 2)), dims=("member", "site"),
                             coords={"site": ["a", "b"]})
        axes = axes_of(array)
        assert axes["member"] == Axis("member", size=3)
        assert axes["member"].labels == (0, 1, 2)
        assert axes["member"] != Axis("member", labels=[0, 1, 2])

    def test_same_labels_from_another_caller_give_equal_axes(self):
        other = xr.Dataset(
            {"lai_obs": (("site", "member"), np.zeros((3, 2)))},
            coords={"site": np.array(SITES, dtype=object), "member": np.array(MEMBERS)},
        )
        assert axes_of(other)["site"] == axes_of(_table())["site"]
        assert axes_of(other)["member"] == axes_of(_table())["member"]

    def test_duplicate_labels_refused(self):
        dataset = xr.Dataset({"x": ("site", [1, 2])}, coords={"site": ["a", "a"]})
        with pytest.raises(ValueError, match=r"dim 'site'.*duplicate labels \['a'\]"):
            axes_of(dataset)


# ---------------------------------------------------------------------------
# fields_from_dataset: plain values
# ---------------------------------------------------------------------------


class TestFieldsFromDataset:
    def test_shared_dim_zips_other_dim_crosses(self):
        dataset = _table()
        fields = fields_from_dataset(dataset)
        spec = EnsembleSpec(inputs=fields)

        assert spec.n_runs == dataset.sizes["member"] * dataset.sizes["site"]
        assert fields["leaf_area"].axes[1] is fields["soil_depth"].axes[0]
        seen = []
        for inputs, coord in spec.iter_runs():
            expected = dataset.sel(coord)
            assert inputs == {
                "leaf_area": expected["leaf_area"].item(),
                "soil_depth": expected["soil_depth"].item(),
            }
            seen.append(tuple(sorted(coord.items())))
        assert len(set(seen)) == spec.n_runs

    def test_values_are_plain_python_values(self):
        fields = fields_from_dataset(_table())
        value = fields["leaf_area"].value_at(
            {ax: 0 for ax in fields["leaf_area"].axes}
        )
        assert type(value) is float

    def test_axes_follow_each_variables_dim_order(self):
        dataset = _table()
        dataset["transposed"] = dataset["leaf_area"].transpose("site", "member")
        fields = fields_from_dataset(dataset)
        assert [a.name for a in fields["transposed"].axes] == ["site", "member"]
        spec = EnsembleSpec(inputs=fields)
        assert spec.n_runs == 6
        for inputs, _ in spec.iter_runs():
            assert inputs["transposed"] == inputs["leaf_area"]

    def test_hand_built_label_keyed_grid_zips(self):
        sites = Axis("site", labels=SITES)
        climate = Grid({"bartlett": "c3", "harvard_forest": "c1", "niwot_ridge": "c2"},
                       along=sites)
        spec = EnsembleSpec(inputs={**fields_from_dataset(_table()), "climate": climate})

        assert spec.n_runs == 6
        by_site = {"harvard_forest": "c1", "niwot_ridge": "c2", "bartlett": "c3"}
        for inputs, coord in spec.iter_runs():
            assert inputs["climate"] == by_site[coord["site"]]

    def test_two_datasets_over_the_same_labels_zip(self):
        observed = xr.Dataset(
            {"lai_obs": ("site", [9.0, 8.0, 7.0])},
            coords={"site": SITES},
        )
        spec = EnsembleSpec(inputs={
            **fields_from_dataset(_table()),
            **fields_from_dataset(observed),
        })
        assert spec.n_runs == 6
        for inputs, coord in spec.iter_runs():
            assert inputs["lai_obs"] == observed["lai_obs"].sel(site=coord["site"]).item()

    def test_positional_dim(self):
        dataset = _table().drop_vars("member")
        fields = fields_from_dataset(dataset)
        assert fields["leaf_area"].axes[0] == Axis("member", size=2)
        spec = EnsembleSpec(inputs=fields)
        for inputs, coord in spec.iter_runs():
            assert type(coord["member"]) is int
            expected = dataset["leaf_area"].isel(member=coord["member"]).sel(site=coord["site"])
            assert inputs["leaf_area"] == expected.item()

    def test_zero_dim_variable_is_fixed(self):
        dataset = _table()
        dataset["timestep"] = ((), 3600)
        fields = fields_from_dataset(dataset)
        assert isinstance(fields["timestep"], Fixed)
        assert fields["timestep"].value_at({}) == 3600
        assert type(fields["timestep"].value_at({})) is int
        assert EnsembleSpec(inputs=fields).n_runs == 6

    def test_duplicate_labels_refused_naming_the_variable(self):
        dataset = xr.Dataset(
            {"soil_depth": ("site", [0.5, 0.8])},
            coords={"site": ["a", "a"]},
        )
        with pytest.raises(ValueError, match=r"variable 'soil_depth'.*dim 'site'.*unique"):
            fields_from_dataset(dataset)

    def test_no_casting_and_nan_kept(self):
        dataset = xr.Dataset(
            {
                "count": ("site", np.array([1, 2], dtype=np.int32)),
                "flux": ("site", [np.nan, 1.5]),
                "name": ("site", np.array(["x", "y"], dtype=object)),
            },
            coords={"site": ["a", "b"]},
        )
        fields = fields_from_dataset(dataset)
        sites = fields["count"].axes[0]
        assert fields["count"].value_at({sites: 1}) == 2
        assert type(fields["count"].value_at({sites: 1})) is int
        assert np.isnan(fields["flux"].value_at({sites: 0}))
        assert fields["name"].value_at({sites: 1}) == "y"

    def test_plain_value_spec_survives_dump_and_load(self, tmp_path):
        dataset = _forcing()[["co2"]].assign(offset=("site", [1.0, 2.0, 3.0]))
        spec = EnsembleSpec(inputs=fields_from_dataset(dataset))
        path = tmp_path / "spec.json"
        spec.dump(path)
        loaded = EnsembleSpec.load(path)
        assert list(loaded.iter_runs()) == list(spec.iter_runs())


# ---------------------------------------------------------------------------
# along=: DataArray slices
# ---------------------------------------------------------------------------


class TestAlong:
    def test_slices_keep_the_other_dims(self):
        forcing = _forcing()
        grid = field_from_dataarray(forcing["tair"], along="site")
        assert grid.axes == (Axis("site", labels=SITES),)

        spec = EnsembleSpec(inputs={"tair": grid})
        assert spec.n_runs == 3
        for inputs, coord in spec.iter_runs():
            value = inputs["tair"]
            assert isinstance(value, xr.DataArray)
            assert value.dims == ("time",)
            xr.testing.assert_identical(value, forcing["tair"].sel(site=coord["site"]))
            assert value["site"].item() == coord["site"]

    def test_along_order_sets_axis_order(self):
        array = _table()["leaf_area"].expand_dims(layer=2).transpose("member", "layer", "site")
        grid = field_from_dataarray(array, along=["site", "member"])
        assert [a.name for a in grid.axes] == ["site", "member"]
        spec = EnsembleSpec(inputs={"x": grid})
        for inputs, coord in spec.iter_runs():
            assert inputs["x"].dims == ("layer",)
            xr.testing.assert_identical(inputs["x"], array.sel(coord))

    def test_along_every_dim_gives_plain_values(self):
        array = _table()["leaf_area"]
        grid = field_from_dataarray(array, along=["site", "member"])
        assert [a.name for a in grid.axes] == ["site", "member"]
        spec = EnsembleSpec(inputs={"x": grid})
        for inputs, coord in spec.iter_runs():
            assert inputs["x"] == array.sel(coord).item()

    def test_dataset_variable_without_along_dims_is_fixed(self):
        forcing = _forcing()
        fields = fields_from_dataset(forcing, along="site")
        assert isinstance(fields["co2"], Fixed)
        xr.testing.assert_identical(fields["co2"].value_at({}), forcing["co2"])
        assert EnsembleSpec(inputs=fields).n_runs == 3

    @pytest.mark.parametrize(
        ("along", "message"),
        [
            ([], "at least one dim"),
            (["site", "site"], "repeats"),
            (["plot"], r"dim\(s\) \['plot'\]"),
        ],
    )
    def test_bad_along_refused(self, along, message):
        with pytest.raises(ValueError, match=message):
            field_from_dataarray(_forcing()["tair"], along=along)
        with pytest.raises(ValueError, match=message):
            fields_from_dataset(_forcing(), along=along)

    def test_slice_pickles_only_its_own_data(self):
        array = xr.DataArray(np.random.default_rng(0).random((50, 2000)),
                             dims=("site", "time"))
        grid = field_from_dataarray(array, along="site")
        cell = grid.value_at({grid.axes[0]: 7})
        assert len(pickle.dumps(cell)) < len(pickle.dumps(array)) / 20

    def test_slices_run_in_worker_processes(self):
        forcing = _forcing()
        spec = EnsembleSpec(inputs={
            "series": field_from_dataarray(forcing["tair"], along="site"),
            "scale": Grid([1.0, 10.0], along=Axis("scale", size=2)),
        })
        result = EnsembleRunner(_models.site_series_total,
                                backend=LocalBackend(n_workers=2)).run(spec)
        assert result.n_failed == 0
        for record in result:
            site = record.coordinate["site"]
            scale = [1.0, 10.0][record.coordinate["scale"]]
            total = float(forcing["tair"].sel(site=site).sum()) * scale
            assert record.output == (site, total)


# ---------------------------------------------------------------------------
# dataset_as_field
# ---------------------------------------------------------------------------


class TestDatasetAsField:
    def test_each_run_gets_a_sub_dataset(self):
        forcing = _forcing()
        climate = dataset_as_field(forcing, along="site")
        assert climate.axes == (Axis("site", labels=SITES),)
        spec = EnsembleSpec(inputs={"climate": climate})
        for inputs, coord in spec.iter_runs():
            value = inputs["climate"]
            assert isinstance(value, xr.Dataset)
            assert set(value.data_vars) == {"tair", "precip", "co2"}
            xr.testing.assert_identical(value, forcing.sel(site=coord["site"]))

    def test_zips_with_fields_from_another_dataset(self):
        spec = EnsembleSpec(inputs={
            **fields_from_dataset(_table()),
            "climate": dataset_as_field(_forcing(), along="site"),
        })
        assert spec.n_runs == 6
        for inputs, coord in spec.iter_runs():
            assert inputs["climate"]["site"].item() == coord["site"]

    def test_along_required(self):
        with pytest.raises(TypeError):
            dataset_as_field(_forcing())  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="at least one dim"):
            dataset_as_field(_forcing(), along=[])


# ---------------------------------------------------------------------------
# axes=
# ---------------------------------------------------------------------------


class TestSuppliedAxes:
    def test_matched_by_label_and_reordered(self):
        dataset = _table()
        sites = Axis("site", labels=list(reversed(SITES)))
        fields = fields_from_dataset(dataset, axes={"site": sites})
        assert fields["leaf_area"].axes[1] is sites
        assert fields["soil_depth"].axes[0] is sites
        spec = EnsembleSpec(inputs=fields)
        for inputs, coord in spec.iter_runs():
            expected = dataset.sel(coord)
            assert inputs["leaf_area"] == expected["leaf_area"].item()
            assert inputs["soil_depth"] == expected["soil_depth"].item()

    def test_reorders_slices_and_sub_datasets(self):
        forcing = _forcing()
        sites = Axis("site", labels=["bartlett", "harvard_forest", "niwot_ridge"])
        for field in (
            field_from_dataarray(forcing["tair"], along="site", axes={"site": sites}),
            dataset_as_field(forcing, along="site", axes={"site": sites}),
        ):
            for i, label in enumerate(sites.labels):
                assert field.value_at({sites: i})["site"].item() == label

    def test_axis_name_is_the_coordinate_key(self):
        stations = Axis("station", labels=SITES)
        spec = EnsembleSpec(inputs=fields_from_dataset(_table(), axes={"site": stations}))
        _, coord = next(spec.iter_runs())
        assert set(coord) == {"member", "station"}

    def test_wrong_size_refused(self):
        with pytest.raises(ValueError, match=r"variable 'leaf_area'.*size 2.*size 3"):
            fields_from_dataset(_table(), axes={"site": Axis("site", labels=["a", "b"])})

    def test_wrong_labels_refused(self):
        sites = Axis("site", labels=["harvard_forest", "niwot_ridge", "howland"])
        with pytest.raises(ValueError, match=r"\['howland'\] that the dim does not"):
            fields_from_dataset(_table(), axes={"site": sites})

    def test_positional_dim_accepts_a_size_axis(self):
        dataset = _table().drop_vars("member")
        members = Axis("member", size=2)
        fields = fields_from_dataset(dataset, axes={"member": members})
        assert fields["leaf_area"].axes[0] is members

    def test_positional_dim_refuses_string_labels(self):
        dataset = _table().drop_vars("member")
        with pytest.raises(ValueError, match=r"no coordinate.*Axis\('member', size=2\)"):
            fields_from_dataset(dataset, axes={"member": Axis("member", labels=["a", "b"])})

    def test_unused_entries_ignored(self):
        fields = fields_from_dataset(_table(), axes={"plot": Axis("plot", size=4)})
        assert EnsembleSpec(inputs=fields).n_runs == 6


# ---------------------------------------------------------------------------
# Date and time labels
# ---------------------------------------------------------------------------


class TestTimeLabels:
    @pytest.mark.parametrize("unit", ["ns", "us", "s"])
    def test_datetime_labels_are_iso_strings(self, unit):
        time = np.array(["2020-01-01T00:00", "2020-01-01T06:30"], dtype=f"datetime64[{unit}]")
        array = xr.DataArray([1.0, 2.0], dims="time", coords={"time": time})
        assert axes_of(array)["time"].labels == ("2020-01-01T00:00:00", "2020-01-01T06:30:00")

    def test_fractional_seconds(self):
        time = pd.to_datetime(["2020-01-01T00:00:00.5"])
        array = xr.DataArray([1.0], dims="time", coords={"time": time})
        assert axes_of(array)["time"].labels == ("2020-01-01T00:00:00.500000",)

    def test_datetime_round_trips_through_run_record(self, tmp_path):
        dataset = xr.Dataset(
            {"tair": ("time", [280.0, 281.0, 282.0])},
            coords={"time": pd.date_range("2020-01-01", periods=3, freq="6h")},
        )
        spec = EnsembleSpec(inputs=fields_from_dataset(dataset))
        result = EnsembleRunner(_identity, backend=SequentialBackend()).run(spec)
        for record in result:
            label = record.coordinate["time"]
            assert isinstance(label, str)
            assert record.output["tair"] == dataset["tair"].sel(time=label).item()
            assert pd.Timestamp(label) in dataset.indexes["time"]

        path = tmp_path / "spec.json"
        spec.dump(path)
        assert EnsembleSpec.load(path).axes == spec.axes

    def test_cftime_labels_are_iso_strings(self):
        pytest.importorskip("cftime")
        time = xr.date_range("2000-02-28", periods=2, calendar="noleap", use_cftime=True)
        array = xr.DataArray([1.0, 2.0], dims="time", coords={"time": time})
        labels = axes_of(array)["time"].labels
        assert labels == ("2000-02-28T00:00:00", "2000-03-01T00:00:00")
        assert array.sel(time=labels[1]).item() == 2.0

    def test_timedelta_labels_are_iso_durations(self):
        lead = pd.to_timedelta(["0h", "6h"]).values
        array = xr.DataArray([1.0, 2.0], dims="lead", coords={"lead": lead})
        assert axes_of(array)["lead"].labels == ("P0DT0H0M0S", "P0DT6H0M0S")

    def test_date_objects(self):
        dates = np.array([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)], dtype=object)
        array = xr.DataArray([1.0, 2.0], dims="day", coords={"day": dates})
        assert axes_of(array)["day"].labels == ("2020-01-01", "2020-01-02")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def test_docstring_examples_run():
    failures, tried = doctest.testmod(pyens_xarray, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tried > 0
    assert failures == 0


def test_import_guard(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)
    monkeypatch.delitem(sys.modules, "pyens.xarray")
    with pytest.raises(ImportError, match=r"pip install pyens\[xarray\]"):
        importlib.import_module("pyens.xarray")


def test_core_does_not_import_xarray():
    code = "import sys, pyens; print('xarray' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
