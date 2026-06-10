"""
Unit tests for the GridToNetCDFProcessor (Pipeline 2) and hexcube helpers.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import box
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="netcdf stack not available")

if HAS_LIBS:
    from geoworkflow.schemas.config_models import GridNetCDFConfig
    from geoworkflow.processors.integration.netcdf_builder import (
        GridToNetCDFProcessor, build_grid_netcdf,
    )
    from geoworkflow.utils import hexcube


def _write_grid(path):
    """3 hexes-as-boxes with GridID + q/r columns."""
    gdf = gpd.GeoDataFrame(
        {"GridID": ["h0", "h1", "h2"], "q": [0, 1, 2], "r": [0, 0, 1]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 1, 3, 2)],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _write_table(path, *, variables=("PM25", "odiac_co2"), months=("2021-01", "2021-02"),
                 statistics=("weighted_mean",)):
    rows = []
    val = 0.0
    for var in variables:
        units = "ug/m3" if var == "PM25" else "gC/m2/day"
        for t in months:
            for stat in statistics:
                for gid in ("h0", "h1", "h2"):
                    val += 1.0
                    rows.append((gid, var, pd.Timestamp(t), stat, val, units))
    df = pd.DataFrame(rows, columns=["GridID", "variable", "time", "statistic", "value", "units"])
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def grid_file(temp_dir):
    return _write_grid(temp_dir / "grid.geojson")


@pytest.fixture
def table_file(temp_dir):
    return _write_table(temp_dir / "stats.parquet")


@pytest.fixture
def cube(temp_dir, grid_file, table_file):
    out = temp_dir / "city.nc"
    build_grid_netcdf(table_file, grid_file, out)
    return out


class TestCubeStructure:
    def test_dims_coords_vars(self, cube):
        ds = xr.open_dataset(cube)
        assert ds.sizes["cell"] == 3
        assert ds.sizes["time"] == 2
        assert set(ds.data_vars) == {"PM25", "odiac_co2"}
        for c in ("GridID", "lat", "lon", "q", "r", "time"):
            assert c in ds.coords
        assert list(ds["cell"].values) == ["h0", "h1", "h2"]
        ds.close()

    def test_units_attr(self, cube):
        ds = xr.open_dataset(cube)
        assert ds["PM25"].attrs["units"] == "ug/m3"
        assert ds["odiac_co2"].attrs["units"] == "gC/m2/day"
        ds.close()

    def test_cells_align_to_grid_order(self, temp_dir, grid_file):
        # Table omits h2 and is in shuffled order; cube must still follow grid order
        # and fill the missing cell with NaN.
        df = pd.DataFrame([
            ("h1", "PM25", pd.Timestamp("2021-01"), "weighted_mean", 5.0, "ug/m3"),
            ("h0", "PM25", pd.Timestamp("2021-01"), "weighted_mean", 9.0, "ug/m3"),
        ], columns=["GridID", "variable", "time", "statistic", "value", "units"])
        tbl = temp_dir / "partial.parquet"
        df.to_parquet(tbl, index=False)
        out = temp_dir / "partial.nc"
        build_grid_netcdf(tbl, grid_file, out)
        ds = xr.open_dataset(out)
        assert list(ds["cell"].values) == ["h0", "h1", "h2"]
        vals = ds["PM25"].sel(time="2021-01").values
        assert vals[0] == 9.0 and vals[1] == 5.0 and np.isnan(vals[2])
        ds.close()


class TestMultiStatistic:
    def test_suffixed_var_names(self, temp_dir, grid_file):
        tbl = _write_table(temp_dir / "multi.parquet",
                           statistics=("weighted_mean", "max"))
        out = temp_dir / "multi.nc"
        build_grid_netcdf(tbl, grid_file, out, statistics=["weighted_mean", "max"])
        ds = xr.open_dataset(out)
        assert {"PM25_weighted_mean", "PM25_max",
                "odiac_co2_weighted_mean", "odiac_co2_max"}.issubset(set(ds.data_vars))
        ds.close()


class TestGuards:
    def test_skip_existing(self, temp_dir, grid_file, table_file):
        out = temp_dir / "skip.nc"
        out.write_text("placeholder")
        config = GridNetCDFConfig(input_file=table_file, grid_file=grid_file,
                                  output_file=out, skip_existing=True)
        result = GridToNetCDFProcessor(config).process()
        assert result.success and result.skipped_count == 1
        assert out.read_text() == "placeholder"


class TestHexcubeHelpers:
    def test_query_helpers(self, cube, grid_file):
        ds = hexcube.open_hex_cube(cube)

        m = hexcube.select_month(ds, "PM25", 2021, 1)
        assert m.dims == ("cell",) and m.sizes["cell"] == 3

        ym = hexcube.annual_mean(ds, "PM25", 2021)
        assert ym.dims == ("cell",)

        jun = hexcube.month_across_years(ds, "PM25", 1)  # only month 1 present
        assert jun.dims == ("cell",)

        cm = hexcube.city_mean(ds, "PM25")
        assert cm.dims == ("time",) and cm.sizes["time"] == 2

        clim = hexcube.monthly_climatology(ds, "PM25")
        assert "month" in clim.dims

        near = hexcube.nearest_cell(ds, lon=0.5, lat=0.5)  # nearest h0 centroid
        assert str(near["cell"].values) == "h0"
        ds.close()

    def test_to_geodataframe(self, cube, grid_file):
        ds = hexcube.open_hex_cube(cube)
        gdf = hexcube.to_geodataframe(ds, grid_file, variable="PM25", time="2021-01")
        assert "geometry" in gdf.columns
        assert any(c.startswith("PM25") for c in gdf.columns)
        assert len(gdf) == 3
        ds.close()


class TestCombinedCube:
    """B + CF metadata: monthly and annual variables in one cube."""

    def test_monthly_and_annual_axes_with_cf_flags(self, temp_dir, grid_file):
        # PM25 monthly (continuous) + landcover single-year categorical (majority).
        rows = []
        for t in ("2021-01", "2021-02"):
            for i, gid in enumerate(("h0", "h1", "h2")):
                rows.append((gid, "PM25", pd.Timestamp(t), "weighted_mean", 5.0 + i, "ug/m3"))
        for gid, code in zip(("h0", "h1", "h2"), (50, 40, 80)):
            rows.append((gid, "landcover", pd.Timestamp("2019-01-01"), "majority", float(code), None))
        df = pd.DataFrame(rows, columns=["GridID", "variable", "time", "statistic", "value", "units"])
        tbl = temp_dir / "combined.parquet"; df.to_parquet(tbl, index=False)
        out = temp_dir / "combined.nc"
        build_grid_netcdf(tbl, grid_file, out,
                          statistics=["weighted_mean", "majority"],
                          annual_variables=["landcover"])
        ds = xr.open_dataset(out)
        # Each variable on its own cadence axis, in one cube.
        assert ds["PM25"].dims == ("cell", "time")
        assert ds["landcover"].dims == ("cell", "year")
        assert list(ds["year"].values) == [2019]
        assert ds.sizes["time"] == 2
        # Any 2019 query resolves through the annual axis.
        assert int(ds["landcover"].sel(year=2019).sel(cell="h0").values) == 50
        # CF flags make the categorical var self-describing (from DEFAULT_LEGENDS).
        assert 50 in list(ds["landcover"].attrs["flag_values"])
        assert "urban_built_up" in ds["landcover"].attrs["flag_meanings"].split()
        ds.close()

    def test_single_year_landcover_needs_override(self, temp_dir, grid_file):
        # Without annual_variables, a single Jan-1 series stays on the monthly axis
        # (one Jan timestamp is ambiguous).
        rows = [(g, "landcover", pd.Timestamp("2019-01-01"), "majority", float(c), None)
                for g, c in zip(("h0", "h1", "h2"), (50, 40, 80))]
        df = pd.DataFrame(rows, columns=["GridID", "variable", "time", "statistic", "value", "units"])
        tbl = temp_dir / "lc.parquet"; df.to_parquet(tbl, index=False)
        out = temp_dir / "lc.nc"
        build_grid_netcdf(tbl, grid_file, out, statistics=["majority"])
        ds = xr.open_dataset(out)
        assert ds["landcover"].dims == ("cell", "time")
        ds.close()

    def test_default_includes_all_statistics(self, temp_dir, grid_file):
        # statistics=None (the default) must include every statistic present --
        # a categorical variable must not vanish because the caller didn't list
        # its statistics explicitly.
        rows = [(g, "PM25", pd.Timestamp("2021-01"), "weighted_mean", 5.0, "ug/m3")
                for g in ("h0", "h1", "h2")]
        rows += [(g, "landcover", pd.Timestamp("2019-01-01"), "majority", 50.0, None)
                 for g in ("h0", "h1", "h2")]
        df = pd.DataFrame(rows, columns=["GridID", "variable", "time", "statistic", "value", "units"])
        tbl = temp_dir / "mixed.parquet"; df.to_parquet(tbl, index=False)
        out = temp_dir / "mixed.nc"
        build_grid_netcdf(tbl, grid_file, out, annual_variables=["landcover"])
        ds = xr.open_dataset(out)
        assert set(ds.data_vars) == {"PM25", "landcover"}
        ds.close()
