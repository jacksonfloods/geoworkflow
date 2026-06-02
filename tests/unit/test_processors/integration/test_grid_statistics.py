"""
Unit tests for the GridStatisticsProcessor (Pipeline 1).
"""

import json

import numpy as np
import pandas as pd
import pytest

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from exactextract import exact_extract  # noqa: F401
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="grid-stats stack not available")

if HAS_LIBS:
    from geoworkflow.schemas.config_models import GridStatisticsConfig
    from geoworkflow.processors.integration.grid_statistics import (
        GridStatisticsProcessor,
        compute_grid_statistics,
    )

TIDY_COLS = ["GridID", "variable", "time", "statistic", "value", "units"]


def _write_grid(path):
    """3 hexes-as-boxes: inside value-0 cell, inside value-55 cell, far away."""
    gdf = gpd.GeoDataFrame(
        {"GridID": ["h0", "h1", "hfar"]},
        geometry=[
            box(0.2, 9.2, 0.6, 9.6),   # cell row0,col0 -> 0
            box(5.2, 4.2, 5.6, 4.6),   # cell row5,col5 -> 55
            box(50, 50, 51, 51),       # outside the raster -> NaN
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _write_raster(path):
    data = np.arange(100, dtype="float32").reshape(10, 10)
    profile = dict(driver="GTiff", height=10, width=10, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(0, 10, 1, 1), nodata=None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def grid_file(temp_dir):
    return _write_grid(temp_dir / "grid.geojson")


@pytest.fixture
def rasters(temp_dir):
    return [_write_raster(temp_dir / "ras1.tif"), _write_raster(temp_dir / "ras2.tif")]


class TestBasicRun:
    def test_tidy_output(self, temp_dir, grid_file, rasters):
        out = temp_dir / "stats.parquet"
        config = GridStatisticsConfig(
            grid_file=grid_file, raster_inputs=rasters, output_file=out,
            statistics=["weighted_mean", "max"],
        )
        result = GridStatisticsProcessor(config).process()
        assert result.success, result.message
        assert out.exists()

        df = pd.read_parquet(out)
        assert list(df.columns) == TIDY_COLS
        # 3 hexes x 2 variables x 2 statistics
        assert len(df) == 3 * 2 * 2
        assert set(df["variable"]) == {"ras1", "ras2"}
        # no registry match -> no time
        assert df["time"].isna().all()

    def test_values_are_coverage_weighted(self, temp_dir, grid_file, rasters):
        out = temp_dir / "stats.parquet"
        compute_grid_statistics(grid_file, rasters, out, statistics=["weighted_mean"])
        df = pd.read_parquet(out)
        wm = df[(df["variable"] == "ras1") & (df["statistic"] == "weighted_mean")]
        by_hex = dict(zip(wm["GridID"], wm["value"]))
        assert by_hex["h0"] == pytest.approx(0.0)
        assert by_hex["h1"] == pytest.approx(55.0)

    def test_non_overlapping_hex_kept_as_nan(self, temp_dir, grid_file, rasters):
        out = temp_dir / "stats.parquet"
        compute_grid_statistics(grid_file, [rasters[0]], out,
                                statistics=["weighted_mean"])
        df = pd.read_parquet(out)
        far = df[df["GridID"] == "hfar"]
        assert len(far) == 1                 # row kept
        assert np.isnan(far["value"].iloc[0])  # value is NaN


class TestRegistryDriven:
    def test_dataset_registry_supplies_time_and_units(self, temp_dir, grid_file, rasters):
        user_reg = temp_dir / "raster_datasets.json"
        user_reg.write_text(json.dumps({"datasets": [
            {"name": "ras1", "match": "ras1.tif", "variable": "myvar",
             "units": "u", "time": {"static": "2022-03"}}
        ]}))
        out = temp_dir / "stats.parquet"
        config = GridStatisticsConfig(
            grid_file=grid_file, raster_inputs=rasters, output_file=out,
            statistics=["weighted_mean"], dataset_registry=user_reg,
        )
        GridStatisticsProcessor(config).process()
        df = pd.read_parquet(out)

        ras1 = df[df["variable"] == "myvar"]
        assert not ras1.empty
        assert (ras1["time"] == pd.Timestamp("2022-03-01")).all()
        assert (ras1["units"] == "u").all()
        # ras2 still falls back to its stem with no time
        assert "ras2" in set(df["variable"])


class TestGuards:
    def test_skip_existing(self, temp_dir, grid_file, rasters):
        out = temp_dir / "stats.parquet"
        out.write_text("placeholder")
        config = GridStatisticsConfig(
            grid_file=grid_file, raster_inputs=rasters, output_file=out,
            skip_existing=True,
        )
        result = GridStatisticsProcessor(config).process()
        assert result.success
        assert result.skipped_count == 1
        assert out.read_text() == "placeholder"

    def test_bad_statistic_fails_validation(self, temp_dir, grid_file, rasters):
        out = temp_dir / "stats.parquet"
        config = GridStatisticsConfig(
            grid_file=grid_file, raster_inputs=rasters, output_file=out,
            statistics=["not_a_stat"],
        )
        result = GridStatisticsProcessor(config).process()
        assert result.success is False
