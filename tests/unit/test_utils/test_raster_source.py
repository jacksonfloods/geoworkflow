"""
Unit tests for the raster source layer (geoworkflow.utils.raster_source).
"""

import numpy as np
import pandas as pd
import pytest

try:
    import geopandas as gpd
    import rasterio
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.transform import from_origin
    from shapely.geometry import box
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="raster source stack not available")

if HAS_LIBS:
    from geoworkflow.core.dataset_registry import DatasetRegistry, RasterDatasetSpec
    from geoworkflow.utils.raster_source import open_raster_slices
    from geoworkflow.utils.zonal_utils import compute_zonal_statistics


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _write_geotiff(path, data, *, crs="EPSG:4326"):
    """10x10-style grid, 1-degree cells, origin (0, N) top-left."""
    h, w = data.shape
    profile = dict(driver="GTiff", height=h, width=w, count=1,
                   dtype="float32", crs=crs, transform=from_origin(0, h, 1, 1),
                   nodata=None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _write_netcdf_2d(path, varname, data, *, attrs=None):
    h, w = data.shape
    lats = np.arange(h - 0.5, -0.5, -1.0)   # descending centers: h-0.5 .. 0.5
    lons = np.arange(0.5, w + 0.5, 1.0)     # ascending centers
    da = xr.DataArray(data.astype("float32"), dims=("lat", "lon"),
                      coords={"lat": lats, "lon": lons}, name=varname,
                      attrs=attrs or {})
    da.to_dataset().to_netcdf(path)
    return path


def _write_netcdf_time(path, varname, data3d, times):
    t, h, w = data3d.shape
    lats = np.arange(h - 0.5, -0.5, -1.0)
    lons = np.arange(0.5, w + 0.5, 1.0)
    da = xr.DataArray(
        data3d.astype("float32"), dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(times), "lat": lats, "lon": lons},
        name=varname,
    )
    da.to_dataset().to_netcdf(path)
    return path


@pytest.fixture
def grid_10():
    """row-major values 0..99 on a 10x10 grid."""
    return np.arange(100, dtype="float32").reshape(10, 10)


@pytest.fixture
def aoi_box():
    # covers lon 2..5, lat 2..5
    return gpd.GeoDataFrame({"id": [1]}, geometry=[box(2, 2, 5, 5)], crs="EPSG:4326")


# --------------------------------------------------------------------------
# GeoTIFF
# --------------------------------------------------------------------------

class TestGeoTIFF:
    def test_static_geotiff_one_slice(self, temp_dir, grid_10):
        path = _write_geotiff(temp_dir / "wsf3d_height.tif", grid_10)
        reg = DatasetRegistry({
            "wsf": RasterDatasetSpec(name="wsf", match="wsf3d_*.tif",
                                     variable="building_height",
                                     time={"static": "2020-01"})
        })
        slices = list(open_raster_slices(path, registry=reg))
        assert len(slices) == 1
        s = slices[0]
        assert s.variable == "building_height"
        assert s.time == pd.Timestamp("2020-01-01")
        assert s.crs == "EPSG:4326"
        assert s.shape == (10, 10)

    def test_no_match_falls_back_to_stem(self, temp_dir, grid_10):
        path = _write_geotiff(temp_dir / "mystery.tif", grid_10)
        reg = DatasetRegistry({})  # empty
        slices = list(open_raster_slices(path, registry=reg))
        assert len(slices) == 1
        assert slices[0].variable == "mystery"
        assert slices[0].time is None

    def test_aoi_windowing_shrinks_array(self, temp_dir, grid_10, aoi_box):
        path = _write_geotiff(temp_dir / "mystery.tif", grid_10)
        full = list(open_raster_slices(path, registry=DatasetRegistry({})))[0]
        clipped = list(open_raster_slices(path, aoi=aoi_box,
                                          registry=DatasetRegistry({})))[0]
        assert clipped.array.size < full.array.size
        assert clipped.shape[0] <= 4 and clipped.shape[1] <= 4


# --------------------------------------------------------------------------
# netCDF
# --------------------------------------------------------------------------

class TestNetCDF:
    def test_netcdf_2d_default_registry_pm25(self, temp_dir, grid_10):
        # Name matches the packaged pm25 entry; no in-file CRS or time dim.
        path = _write_netcdf_2d(
            temp_dir / "V6GL02.04.CNNPM25.GL.202005-202005.nc", "PM25", grid_10,
            attrs={"units": "ug/m3"},
        )
        slices = list(open_raster_slices(path))  # default registry
        assert len(slices) == 1
        s = slices[0]
        assert s.variable == "PM25"
        assert s.time == pd.Timestamp("2020-05-01")
        assert s.crs == "EPSG:4326"          # assigned via spec default
        assert s.units == "ug/m3"

    def test_netcdf_time_dim_yields_one_slice_per_step(self, temp_dir):
        data = np.random.rand(3, 6, 6).astype("float32")
        times = ["2021-01-01", "2021-02-01", "2021-03-01"]
        path = _write_netcdf_time(temp_dir / "stack.nc", "co2", data, times)
        reg = DatasetRegistry({
            "stack": RasterDatasetSpec(name="stack", match="stack.nc",
                                       variable="co2", crs="EPSG:4326")
        })
        slices = list(open_raster_slices(path, registry=reg))
        assert len(slices) == 3
        assert [s.time for s in slices] == [pd.Timestamp(t) for t in times]
        assert all(s.variable == "co2" for s in slices)

    def test_netcdf_clip_box_shrinks(self, temp_dir, grid_10, aoi_box):
        path = _write_netcdf_2d(temp_dir / "plain.nc", "v", grid_10)
        reg = DatasetRegistry({
            "v": RasterDatasetSpec(name="v", match="plain.nc", variable="v",
                                   crs="EPSG:4326")
        })
        s = list(open_raster_slices(path, aoi=aoi_box, registry=reg))[0]
        assert s.array.size < grid_10.size


# --------------------------------------------------------------------------
# Multi-input + engine integration
# --------------------------------------------------------------------------

class TestIntegration:
    def test_directory_input_multiple_files(self, temp_dir, grid_10):
        d = temp_dir / "rasters"
        d.mkdir()
        _write_geotiff(d / "a.tif", grid_10)
        _write_geotiff(d / "b.tif", grid_10)
        slices = list(open_raster_slices(d, registry=DatasetRegistry({})))
        assert {s.variable for s in slices} == {"a", "b"}

    def test_slice_feeds_zonal_engine(self, temp_dir, grid_10):
        # cell (row5,col5)=55 spans lon[5,6], lat[4,5]; a hex inside -> wmean 55.
        path = _write_geotiff(temp_dir / "mystery.tif", grid_10)
        s = list(open_raster_slices(path, registry=DatasetRegistry({})))[0]
        hexes = gpd.GeoDataFrame(
            {"GridID": ["h"]}, geometry=[box(5.2, 4.2, 5.6, 4.6)], crs="EPSG:4326"
        )
        with s.open() as ds:
            out = compute_zonal_statistics(ds, hexes, ["weighted_mean"],
                                           include_cols=["GridID"])
        assert out["weighted_mean"].iloc[0] == pytest.approx(55.0)


# --------------------------------------------------------------------------
# Real-data memory-safety smoke test (skipped if the file is absent)
# --------------------------------------------------------------------------

_PM25 = (
    "/home/sjs96_file_share/data/global/PM25/2019/"
    "V6GL02.04.CNNPM25.GL.201901-201901.nc"
)


@pytest.mark.slow
@pytest.mark.skipif(not __import__("pathlib").Path(_PM25).exists(),
                    reason="global PM2.5 netCDF not present")
def test_global_pm25_clip_is_memory_safe():
    # Clip the 13000x36000 global grid to a small Nairobi-ish bbox.
    aoi = gpd.GeoDataFrame({"id": [1]}, geometry=[box(36.6, -1.5, 37.1, -1.1)],
                           crs="EPSG:4326")
    slices = list(open_raster_slices(_PM25, aoi=aoi))
    assert len(slices) == 1
    s = slices[0]
    # The window must be tiny (~50x50 at 0.01deg), proving lazy windowed reads.
    assert s.array.size < 100_000, f"clip too large: {s.shape}"
    assert s.variable == "PM25"
    assert s.time == pd.Timestamp("2019-01-01")
    assert np.isfinite(s.array).any()
