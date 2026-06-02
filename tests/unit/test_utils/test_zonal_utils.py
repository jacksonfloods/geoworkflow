"""
Unit tests for the coverage-weighted zonal statistics engine
(geoworkflow.utils.zonal_utils).

A 2x2, 1-degree raster is used so coverage fractions are easy to reason about::

    row 0:   0   10        (lat 1..2)
    row 1:  20   30        (lat 0..1)
    cols:  lon 0..1 | 1..2
"""

import numpy as np
import pytest

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from exactextract import exact_extract  # noqa: F401  (presence check)
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="zonal stack not available")

if HAS_LIBS:
    from geoworkflow.utils.zonal_utils import compute_zonal_statistics


@pytest.fixture
def raster_2x2(temp_dir):
    """Write the 2x2 reference raster and return its path."""
    path = temp_dir / "ref.tif"
    data = np.array([[0.0, 10.0], [20.0, 30.0]], dtype="float32")
    profile = dict(
        driver="GTiff", height=2, width=2, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(0, 2, 1, 1), nodata=None,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def hexes():
    """Three test polygons in EPSG:4326 with stable GridIDs."""
    poly_a = box(0.1, 1.1, 0.4, 1.4)    # inside top-left cell (value 0)
    poly_b = box(0.25, 1.1, 1.75, 1.9)  # straddles row-0 cells (values 0 and 10)
    poly_c = box(1.2, 0.2, 1.4, 0.4)    # inside bottom-right cell (value 30)
    return gpd.GeoDataFrame(
        {"GridID": ["A", "B", "C"]},
        geometry=[poly_a, poly_b, poly_c],
        crs="EPSG:4326",
    )


class TestWeightedMean:
    def test_coverage_weighted_mean(self, raster_2x2, hexes):
        out = compute_zonal_statistics(
            raster_2x2, hexes, ["weighted_mean"], include_cols=["GridID"]
        )
        result = dict(zip(out["GridID"], out["weighted_mean"]))
        assert result["A"] == pytest.approx(0.0)   # entirely inside value-0 pixel
        assert result["B"] == pytest.approx(5.0)   # area-weighted avg of 0 and 10
        assert result["C"] == pytest.approx(30.0)  # entirely inside value-30 pixel

    def test_subpixel_polygon_is_not_null(self, raster_2x2, hexes):
        # The whole point of exactextract over rasterstats: a polygon smaller
        # than a pixel still gets a value, not NaN.
        out = compute_zonal_statistics(raster_2x2, hexes, ["weighted_mean"])
        assert out["weighted_mean"].notna().all()

    def test_output_row_order_matches_input(self, raster_2x2, hexes):
        out = compute_zonal_statistics(
            raster_2x2, hexes, ["weighted_mean"], include_cols=["GridID"]
        )
        assert list(out["GridID"]) == ["A", "B", "C"]


class TestMultipleStatistics:
    def test_op_and_reducer_together(self, raster_2x2, hexes):
        out = compute_zonal_statistics(
            raster_2x2, hexes, ["min", "max", "range", "p25"], include_cols=["GridID"]
        )
        row_b = out[out["GridID"] == "B"].iloc[0]
        assert row_b["min"] == pytest.approx(0.0)
        assert row_b["max"] == pytest.approx(10.0)
        assert row_b["range"] == pytest.approx(10.0)  # reducer-backed
        # quantile column resolves and is between min and max
        assert 0.0 <= row_b["p25"] <= 10.0

    def test_alias_and_canonical_share_op(self, raster_2x2, hexes):
        out = compute_zonal_statistics(raster_2x2, hexes, ["mean", "weighted_mean"])
        assert out["mean"].equals(out["weighted_mean"])


class TestCrsAndEdges:
    def test_reprojects_vectors_to_raster_crs(self, raster_2x2, hexes):
        hexes_3857 = hexes.to_crs("EPSG:3857")
        out = compute_zonal_statistics(
            raster_2x2, hexes_3857, ["weighted_mean"], include_cols=["GridID"]
        )
        result = dict(zip(out["GridID"], out["weighted_mean"]))
        # A and C are well inside single cells, so reprojection round-trip is safe.
        assert result["A"] == pytest.approx(0.0)
        assert result["C"] == pytest.approx(30.0)

    def test_non_overlapping_polygon_is_nan(self, raster_2x2):
        far = gpd.GeoDataFrame(
            {"GridID": ["far"]}, geometry=[box(50, 50, 51, 51)], crs="EPSG:4326"
        )
        out = compute_zonal_statistics(raster_2x2, far, ["weighted_mean"])
        assert np.isnan(out["weighted_mean"].iloc[0])

    def test_accepts_open_dataset(self, raster_2x2, hexes):
        with rasterio.open(raster_2x2) as src:
            out = compute_zonal_statistics(src, hexes, ["weighted_mean"])
        assert len(out) == 3
