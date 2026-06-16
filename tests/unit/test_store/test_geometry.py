"""Tests for the lazy geometry join + multi-city reprojection helpers."""
import pytest

pd = pytest.importorskip("pandas")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from geoworkflow.store import geometry as geom


@pytest.fixture
def gpkg(tmp_path):
    path = tmp_path / "city_hex.gpkg"
    gpd.GeoDataFrame(
        {"GridID": ["A", "B", "C"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:32737",
    ).to_file(path, driver="GPKG")
    return path


def test_load_hex_geometry(gpkg):
    gdf = geom.load_hex_geometry(gpkg)
    assert list(gdf.columns) == ["GridID", "geometry"]
    assert len(gdf) == 3 and gdf.crs.to_epsg() == 32737


def test_load_hex_geometry_subset(gpkg):
    gdf = geom.load_hex_geometry(gpkg, gridids=["A", "C"])
    assert set(gdf["GridID"]) == {"A", "C"}


def test_join_values_maps_by_gridid_with_nan(gpkg):
    gdf = geom.load_hex_geometry(gpkg)
    values = pd.DataFrame({"GridID": ["A", "B"], "value": [10.0, 20.0]})
    joined = geom.join_values(gdf, values)
    by = dict(zip(joined["GridID"], joined["value"]))
    assert by["A"] == 10.0 and by["B"] == 20.0
    assert pd.isna(by["C"])          # absent hex -> NaN (uncoloured)


def test_concat_for_plot_reprojects(tmp_path):
    a = gpd.GeoDataFrame({"GridID": ["A"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32737")
    b = gpd.GeoDataFrame({"GridID": ["B"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32633")
    out = geom.concat_for_plot([a, b], target_crs="EPSG:4326")
    assert out.crs.to_epsg() == 4326 and len(out) == 2
