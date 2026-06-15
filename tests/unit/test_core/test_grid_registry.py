"""Unit tests for the UTM zone master list (grid zone manifest)."""

import pytest

try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS = True
except ImportError:
    HAS = False

pytestmark = pytest.mark.skipif(not HAS, reason="geopandas not available")

if HAS:
    from geoworkflow.core.grid_registry import (
        load_zone_manifest, resolve_utm_epsg, epsg_from_grid_file)


def _aoi(lon, lat):
    return gpd.GeoDataFrame(
        {"x": [1]}, geometry=[box(lon, lat, lon + 0.05, lat + 0.05)], crs="EPSG:4326")


class TestZoneManifest:
    def test_resolve_computes_and_appends(self, temp_dir):
        m = temp_dir / "zones.csv"
        # Dakar ~17.45 W, 14.69 N -> UTM 28N = EPSG:32628 (matches his file)
        epsg = resolve_utm_epsg(6509, _aoi(-17.45, 14.69), manifest_path=m,
                                iso3="SEN", name="Dakar")
        assert epsg == 32628
        man = load_zone_manifest(m)
        assert 6509 in man and int(man[6509]["utm_epsg"]) == 32628
        assert man[6509]["source"] == "estimate_utm_crs"

    def test_pin_is_stable_against_aoi_change(self, temp_dir):
        m = temp_dir / "zones.csv"
        resolve_utm_epsg(1, _aoi(-17.45, 14.69), manifest_path=m)        # records 32628
        # later the AOI "moves" across a zone line (lon 10 -> would be 32632);
        # the pinned zone must NOT change.
        epsg2 = resolve_utm_epsg(1, _aoi(10.0, 14.69), manifest_path=m)
        assert epsg2 == 32628

    def test_override_recorded_as_morphology(self, temp_dir):
        m = temp_dir / "zones.csv"
        epsg = resolve_utm_epsg(2, _aoi(0.0, 5.5), manifest_path=m, override_epsg=32630)
        assert epsg == 32630
        assert load_zone_manifest(m)[2]["source"] == "morphology_grid"

    def test_append_false_writes_nothing(self, temp_dir):
        m = temp_dir / "zones.csv"
        resolve_utm_epsg(3, _aoi(0.0, 5.5), manifest_path=m, append=False)
        assert load_zone_manifest(m) == {}

    def test_load_missing_is_empty(self, temp_dir):
        assert load_zone_manifest(temp_dir / "nope.csv") == {}

    def test_epsg_from_grid_file(self, temp_dir):
        g = gpd.GeoDataFrame({"GridID": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32630")
        p = temp_dir / "g.gpkg"; g.to_file(p, driver="GPKG")
        assert epsg_from_grid_file(p) == 32630
