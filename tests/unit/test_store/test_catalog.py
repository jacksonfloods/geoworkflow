"""Tests for the manifest-driven city catalog (synthetic data, no network)."""
import pytest

pd = pytest.importorskip("pandas")

from geoworkflow.store.catalog import (
    Catalog, CityRecord, AmbiguousCityError, CityNotFoundError,
)

MANIFEST_ROWS = [
    # Agglomeration_ID, ISO3, name, utm_epsg, side_length, origin_x, origin_y, source
    (6509, "KEN", "nairobi", 32737, 250.0, -3e6, -2e6, "estimate_utm_crs"),
    (1001, "KEN", "malindi", 32737, 250.0, -3e6, -2e6, "estimate_utm_crs"),
    (2002, "TZA", "malindi", 32737, 250.0, -3e6, -2e6, "estimate_utm_crs"),  # dup name, diff ISO3
    (4858, "COD", "kinshasa", 32733, 250.0, -3e6, -2e6, "estimate_utm_crs"),
]


@pytest.fixture
def catalog(tmp_path):
    man = tmp_path / "grid_zone_manifest.csv"
    pd.DataFrame(MANIFEST_ROWS, columns=[
        "Agglomeration_ID", "ISO3", "name", "utm_epsg",
        "side_length", "origin_x", "origin_y", "source",
    ]).to_csv(man, index=False)

    cx = tmp_path / "complexity.csv"
    pd.DataFrame([
        ("Nairobi", "KEN", 6509.0, 15534),
        ("Kinshasa", "COD", 4858.0, 50000),
    ], columns=["Agglomeration_Name", "ISO3", "Agglomeration_ID", "n_nodes"]).to_csv(cx, index=False)

    # legacy grid_stats: one aggid-named, one bare-named (the 7-city quirk)
    gs = tmp_path / "grid_stats"; gs.mkdir()
    (gs / "6509_nairobi_combined.parquet").write_bytes(b"x")
    (gs / "kinshasa_combined.parquet").write_bytes(b"x")   # bare name

    return Catalog(man, tmp_path / "hexagglo", tmp_path / "hexdb",
                   complexity_csv=cx, grid_stats_dir=gs)


def test_resolve_aggid(catalog):
    rec = catalog.resolve_aggid(6509)
    assert rec.iso3 == "KEN" and rec.name == "nairobi"
    assert rec.display_name == "Nairobi"        # from complexity
    assert rec.utm_epsg == 32737 and rec.n_nodes == 15534
    assert rec.gpkg.name == "6509_nairobi_hex.gpkg"
    assert rec.partition.as_posix().endswith("ISO3=KEN/aggid=6509/part.parquet")


def test_resolve_by_name_case_insensitive(catalog):
    assert catalog.resolve("KEN", "Nairobi").aggid == 6509
    assert catalog.resolve("KEN", "  nairobi ").aggid == 6509


def test_display_name_fallback_titlecase(catalog):
    # malindi has no complexity row -> title-cased slug
    assert catalog.resolve("KEN", "malindi").display_name == "Malindi"


def test_ambiguous_only_within_iso3(catalog):
    # two "malindi" but in different ISO3 -> each unambiguous within its country
    assert catalog.resolve("KEN", "malindi").aggid == 1001
    assert catalog.resolve("TZA", "malindi").aggid == 2002


def test_resolve_missing(catalog):
    with pytest.raises(CityNotFoundError):
        catalog.resolve("KEN", "atlantis")


def test_resolve_aggid_wrong_iso3(catalog):
    with pytest.raises(CityNotFoundError):
        catalog.resolve("TZA", aggid=6509)


def test_bare_name_parquet_fallback(catalog):
    assert catalog.resolve_aggid(6509).combined_parquet.name == "6509_nairobi_combined.parquet"
    assert catalog.resolve_aggid(4858).combined_parquet.name == "kinshasa_combined.parquet"
    assert catalog.resolve_aggid(1001).combined_parquet is None  # neither exists


def test_find_hexagon_prunes_by_epsg(catalog):
    cands = catalog.find_hexagon("UTM32737_HQ-008225_R+031543")
    aggids = {c.aggid for c in cands}
    assert aggids == {6509, 1001, 2002}        # all zone-32737 cities
    assert catalog.find_hexagon("UTM32733_HQ+0_R+0")[0].aggid == 4858
    assert catalog.find_hexagon("not-a-gridid") == []


def test_listings(catalog):
    assert catalog.countries() == ["COD", "KEN", "TZA"]
    assert "Nairobi" in catalog.cities("KEN")


def test_resolve_selector(catalog):
    from geoworkflow.schemas.config_models import CitySelector
    assert len(catalog.resolve_selector(CitySelector(all=True))) == 4
    assert {r.aggid for r in catalog.resolve_selector(CitySelector(iso3=["KEN"]))} == {6509, 1001}
    assert {r.aggid for r in catalog.resolve_selector(CitySelector(aggid=[6509, 4858]))} == {6509, 4858}
    with pytest.raises(AmbiguousCityError):
        catalog.resolve_selector(CitySelector(name=["malindi"]))   # ambiguous across ISO3
    assert catalog.resolve_selector(CitySelector(name=["nairobi"]))[0].aggid == 6509
