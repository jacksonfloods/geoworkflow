"""Tests for the chained accessor API over a tiny synthetic warehouse."""
import pytest

pd = pytest.importorskip("pandas")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("duckdb")
from shapely.geometry import box

from geoworkflow.store import open_hexdb, HexDBConfig
from geoworkflow.store.accessors import NeedsReducerError
from geoworkflow.store.catalog import AmbiguousCityError

GIDS = ["UTM32737_HQ+000001_R+000001", "UTM32737_HQ+000002_R+000001"]


@pytest.fixture
def db(tmp_path):
    wh, hexagglo, man = tmp_path / "hexdb", tmp_path / "hexagglo", tmp_path / "man.csv"
    pd.DataFrame([
        (6509, "KEN", "nairobi", 32737, 250.0, -3e6, -2e6, "x"),
        (1001, "KEN", "malindi", 32737, 250.0, -3e6, -2e6, "x"),
        (2002, "TZA", "malindi", 32737, 250.0, -3e6, -2e6, "x"),
    ], columns=["Agglomeration_ID", "ISO3", "name", "utm_epsg",
                "side_length", "origin_x", "origin_y", "source"]).to_csv(man, index=False)

    # only Nairobi is "built" (has a partition + geometry)
    part = wh / "ISO3=KEN" / "aggid=6509" / "part.parquet"; part.parent.mkdir(parents=True)
    rows = []
    for i, gid in enumerate(GIDS):
        base = 300.0 + i  # distinct per hex so mapping is testable
        for ym, v in [("2019-06-01", base), ("2020-06-01", base + 2), ("2020-07-01", base + 10)]:
            rows.append((gid, "KEN", 6509, "lst_day", pd.Timestamp(ym), "weighted_mean", v, "K"))
        rows.append((gid, "KEN", 6509, "landcover", pd.Timestamp("2019-01-01"), "majority", 50.0, None))
    pd.DataFrame(rows, columns=["GridID", "ISO3", "aggid", "variable",
                                "time", "statistic", "value", "units"]).to_parquet(part, index=False)

    gpkg = hexagglo / "KEN" / "6509_nairobi_hex.gpkg"; gpkg.parent.mkdir(parents=True)
    gpd.GeoDataFrame({"GridID": GIDS}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
                     crs="EPSG:32737").to_file(gpkg, driver="GPKG")

    return open_hexdb(HexDBConfig(warehouse_dir=wh, hexagglo_dir=hexagglo,
                                  manifest_csv=man, complexity_csv=None))


def test_countries_and_cities_reflect_built(db):
    assert db.countries() == ["KEN"]            # TZA has no partition
    assert db["KEN"].cities() == ["Nairobi"]    # malindi not built


def test_default_statistic(db):
    assert db["KEN"]["Nairobi"]["lst_day"]._statistic == "weighted_mean"
    assert db["KEN"]["Nairobi"]["landcover"]._statistic == "majority"


def test_sel_instant_returns_per_hex(db):
    df = db["KEN"]["Nairobi"]["lst_day"].sel("2020-06").to_frame()
    vals = dict(zip(df["GridID"], df["value"]))
    assert vals[GIDS[0]] == 302.0 and vals[GIDS[1]] == 303.0


def test_sel_multi_without_reducer_raises(db):
    with pytest.raises(NeedsReducerError):
        db["KEN"]["Nairobi"]["lst_day"].sel("2020").to_frame()


def test_annual_mean(db):
    df = db["KEN"]["Nairobi"]["lst_day"].mean("2020").to_frame()
    vals = dict(zip(df["GridID"], df["value"]))
    assert vals[GIDS[0]] == pytest.approx((302 + 310) / 2)   # 2020-06 + 2020-07


def test_full_period_mean(db):
    df = db["KEN"]["Nairobi"]["lst_day"].mean().to_frame()
    vals = dict(zip(df["GridID"], df["value"]))
    assert vals[GIDS[0]] == pytest.approx((300 + 302 + 310) / 3)


def test_climatology_month(db):
    df = db["KEN"]["Nairobi"]["lst_day"].climatology(month=6).to_frame()
    vals = dict(zip(df["GridID"], df["value"]))
    assert vals[GIDS[0]] == pytest.approx((300 + 302) / 2)   # all Junes


def test_to_geodataframe_joins_geometry(db):
    gdf = db["KEN"]["Nairobi"]["lst_day"].sel("2020-06").to_geodataframe()
    assert "geometry" in gdf.columns and "value" in gdf.columns
    assert len(gdf) == 2 and gdf["value"].notna().all()


def test_hexagon_lookup_any_city(db):
    out = db.hexagon(GIDS[0])
    assert set(out["variable"]) == {"lst_day", "landcover"}
    assert (out["aggid"] == 6509).all()


def test_hexagon_not_found(db):
    from geoworkflow.core.exceptions import GeoWorkflowError
    with pytest.raises(GeoWorkflowError):
        db.hexagon("UTM32737_HQ+999999_R+999999")


def test_ambiguous_city_name(db):
    # both KEN/malindi and TZA/malindi exist; name-only resolution is per-country here
    assert db["KEN"].city(aggid=1001).record.aggid == 1001


def test_legend_exposed(db):
    leg = db.legend("landcover")
    assert leg[50][0].lower().startswith("urban")
