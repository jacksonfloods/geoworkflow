"""Tests for the builder + living provenance (reuse path; no rasters needed)."""
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("duckdb")

from geoworkflow.schemas.config_models import (
    HexDBConfig, HexDBRecipe, GridSpec, CitySelector, TimeRange, MetricSpec,
)
from geoworkflow.store import builder, provenance
from geoworkflow.core.exceptions import GeoWorkflowError


def _tidy(gid_prefix, n=2):
    rows = []
    for i in range(n):
        gid = f"{gid_prefix}{i:06d}"
        for ym in ["2019-06-01", "2020-06-01"]:
            rows.append((gid, "lst_day", pd.Timestamp(ym), "weighted_mean", 300.0 + i, "K"))
    return pd.DataFrame(rows, columns=["GridID", "variable", "time", "statistic", "value", "units"])


@pytest.fixture
def setup(tmp_path):
    wh = tmp_path / "hexdb"
    man = tmp_path / "man.csv"
    pd.DataFrame([
        (6509, "KEN", "nairobi", 32737, 250.0, -3e6, -2e6, "x"),
        (4858, "COD", "kinshasa", 32733, 250.0, -3e6, -2e6, "x"),
    ], columns=["Agglomeration_ID", "ISO3", "name", "utm_epsg",
                "side_length", "origin_x", "origin_y", "source"]).to_csv(man, index=False)

    # legacy grid_stats parquets + (empty) gpkg placeholders so gpkg.exists() is True
    gs = tmp_path / "grid_stats"; gs.mkdir()
    _tidy("UTM32737_HQ").to_parquet(gs / "6509_nairobi_combined.parquet", index=False)
    _tidy("UTM32733_HQ").to_parquet(gs / "kinshasa_combined.parquet", index=False)  # bare name
    hexagglo = tmp_path / "hexagglo"
    for iso, fn in [("KEN", "6509_nairobi_hex.gpkg"), ("COD", "4858_kinshasa_hex.gpkg")]:
        d = hexagglo / iso; d.mkdir(parents=True); (d / fn).write_bytes(b"x")

    config = HexDBConfig(warehouse_dir=wh, hexagglo_dir=hexagglo, manifest_csv=man,
                         complexity_csv=None, raster_registry=None,
                         global_dir=tmp_path / "global", city_dir=tmp_path / "city")
    recipe = HexDBRecipe(
        grid=GridSpec(side_length=250.0, crs_mode="utm_local"),
        cities=CitySelector(all=True), time=TimeRange(),
        metrics=[MetricSpec(name="mod11a1_lst", scope="city", statistics=["weighted_mean"])],
    )
    return config, recipe, gs, wh


def test_build_reuses_and_writes_provenance(setup):
    config, recipe, gs, wh = setup
    res = builder.build(recipe, config, max_workers=2, grid_stats_dir=gs, reuse=True)
    assert res.done == 2 and res.failed == 0
    # partitions written
    assert (wh / "ISO3=KEN" / "aggid=6509" / "part.parquet").exists()
    df = pd.read_parquet(wh / "ISO3=COD" / "aggid=4858" / "part.parquet")
    assert set(df.columns) >= {"GridID", "ISO3", "aggid", "variable", "value"}
    assert (df["ISO3"] == "COD").all() and (df["aggid"] == 4858).all()
    # provenance
    assert provenance.read_recipe(wh) == recipe
    log = provenance.read_log(wh)
    assert log and log[-1]["op"] == "build"


def test_build_is_resumable(setup):
    config, recipe, gs, wh = setup
    builder.build(recipe, config, max_workers=2, grid_stats_dir=gs)
    res2 = builder.build(recipe, config, max_workers=2, grid_stats_dir=gs)
    assert res2.skipped == 2 and res2.done == 0


def test_consistency_guard_rejects_mismatched_grid(setup):
    config, recipe, gs, wh = setup
    builder.build(recipe, config, max_workers=2, grid_stats_dir=gs)
    bad = recipe.model_copy(deep=True)
    bad.grid.side_length = 500.0
    with pytest.raises(GeoWorkflowError):
        builder.build(bad, config, max_workers=2, grid_stats_dir=gs)


def test_remove_city_updates_log(setup):
    config, recipe, gs, wh = setup
    builder.build(recipe, config, max_workers=2, grid_stats_dir=gs)
    status = builder.remove_city(config, "KEN", "nairobi")
    assert status == "removed"
    assert not (wh / "ISO3=KEN" / "aggid=6509" / "part.parquet").exists()
    assert provenance.read_log(wh)[-1]["op"] == "remove-city"


def test_add_city_uses_live_recipe(setup):
    config, recipe, gs, wh = setup
    builder.build(recipe, config, max_workers=2, grid_stats_dir=gs)
    builder.remove_city(config, "COD", "kinshasa")
    status = builder.add_city(config, "COD", "kinshasa", grid_stats_dir=gs)
    # no reuse in add_city -> needs rasters it doesn't have -> skip-no-rasters
    assert status in ("built", "skip-no-rasters", "reused")
    assert provenance.read_log(wh)[-1]["op"] == "add-city"
