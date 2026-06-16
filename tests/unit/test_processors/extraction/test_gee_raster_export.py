"""Unit tests for the declarative GEE raster export processor (pure logic only —
no Earth Engine network calls)."""

import pytest

try:
    import geopandas as gpd  # noqa: F401
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="geopandas not available")

if HAS_LIBS:
    from geoworkflow.schemas.config_models import GEERasterExportConfig
    from geoworkflow.processors.extraction.gee_raster_export import (
        GEERasterExportProcessor, iter_periods, city_slug, qc_band_for,
    )


class TestPeriods:
    def test_monthly_spans_years(self):
        periods = iter_periods("monthly", "2019-11", "2020-02")
        assert [p[0] for p in periods] == ["201911", "201912", "202001", "202002"]
        # each period is [month start, next month start)
        assert periods[1][1:] == ("2019-12-01", "2020-01-01")
        assert periods[3][1:] == ("2020-02-01", "2020-03-01")

    def test_yearly(self):
        periods = iter_periods("yearly", "2019", "2021")
        assert [p[0] for p in periods] == ["2019", "2020", "2021"]
        assert periods[0][1:] == ("2019-01-01", "2020-01-01")

    def test_static_single_period(self):
        assert iter_periods("static", None, None) == [(None, None, None)]


class TestNaming:
    def test_city_slug_strips_id_and_hex(self):
        assert city_slug("hexagglo/COD/4858_kinshasa_hex.geojson") == "kinshasa"
        assert city_slug("dar_es_salaam_hex.geojson") == "dar_es_salaam"
        assert city_slug("nairobi.gpkg") == "nairobi"

    def test_qc_band_day_night_pairing(self):
        assert qc_band_for("LST_Night_1km", "QC_Day") == "QC_Night"
        assert qc_band_for("LST_Day_1km", "QC_Day") == "QC_Day"
        assert qc_band_for("anything", None) is None


class TestConfigAndDiscovery:
    def _grid_tree(self, temp_dir):
        # hexagglo-style layout with two countries
        for iso, name in [("COD", "4858_kinshasa"), ("KEN", "6509_nairobi")]:
            d = temp_dir / "hexagglo" / iso
            d.mkdir(parents=True)
            (d / f"{name}_hex.geojson").write_text("{}")
        return temp_dir / "hexagglo"

    def test_requires_exactly_one_target_mode(self, temp_dir):
        with pytest.raises(ValueError, match="exactly one"):
            GEERasterExportConfig(source="X", bands=["b"], output_dir=temp_dir,
                                  dataset="d", scale_m=100)

    def test_monthly_requires_dates(self, temp_dir):
        with pytest.raises(ValueError, match="requires start and end"):
            GEERasterExportConfig(source="X", bands=["b"], cadence="monthly",
                                  grid_dir=temp_dir, output_dir=temp_dir,
                                  dataset="d", scale_m=100)

    def test_discovery_and_filenames_match_lab_layout(self, temp_dir):
        grid_dir = self._grid_tree(temp_dir)
        cfg = GEERasterExportConfig(
            source="MODIS/061/MOD11A1", bands=["LST_Day_1km", "LST_Night_1km"],
            band_tags={"LST_Day_1km": "day", "LST_Night_1km": "night"},
            cadence="monthly", start="2019-01", end="2019-01",
            grid_dir=grid_dir, output_dir=temp_dir / "city",
            dataset="mod11a1_lst",
            filename_template="{city}_lst_{band_tag}_{time}.tif", scale_m=1000,
        )
        proc = GEERasterExportProcessor(cfg)
        targets = proc._discover_targets()
        assert [(i, c) for i, c, _ in targets] == [
            ("COD", "kinshasa"), ("KEN", "nairobi")]
        # filenames must match what the registry already parses
        assert proc._filename("kinshasa", "COD", "LST_Day_1km", "201901") == \
            "kinshasa_lst_day_201901.tif"
        assert proc._filename("kinshasa", "COD", "LST_Night_1km", "201901") == \
            "kinshasa_lst_night_201901.tif"

    def test_static_filename_uses_label(self, temp_dir):
        grid_dir = self._grid_tree(temp_dir)
        cfg = GEERasterExportConfig(
            source="COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019",
            bands=["discrete_classification"],
            grid_dir=grid_dir, output_dir=temp_dir / "city",
            dataset="copernicus_lc100",
            filename_template="{city}_lc100_{time}.tif", static_label="2019",
            scale_m=100,
        )
        proc = GEERasterExportProcessor(cfg)
        assert proc._filename("nairobi", "KEN", "discrete_classification", None) == \
            "nairobi_lc100_2019.tif"


class TestBatchSlicing:
    """The batched path: one multi-band download split into the same per-file
    outputs. The slicing is pure rasterio — no Earth Engine — so it's unit-tested
    directly to guarantee byte-equivalence with the per-request product."""

    def _proc(self, temp_dir):
        cfg = GEERasterExportConfig(
            source="X", bands=["b"], cadence="static",
            grid_dir=temp_dir, output_dir=temp_dir, dataset="d", scale_m=100)
        return GEERasterExportProcessor(cfg)

    def _multiband_bytes(self, n_bands):
        import numpy as np
        rasterio = pytest.importorskip("rasterio")
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from rasterio.crs import CRS
        data = np.stack([np.full((4, 5), i + 1, dtype="float32") for i in range(n_bands)])
        transform = from_origin(36.0, -1.0, 0.01, 0.01)
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", height=4, width=5, count=n_bands,
                          dtype="float32", crs=CRS.from_epsg(4326),
                          transform=transform) as dst:
                dst.write(data)
            return mem.read(), transform

    def test_batch_size_defaults_to_none(self, temp_dir):
        assert self._proc(temp_dir).gee_config.batch_size is None

    def test_slice_splits_bands_preserving_grid(self, temp_dir):
        import numpy as np
        rasterio = pytest.importorskip("rasterio")
        content, transform = self._multiband_bytes(3)
        outs = [temp_dir / f"band_{i}.tif" for i in range(3)]
        self._proc(temp_dir)._slice_to_files(content, outs)
        for i, op in enumerate(outs):
            assert op.exists()
            with rasterio.open(op) as src:
                assert src.count == 1                       # split to single-band
                assert src.crs.to_epsg() == 4326            # CRS preserved
                assert src.transform == transform           # georeferencing preserved
                assert np.allclose(src.read(1), i + 1)      # band i -> out_paths[i]

    def test_slice_band_count_mismatch_raises(self, temp_dir):
        from geoworkflow.core.exceptions import ProcessingError
        content, _ = self._multiband_bytes(2)
        outs = [temp_dir / f"band_{i}.tif" for i in range(3)]   # 3 paths, 2 bands
        with pytest.raises(ProcessingError, match="expected 3 bands"):
            self._proc(temp_dir)._slice_to_files(content, outs)
