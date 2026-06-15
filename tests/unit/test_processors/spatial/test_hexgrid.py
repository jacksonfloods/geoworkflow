"""
Unit tests for HexGridProcessor.
"""

import re

import pytest
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False

from pydantic import ValidationError as PydanticValidationError

from geoworkflow.schemas.config_models import HexGridConfig


class TestHexGridConfig:
    """Tests for HexGridConfig schema."""

    def test_config_valid(self, temp_dir, sample_aoi_small):
        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=temp_dir / "hex.geojson",
            side_length=150.0,
        )
        assert config.side_length == 150.0
        assert config.orientation == "flat_top"
        assert config.output_crs == "EPSG:4326"
        assert config.grid_origin_x == -3000000.0
        assert config.grid_origin_y == -2000000.0
        assert config.skip_existing is False

    def test_config_invalid_orientation(self, temp_dir, sample_aoi_small):
        with pytest.raises(PydanticValidationError):
            HexGridConfig(
                aoi_file=sample_aoi_small,
                output_file=temp_dir / "hex.geojson",
                orientation="pointy_top",
            )

    def test_config_invalid_side_length(self, temp_dir, sample_aoi_small):
        with pytest.raises(PydanticValidationError):
            HexGridConfig(
                aoi_file=sample_aoi_small,
                output_file=temp_dir / "hex.geojson",
                side_length=0,
            )

    def test_config_negative_side_length(self, temp_dir, sample_aoi_small):
        with pytest.raises(PydanticValidationError):
            HexGridConfig(
                aoi_file=sample_aoi_small,
                output_file=temp_dir / "hex.geojson",
                side_length=-50,
            )


@pytest.mark.skipif(not HAS_GEOSPATIAL, reason="geopandas not available")
class TestHexGridProcessor:
    """Tests for HexGridProcessor end-to-end behavior."""

    def test_generates_hexagons(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor

        output_file = temp_dir / "hex_output.geojson"
        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=output_file,
            side_length=500.0,
        )

        processor = HexGridProcessor(config)
        result = processor.process()

        assert result.success is True, f"Processing failed: {result.message}"
        assert output_file.exists()
        assert result.processed_count > 0

        gdf = gpd.read_file(output_file)
        assert len(gdf) > 0
        assert "GridID" in gdf.columns
        assert "q" in gdf.columns
        assert "r" in gdf.columns
        assert "geometry" in gdf.columns
        assert all(gdf["GridID"].str.startswith("SSA_HQ"))

    def test_output_crs(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor

        output_file = temp_dir / "hex_crs.geojson"
        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=output_file,
            side_length=500.0,
            output_crs="EPSG:4326",
        )
        processor = HexGridProcessor(config)
        result = processor.process()

        assert result.success is True
        gdf = gpd.read_file(output_file)
        assert str(gdf.crs) == "EPSG:4326"

    def test_skip_existing(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor

        output_file = temp_dir / "hex_skip.geojson"
        output_file.write_text("placeholder")

        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=output_file,
            side_length=500.0,
            skip_existing=True,
        )
        processor = HexGridProcessor(config)
        result = processor.process()

        assert result.success is True
        assert result.skipped_count == 1
        assert output_file.read_text() == "placeholder"

    def test_hexagons_intersect_aoi(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor

        output_file = temp_dir / "hex_intersect.geojson"
        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=output_file,
            side_length=500.0,
        )
        processor = HexGridProcessor(config)
        result = processor.process()
        assert result.success is True

        hex_gdf = gpd.read_file(output_file)
        aoi_gdf = gpd.read_file(sample_aoi_small).to_crs(hex_gdf.crs)
        aoi_union = aoi_gdf.geometry.union_all()

        assert hex_gdf.geometry.intersects(aoi_union).all()

    def test_grid_id_format(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor

        output_file = temp_dir / "hex_id.geojson"
        config = HexGridConfig(
            aoi_file=sample_aoi_small,
            output_file=output_file,
            side_length=500.0,
        )
        processor = HexGridProcessor(config)
        processor.process()

        gdf = gpd.read_file(output_file)
        sample_id = gdf["GridID"].iloc[0]
        # GridID format is f"SSA_HQ{q:+07d}_R{r:+07d}" -> sign + 6 zero-padded
        # digits each, matching production data (e.g. "SSA_HQ+014562_R+005142").
        assert re.fullmatch(r"SSA_HQ[+-]\d{6}_R[+-]\d{6}", sample_id), sample_id

    def test_missing_aoi_file(self, temp_dir):
        from geoworkflow.processors.spatial import HexGridProcessor

        config = HexGridConfig(
            aoi_file=temp_dir / "nonexistent.geojson",
            output_file=temp_dir / "hex.geojson",
            side_length=150.0,
        )
        processor = HexGridProcessor(config)
        result = processor.process()
        assert result.success is False


@pytest.mark.skipif(not HAS_GEOSPATIAL, reason="geopandas not available")
class TestDualMode:
    """Albers (global) and per-city UTM as side-by-side options."""

    FULL_250 = (3 * (3 ** 0.5) / 2) * 250.0 ** 2   # true area of a 250 m hex

    def test_albers_default_has_area_and_ssa_ids(self, temp_dir, sample_aoi_small):
        from geoworkflow.processors.spatial import HexGridProcessor
        out = temp_dir / "albers.geojson"
        res = HexGridProcessor(HexGridConfig(
            aoi_file=sample_aoi_small, output_file=out,  # default mode + 250 m
        )).process()
        assert res.success, res.message
        gdf = gpd.read_file(out)
        assert all(gdf["GridID"].str.startswith("SSA_HQ"))
        assert "area_m2" in gdf.columns
        assert abs(gdf["area_m2"].median() - self.FULL_250) < 50

    def test_utm_local_mode(self, temp_dir, sample_aoi_small):
        import json
        from geoworkflow.processors.spatial import HexGridProcessor
        out = temp_dir / "utm.geojson"
        res = HexGridProcessor(HexGridConfig(
            aoi_file=sample_aoi_small, output_file=out,
            side_length=250.0, crs_mode="utm_local",
        )).process()
        assert res.success, res.message
        gdf = gpd.read_file(out)
        # zone-qualified positional IDs, not SSA, not sequential
        assert all(gdf["GridID"].str.startswith("UTM")), gdf["GridID"].iloc[0]
        # area_m2 is the TRUE ground area. In UTM it differs from the nominal 250 m
        # hex area by the ~0.2% scale-factor drift (UTM isn't equal-area) -- which is
        # exactly why we store it rather than trust the geometry's planar area.
        med = gdf["area_m2"].median()
        assert abs(med - self.FULL_250) / self.FULL_250 < 0.01   # right hex, true area
        assert abs(med - self.FULL_250) > 50                     # drift is real, not nominal
        meta = json.loads(Path(str(out) + ".meta.json").read_text())
        assert meta["crs_mode"] == "utm_local"
        assert str(meta["utm_epsg"]).startswith("326")  # northern-hemisphere UTM

    def test_lattice_self_coincidence(self, temp_dir, sample_aoi_small):
        # Stored in the lattice CRS, every hex centroid must round-trip through the
        # axial math to itself -> the grid sits exactly on the global lattice (the
        # in-repo analogue of the Dakar coincidence spike).
        from geoworkflow.processors.spatial import HexGridProcessor
        from geoworkflow.processors.spatial.hexgrid import (
            _axial_to_cartesian, _cartesian_to_axial)
        out = temp_dir / "albers_native.geojson"
        HexGridProcessor(HexGridConfig(
            aoi_file=sample_aoi_small, output_file=out,
            side_length=250.0, output_crs="ESRI:102022",
        )).process()
        gdf = gpd.read_file(out)
        ox, oy, s = -3_000_000.0, -2_000_000.0, 250.0
        maxres = 0.0
        for c in gdf.geometry.centroid:
            q, r = _cartesian_to_axial(c.x, c.y, ox, oy, s)
            rx, ry = _axial_to_cartesian(q, r, ox, oy, s)
            maxres = max(maxres, ((rx - c.x) ** 2 + (ry - c.y) ** 2) ** 0.5)
        assert maxres < 1e-3, f"max round-trip residual {maxres} m"

    def test_clip_toggle(self, temp_dir):
        from shapely.geometry import Polygon
        from geoworkflow.processors.spatial import HexGridProcessor
        # a triangle: its bbox has ~2x its area, so the unclipped (bbox-fill) grid
        # must contain strictly more hexes than the AOI-clipped one.
        tri = gpd.GeoDataFrame(
            {"name": ["t"]},
            geometry=[Polygon([(-0.2, 5.5), (0.0, 5.5), (-0.2, 5.7)])], crs="EPSG:4326")
        aoi = temp_dir / "tri.geojson"; tri.to_file(aoi, driver="GeoJSON")
        clipped, unclipped = temp_dir / "c.geojson", temp_dir / "u.geojson"
        HexGridProcessor(HexGridConfig(aoi_file=aoi, output_file=clipped,
                                       side_length=250.0, clip_to_aoi=True)).process()
        HexGridProcessor(HexGridConfig(aoi_file=aoi, output_file=unclipped,
                                       side_length=250.0, clip_to_aoi=False)).process()
        assert len(gpd.read_file(unclipped)) > len(gpd.read_file(clipped))

    def test_gpkg_output_preserves_crs_and_columns(self, temp_dir, sample_aoi_small):
        # The driver is inferred from the extension; .gpkg must round-trip the UTM
        # CRS and the columns cleanly (the canonical format for the all-city run).
        from geoworkflow.processors.spatial import HexGridProcessor
        out = temp_dir / "hex_utm.gpkg"
        res = HexGridProcessor(HexGridConfig(
            aoi_file=sample_aoi_small, output_file=out,
            side_length=250.0, crs_mode="utm_local",
        )).process()
        assert res.success, res.message
        assert out.exists()
        g = gpd.read_file(out)
        assert g.crs.to_epsg() and g.crs.to_epsg() != 4326   # native UTM preserved
        assert set(["GridID", "q", "r", "area_m2"]).issubset(g.columns)
        assert all(g["GridID"].str.startswith("UTM"))
