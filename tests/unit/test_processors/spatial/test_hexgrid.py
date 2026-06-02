"""
Unit tests for HexGridProcessor.
"""

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
        assert len(sample_id) == len("SSA_HQ+0000000_R+0000000")
        assert sample_id.startswith("SSA_HQ")
        assert "_R" in sample_id

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
