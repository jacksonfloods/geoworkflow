"""
Unit tests for OSM Highways processor.


Tests cover:
- Configuration validation
- Region detection
- Highway filtering
- Attribute selection
- Geometry processing
- Export functionality
"""


import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile


try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString, box
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False


pytestmark = pytest.mark.skipif(
    not HAS_GEOSPATIAL_LIBS,
    reason="Geospatial libraries not available"
)


from geoworkflow.processors.extraction.osm_highways import OSMHighwaysProcessor
from geoworkflow.schemas.osm_highways_config import OSMHighwaysConfig
from geoworkflow.utils.osm_utils import (
    filter_highways_by_type,
    select_highway_attributes,
    validate_highway_geometries
)




# ==================== FIXTURES ====================


@pytest.fixture
def tmp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)




@pytest.fixture
def sample_aoi(tmp_dir):
    """Create sample AOI GeoJSON."""
    aoi_file = tmp_dir / "test_aoi.geojson"
    gdf = gpd.GeoDataFrame(
        {'name': ['test_area']},
        geometry=[box(36.8, -1.3, 36.9, -1.2)],  # Nairobi area
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file




@pytest.fixture
def sample_highways():
    """Create sample highway GeoDataFrame."""
    data = {
        'osm_id': [1, 2, 3, 4],
        'highway': ['motorway', 'residential', 'footway', 'primary'],
        'name': ['Highway 1', 'Main St', None, 'Route 2'],
        'surface': ['asphalt', 'paved', 'unpaved', 'concrete'],
        'lanes': ['4', '2', None, '3'],
        'geometry': [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(2, 2), (3, 3)]),
            LineString([(3, 3), (4, 4)])
        ]
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")




# ==================== CONFIG TESTS ====================


class TestOSMHighwaysConfig:
    """Test configuration model."""
    
    def test_valid_config(self, sample_aoi, tmp_dir):
        """Test valid configuration creation."""
        config = OSMHighwaysConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_dir,
            geofabrik_regions=["kenya"],
            highway_types=["motorway", "primary"],
            include_attributes=["highway", "name", "surface"]
        )
        
        assert config.aoi_file == sample_aoi
        assert config.geofabrik_regions == ["kenya"]
        assert "highway" in config.include_attributes
    
    def test_missing_aoi_file(self, tmp_dir):
        """Test error on missing AOI file."""
        with pytest.raises(ValueError, match="AOI file not found"):
            OSMHighwaysConfig(
                aoi_file=tmp_dir / "nonexistent.geojson",
                output_dir=tmp_dir
            )
    
    def test_invalid_highway_types(self, sample_aoi, tmp_dir):
        """Test warning on invalid highway types."""
        with pytest.warns(UserWarning, match="Unrecognized highway types"):
            config = OSMHighwaysConfig(
                aoi_file=sample_aoi,
                output_dir=tmp_dir,
                highway_types=["motorway", "fake_type", "flying_car_lane"]
            )
    
    def test_auto_add_highway_attribute(self, sample_aoi, tmp_dir):
        """Test automatic addition of 'highway' attribute."""
        with pytest.warns(UserWarning, match="Adding 'highway'"):
            config = OSMHighwaysConfig(
                aoi_file=sample_aoi,
                output_dir=tmp_dir,
                include_attributes=["name", "surface"]  # Missing 'highway'
            )
            assert "highway" in config.include_attributes




# ==================== UTILITY FUNCTION TESTS ====================


class TestOSMUtils:
    """Test OSM utility functions."""
    
    def test_filter_highways_by_type(self, sample_highways):
        """Test highway type filtering."""
        filtered = filter_highways_by_type(
            sample_highways,
            ["motorway", "primary"]
        )
        assert len(filtered) == 2
        assert set(filtered['highway']) == {'motorway', 'primary'}
    
    def test_select_attributes(self, sample_highways):
        """Test attribute selection."""
        selected = select_highway_attributes(
            sample_highways,
            ["highway", "name", "surface"]
        )
        assert "lanes" not in selected.columns
        assert "highway" in selected.columns
        assert "geometry" in selected.columns
    
    def test_validate_geometries(self, sample_highways):
        """Test geometry validation."""
        # Add an invalid geometry
        bad_geom = sample_highways.copy()
        bad_geom.loc[0, 'geometry'] = None
        
        validated = validate_highway_geometries(bad_geom)
        assert len(validated) == 3  # One removed




# ==================== PROCESSOR TESTS ====================


class TestOSMHighwaysProcessor:
    """Test main processor."""
    
    def test_initialization(self, sample_aoi, tmp_dir):
        """Test processor initialization."""
        config = OSMHighwaysConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_dir,
            geofabrik_regions=["kenya"]
        )
        
        processor = OSMHighwaysProcessor(config)
        assert processor.highways_config.aoi_file == sample_aoi
        assert processor.regions == []  # Not set until setup
    
    @patch('geoworkflow.processors.extraction.osm_highways.get_cached_pbf')
    @patch('pyrosm.OSM')
    def test_full_processing_flow(
        self,
        mock_osm,
        mock_get_pbf,
        sample_aoi,
        sample_highways,
        tmp_dir
    ):
        """Test complete processing workflow (mocked)."""
        # Mock PBF download
        mock_pbf_path = tmp_dir / "kenya-latest.osm.pbf"
        mock_pbf_path.touch()
        mock_metadata = Mock()
        mock_metadata.age_days.return_value = 5
        mock_metadata.file_size_mb = 100.0
        mock_get_pbf.return_value = (mock_pbf_path, mock_metadata)
        
        # Mock Pyrosm
        mock_osm_instance = MagicMock()
        mock_osm_instance.get_network.return_value = sample_highways
        mock_osm.return_value = mock_osm_instance
        
        # Create config
        config = OSMHighwaysConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_dir,
            geofabrik_regions=["kenya"],
            highway_types=["motorway", "primary"],
            export_format="geojson"
        )
        
        # Process
        processor = OSMHighwaysProcessor(config)
        result = processor.process()
        
        # Assertions
        assert result.success
        assert result.processed_count > 0
        assert processor.output_file is not None
        assert processor.output_file.exists()




# ==================== INTEGRATION TESTS ====================


@pytest.mark.integration
class TestIntegration:
    """
    Integration tests requiring actual PBF files.
    
    These tests are skipped by default. Run with:
        pytest -m integration
    """
    
    def test_real_extraction(self, sample_aoi, tmp_dir):
        """Test with real Geofabrik download (slow)."""
        pytest.skip("Integration test - requires network and time")
        
        config = OSMHighwaysConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_dir,
            geofabrik_regions=["kenya"],
            force_redownload=False
        )
        
        processor = OSMHighwaysProcessor(config)
        result = processor.process()
        
        assert result.success
        assert processor.output_file.exists()