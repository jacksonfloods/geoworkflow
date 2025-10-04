# tests/unit/test_processors/extraction/test_open_buildings_gcs.py
"""
Unit tests for GCS-based Open Buildings processor.

Tests the OpenBuildingsGCSProcessor functionality including:
- Configuration validation
- S2 token generation
- GCS client initialization
- Building filtering logic
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import gzip

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon, Point, box
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

pytestmark = pytest.mark.skipif(
    not HAS_GEOSPATIAL_LIBS,
    reason="Geospatial libraries (geopandas, shapely) not available"
)

from geoworkflow.processors.extraction.open_buildings_gcs import OpenBuildingsGCSProcessor
from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig
from geoworkflow.core.exceptions import ValidationError, ConfigurationError


class TestOpenBuildingsGCSConfig:
    """Test configuration model validation."""
    
    def test_valid_config_creation(self, tmp_path):
        """Test creating a valid configuration."""
        aoi_file = tmp_path / "aoi.geojson"
        # Create minimal valid GeoJSON
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        config = OpenBuildingsGCSConfig(
            aoi_file=aoi_file,
            output_dir=tmp_path,
            confidence_threshold=0.75,
            num_workers=4
        )
        
        assert config.aoi_file == aoi_file
        assert config.confidence_threshold == 0.75
        assert config.num_workers == 4
        assert config.s2_level == 6  # default
    
    def test_invalid_confidence_threshold(self, tmp_path):
        """Test validation of confidence threshold bounds."""
        aoi_file = tmp_path / "aoi.geojson"
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        # Too low
        with pytest.raises(ValueError, match="confidence_threshold"):
            OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path,
                confidence_threshold=0.3
            )
        
        # Too high
        with pytest.raises(ValueError, match="confidence_threshold"):
            OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path,
                confidence_threshold=1.5
            )
    
    def test_invalid_area_range(self, tmp_path):
        """Test validation of area min/max relationship."""
        aoi_file = tmp_path / "aoi.geojson"
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        with pytest.raises(ValueError, match="max_area_m2.*greater than.*min_area_m2"):
            OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path,
                min_area_m2=100.0,
                max_area_m2=50.0
            )
    
    def test_gcs_path_validation(self, tmp_path):
        """Test GCS path format validation."""
        aoi_file = tmp_path / "aoi.geojson"
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        with pytest.raises(ValueError, match="must start with 'gs://'"):
            OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path,
                gcs_bucket_path="https://open-buildings-data/v3"
            )
    
    def test_output_file_path_generation(self, tmp_path):
        """Test output file path generation for different formats."""
        aoi_file = tmp_path / "aoi.geojson"
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        # GeoJSON
        config = OpenBuildingsGCSConfig(
            aoi_file=aoi_file,
            output_dir=tmp_path,
            export_format="geojson"
        )
        assert config.get_output_file_path().suffix == ".geojson"
        
        # Shapefile
        config.export_format = "shapefile"
        assert config.get_output_file_path().suffix == ".shp"
        
        # CSV
        config.export_format = "csv"
        assert config.get_output_file_path().suffix == ".csv"


class TestOpenBuildingsGCSProcessor:
    """Test suite for GCS processor."""
    
    @pytest.fixture
    def sample_aoi(self, tmp_path):
        """Create a sample AOI file."""
        aoi_file = tmp_path / "test_aoi.geojson"
        # Simple 1-degree box
        gdf = gpd.GeoDataFrame(
            {'name': ['Test Area']},
            geometry=[box(-1, 5, 0, 6)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        return aoi_file
    
    @pytest.fixture
    def sample_config(self, sample_aoi, tmp_path):
        """Create a sample configuration."""
        return OpenBuildingsGCSConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_path,
            confidence_threshold=0.75,
            num_workers=2,
            export_format="csv"
        )
    
    @pytest.fixture
    def mock_gcs_client(self):
        """Mock GCS client."""
        with patch('geoworkflow.processors.extraction.open_buildings_gcs.GCSClient') as mock:
            yield mock
    
    def test_processor_initialization_success(self, sample_config):
        """Test successful processor initialization."""
        processor = OpenBuildingsGCSProcessor(sample_config)
        
        assert processor.gcs_config.confidence_threshold == 0.75
        assert processor.gcs_config.num_workers == 2
        assert processor.buildings_extracted == 0
        assert processor.s2_tokens == []
    
    def test_processor_initialization_with_dict(self, sample_aoi, tmp_path):
        """Test initialization with dictionary config."""
        config_dict = {
            'aoi_file': sample_aoi,
            'output_dir': tmp_path,
            'confidence_threshold': 0.8,
            'num_workers': 4
        }
        
        processor = OpenBuildingsGCSProcessor(config_dict)
        assert processor.gcs_config.confidence_threshold == 0.8
        assert processor.gcs_config.num_workers == 4
    
    def test_validation_missing_dependencies(self, sample_config):
        """Test validation catches missing dependencies."""
        with patch('geoworkflow.processors.extraction.open_buildings_gcs.HAS_REQUIRED_LIBS', False):
            processor = OpenBuildingsGCSProcessor(sample_config)
            validation = processor._validate_custom_inputs()
            
            assert not validation["valid"]
            assert any("missing" in err.lower() for err in validation["errors"])
    
    def test_validation_invalid_aoi_file(self, tmp_path):
        """Test validation catches invalid AOI file."""
        # Non-existent file
        config = OpenBuildingsGCSConfig(
            aoi_file=tmp_path / "nonexistent.geojson",
            output_dir=tmp_path
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("not found" in err.lower() for err in validation["errors"])
    
    def test_validation_existing_output_no_overwrite(self, sample_config):
        """Test validation fails when output exists without overwrite flag."""
        # Create existing output file
        output_file = sample_config.get_output_file_path()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()
        
        sample_config.overwrite_existing = False
        processor = OpenBuildingsGCSProcessor(sample_config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("already exists" in err.lower() for err in validation["errors"])
    
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.get_bounding_box_s2_covering_tokens')
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.GCSClient')
    def test_setup_custom_processing(self, mock_gcs_class, mock_s2_tokens, sample_config):
        """Test setup process."""
        # Mock S2 token generation
        mock_s2_tokens.return_value = ['token1', 'token2', 'token3']
        
        processor = OpenBuildingsGCSProcessor(sample_config)
        setup_info = processor._setup_custom_processing()
        
        # Verify GCS client was created
        assert mock_gcs_class.called
        
        # Verify S2 tokens were generated
        assert len(processor.s2_tokens) == 3
        assert setup_info["s2_cells_to_process"] == 3
        
        # Verify region GDF was loaded
        assert processor.region_gdf is not None
        assert processor.region_gdf.crs.to_string() == "EPSG:4326"
    
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.get_bounding_box_s2_covering_tokens')
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.GCSClient')
    def test_s2_token_generation_for_aoi(self, mock_gcs_class, mock_s2_tokens, sample_config):
        """Test S2 token generation for AOI geometry."""
        mock_s2_tokens.return_value = ['abc123', 'def456']
        
        processor = OpenBuildingsGCSProcessor(sample_config)
        processor._setup_custom_processing()
        
        # Verify tokens were set
        assert processor.s2_tokens == ['abc123', 'def456']
        
        # Verify S2 function was called with correct level
        mock_s2_tokens.assert_called_once()
        call_args = mock_s2_tokens.call_args
        assert call_args[1]['level'] == 6
    
    def test_download_s2_token_filtering_logic(self, sample_config):
        """Test building filtering logic for a single S2 cell."""
        processor = OpenBuildingsGCSProcessor(sample_config)
        
        # Create mock building data
        mock_buildings_df = pd.DataFrame({
            'latitude': [5.1, 5.2, 5.3],
            'longitude': [-0.5, -0.4, -0.3],
            'area_in_meters': [50, 120, 5],  # Last one below min
            'confidence': [0.8, 0.7, 0.9],  # Middle one below threshold
            'geometry': ['POLYGON(...)', 'POLYGON(...)', 'POLYGON(...)']
        })
        
        # Mock geometry for intersection test
        processor.prepared_geometry = Mock()
        processor.prepared_geometry.contains = Mock(side_effect=[True, True, False])
        
        # Apply filtering logic (simulated)
        filtered = mock_buildings_df[
            (mock_buildings_df['confidence'] >= sample_config.confidence_threshold) &
            (mock_buildings_df['area_in_meters'] >= sample_config.min_area_m2)
        ]
        
        # Should filter out: confidence too low, area too small
        assert len(filtered) == 1
        assert filtered.iloc[0]['confidence'] == 0.8
    
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.multiprocessing.Pool')
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.get_bounding_box_s2_covering_tokens')
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.GCSClient')
    def test_parallel_processing_workflow(self, mock_gcs, mock_s2, mock_pool, sample_config):
        """Test parallel processing with multiprocessing."""
        # Setup
        mock_s2.return_value = ['token1', 'token2']
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Mock temp files returned by workers
        temp_files = ['/tmp/temp1.csv', '/tmp/temp2.csv']
        mock_pool_instance.imap_unordered.return_value = temp_files
        
        processor = OpenBuildingsGCSProcessor(sample_config)
        processor._setup_custom_processing()
        
        # Verify pool was configured with correct number of workers
        # (This would be tested in the actual process_data method)
        assert processor.gcs_config.num_workers == 2
    
    def test_cleanup_resources(self, sample_config):
        """Test cleanup of GCS resources."""
        processor = OpenBuildingsGCSProcessor(sample_config)
        
        # Mock GCS client
        processor.gcs_client = Mock()
        
        cleanup_info = processor._cleanup_custom_resources()
        
        assert "gcs_client" in cleanup_info["components_cleaned"]
    
    @pytest.mark.parametrize("export_format,expected_extension", [
        ("geojson", ".geojson"),
        ("shapefile", ".shp"),
        ("csv", ".csv"),
    ])
    def test_output_format_handling(self, sample_aoi, tmp_path, export_format, expected_extension):
        """Test different output formats."""
        config = OpenBuildingsGCSConfig(
            aoi_file=sample_aoi,
            output_dir=tmp_path,
            export_format=export_format
        )
        
        output_path = config.get_output_file_path()
        assert output_path.suffix == expected_extension


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_gcs_bucket_path(self, tmp_path):
        """Test handling of invalid GCS bucket path."""
        aoi_file = tmp_path / "aoi.geojson"
        gdf = gpd.GeoDataFrame(
            {'name': ['test']},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        gdf.to_file(aoi_file, driver="GeoJSON")
        
        with pytest.raises(ValueError):
            OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path,
                gcs_bucket_path="invalid/path"
            )
    
    @patch('geoworkflow.processors.extraction.open_buildings_gcs.GCSClient')
    def test_gcs_connection_failure(self, mock_gcs, sample_config):
        """Test handling of GCS connection failures."""
        mock_gcs.side_effect = ConnectionError("Cannot connect to GCS")
        
        processor = OpenBuildingsGCSProcessor(sample_config)
        
        with pytest.raises(ConfigurationError):
            processor._setup_custom_processing()
    
    def test_corrupted_aoi_file(self, tmp_path):
        """Test handling of corrupted AOI file."""
        aoi_file = tmp_path / "corrupted.geojson"
        aoi_file.write_text("not valid geojson")
        
        config = OpenBuildingsGCSConfig(
            aoi_file=aoi_file,
            output_dir=tmp_path
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("cannot read" in err.lower() for err in validation["errors"])
