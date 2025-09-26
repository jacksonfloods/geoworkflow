# File: tests/processors/extraction/test_open_buildings.py
"""
Unit tests for Open Buildings extraction processor.
Uses mocked Earth Engine API for CI/CD compatibility.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from geoworkflow.processors.extraction.open_buildings import OpenBuildingsExtractionProcessor
from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
from geoworkflow.core.exceptions import ExtractionError, ConfigurationError, ValidationError


class TestOpenBuildingsExtractionProcessor:
    """Test suite for Open Buildings processor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def sample_aoi_file(self, temp_dir):
        """Create a sample AOI GeoJSON file."""
        aoi_file = temp_dir / "test_aoi.geojson"
        sample_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-2.0, 5.0], [-1.0, 5.0], [-1.0, 6.0], [-2.0, 6.0], [-2.0, 5.0]
                    ]]
                },
                "properties": {"name": "test_area"}
            }]
        }
        
        with open(aoi_file, 'w') as f:
            json.dump(sample_geojson, f)
        
        return aoi_file
    
    @pytest.fixture
    def sample_config(self, temp_dir, sample_aoi_file):
        """Create sample configuration for testing."""
        return OpenBuildingsExtractionConfig(
            aoi_file=sample_aoi_file,
            output_dir=temp_dir / "output",
            confidence_threshold=0.75,
            export_format="geojson",
            overwrite_existing=True
        )
    
    def test_initialization_success(self, sample_config):
        """Test successful processor initialization."""
        processor = OpenBuildingsExtractionProcessor(sample_config)
        assert processor.buildings_config.confidence_threshold == 0.75
        assert processor.buildings_config.export_format == "geojson"
        assert processor.ee_api is None  # Not initialized yet
    
    def test_initialization_with_dict(self, temp_dir, sample_aoi_file):
        """Test processor initialization with dictionary config."""
        config_dict = {
            'aoi_file': str(sample_aoi_file),
            'output_dir': str(temp_dir / "output"),
            'confidence_threshold': 0.8,
            'export_format': 'shapefile'
        }
        
        processor = OpenBuildingsExtractionProcessor(config_dict)
        assert processor.buildings_config.confidence_threshold == 0.8
        assert processor.buildings_config.export_format == "shapefile"
    
    @patch('geoworkflow.processors.extraction.open_buildings.HAS_REQUIRED_LIBS', False)
    def test_missing_dependencies(self, sample_config):
        """Test handling of missing Earth Engine dependency."""
        processor = OpenBuildingsExtractionProcessor(sample_config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("Required libraries missing" in error for error in validation["errors"])
    
    def test_configuration_validation_missing_aoi(self, temp_dir):
        """Test configuration validation with missing AOI file."""
        with pytest.raises(ValidationError):
            OpenBuildingsExtractionConfig(
                aoi_file=Path("nonexistent.geojson"),
                output_dir=temp_dir / "output",
                confidence_threshold=0.75
            )
    
    def test_configuration_validation_invalid_confidence(self, temp_dir, sample_aoi_file):
        """Test configuration validation with invalid confidence threshold."""
        with pytest.raises(ValidationError):
            OpenBuildingsExtractionConfig(
                aoi_file=sample_aoi_file,
                output_dir=temp_dir / "output",
                confidence_threshold=1.5  # Invalid: > 1.0
            )
    
    def test_output_file_path_generation(self, sample_config):
        """Test output file path generation for different formats."""
        # Test GeoJSON
        sample_config.export_format = "geojson"
        expected_geojson = sample_config.output_dir / "open_buildings.geojson"
        assert sample_config.get_output_file_path() == expected_geojson
        
        # Test Shapefile
        sample_config.export_format = "shapefile"
        expected_shp = sample_config.output_dir / "open_buildings.shp"
        assert sample_config.get_output_file_path() == expected_shp
        
        # Test CSV
        sample_config.export_format = "csv"
        expected_csv = sample_config.output_dir / "open_buildings.csv"
        assert sample_config.get_output_file_path() == expected_csv
    
    @patch('geoworkflow.processors.extraction.open_buildings.check_earth_engine_available')
    def test_validation_earth_engine_unavailable(self, mock_ee_check, sample_config):
        """Test validation when Earth Engine is unavailable."""
        mock_ee_check.return_value = False
        
        processor = OpenBuildingsExtractionProcessor(sample_config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("Earth Engine API not available" in error for error in validation["errors"])
    
    def test_validation_existing_output_file(self, sample_config):
        """Test validation when output file exists and overwrite is False."""
        # Create the output file
        sample_config.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = sample_config.get_output_file_path()
        output_file.touch()
        
        # Set overwrite to False
        sample_config.overwrite_existing = False
        
        processor = OpenBuildingsExtractionProcessor(sample_config)
        validation = processor._validate_custom_inputs()
        
        assert not validation["valid"]
        assert any("already exists" in error for error in validation["errors"])
    
    @patch('geoworkflow.utils.earth_engine_utils.ee')
    @patch('geoworkflow.processors.extraction.open_buildings.check_earth_engine_available')
    def test_setup_processing_success(self, mock_ee_check, mock_ee, sample_config):
        """Test successful setup processing."""
        mock_ee_check.return_value = True
        
        # Mock Earth Engine authentication
        mock_auth_result = "test-project-123"
        
        with patch('geoworkflow.utils.earth_engine_utils.EarthEngineAuth.authenticate') as mock_auth, \
             patch('geoworkflow.utils.earth_engine_utils.OpenBuildingsAPI') as mock_api:
            
            mock_auth.return_value = mock_auth_result
            mock_api_instance = MagicMock()
            mock_api.return_value = mock_api_instance
            
            processor = OpenBuildingsExtractionProcessor(sample_config)
            setup_result = processor._setup_custom_processing()
            
            assert "earth_engine_auth" in setup_result["components"]
            assert setup_result["project_id"] == mock_auth_result
            assert processor.ee_api is not None
    
    def test_academic_setup_guidance(self, sample_config):
        """Test academic setup guidance generation."""
        guidance = sample_config.get_academic_setup_guidance()
        
        assert "Earth Engine Setup for Academic Users" in guidance
        assert "earthengine.google.com/signup" in guidance
        assert "Service Account" in guidance
        assert "User Credentials" in guidance
    
    @patch('geoworkflow.processors.extraction.open_buildings.HAS_REQUIRED_LIBS', True)
    @patch('geoworkflow.processors.extraction.open_buildings.check_earth_engine_available')
    def test_full_validation_success(self, mock_ee_check, sample_config):
        """Test full validation with all requirements met."""
        mock_ee_check.return_value = True
        
        # Mock credential validation
        with patch('geoworkflow.utils.earth_engine_utils.validate_earth_engine_setup') as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": [], "warnings": []}
            
            processor = OpenBuildingsExtractionProcessor(sample_config)
            validation = processor._validate_custom_inputs()
            
            assert validation["valid"]
            assert len(validation["errors"]) == 0


class TestOpenBuildingsConfigurationValidation:
    """Test configuration validation separately."""
    
    def test_confidence_threshold_warning(self, tmp_path):
        """Test warning for low confidence thresholds."""
        aoi_file = tmp_path / "test.geojson"
        aoi_file.write_text('{"type": "FeatureCollection", "features": []}')
        
        with pytest.warns(UserWarning, match="below 0.7"):
            OpenBuildingsExtractionConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path / "output",
                confidence_threshold=0.6  # Should trigger warning
            )
    
    def test_auth_method_correction(self, tmp_path):
        """Test automatic auth method correction."""
        aoi_file = tmp_path / "test.geojson"
        aoi_file.write_text('{"type": "FeatureCollection", "features": []}')
        
        service_key = tmp_path / "key.json"
        service_key.write_text('{"project_id": "test"}')
        
        with pytest.warns(UserWarning, match="Setting auth_method"):
            config = OpenBuildingsExtractionConfig(
                aoi_file=aoi_file,
                output_dir=tmp_path / "output",
                service_account_key=service_key,
                auth_method="user_credentials"  # Wrong method
            )
            
        assert config.auth_method == "service_account"
