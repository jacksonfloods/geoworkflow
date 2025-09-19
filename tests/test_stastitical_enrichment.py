# File: tests/test_statistical_enrichment.py
"""
Test suite for the Statistical Enrichment Processor.

This module contains comprehensive tests for the StatisticalEnrichmentProcessor
including unit tests, integration tests, and error condition testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

try:
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import Polygon
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

# Skip all tests if geospatial libraries not available
pytestmark = pytest.mark.skipif(
    not HAS_GEOSPATIAL_LIBS,
    reason="Geospatial libraries (geopandas, rasterio) not available"
)

from geoworkflow.processors.integration.enrichment import StatisticalEnrichmentProcessor
from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig
from geoworkflow.core.exceptions import ProcessingError, ValidationError


class TestStatisticalEnrichmentConfig:
    """Test the StatisticalEnrichmentConfig class."""
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = StatisticalEnrichmentConfig(
            coi_directory=Path("/tmp/coi"),
            coi_pattern="*AFRICAPOLIS*",
            raster_directory=Path("/tmp/rasters"),
            raster_pattern="*.tif",
            output_file=Path("/tmp/output.geojson"),
            statistics=["mean", "max", "min"]
        )
        
        assert config.coi_directory == Path("/tmp/coi")
        assert config.coi_pattern == "*AFRICAPOLIS*"
        assert config.statistics == ["mean", "max", "min"]
        assert config.area_units == "km2"  # default
        assert config.skip_existing == True  # default
    
    def test_invalid_statistics(self):
        """Test validation of invalid statistics."""
        with pytest.raises(ValueError, match="Invalid statistics"):
            StatisticalEnrichmentConfig(
                coi_directory=Path("/tmp/coi"),
                raster_directory=Path("/tmp/rasters"),
                output_file=Path("/tmp/output.geojson"),
                statistics=["invalid_stat", "mean"]
            )
    
    def test_invalid_area_units(self):
        """Test validation of invalid area units."""
        with pytest.raises(ValueError, match="area_units must be"):
            StatisticalEnrichmentConfig(
                coi_directory=Path("/tmp/coi"),
                raster_directory=Path("/tmp/rasters"),
                output_file=Path("/tmp/output.geojson"),
                statistics=["mean"],
                area_units="invalid"
            )
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = StatisticalEnrichmentConfig(
            coi_directory=Path("/tmp/coi"),
            raster_directory=Path("/tmp/rasters"),
            output_file=Path("/tmp/output.geojson"),
            statistics=["mean", "std"],
            area_units="m2"
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["statistics"] == ["mean", "std"]
        assert config_dict["area_units"] == "m2"
        
        # Test from_dict
        config_2 = StatisticalEnrichmentConfig.from_dict(config_dict)
        assert config_2.statistics == config.statistics
        assert config_2.area_units == config.area_units


class TestStatisticalEnrichmentProcessor:
    """Test the StatisticalEnrichmentProcessor class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample configuration for testing."""
        coi_dir = temp_dir / "coi"
        raster_dir = temp_dir / "rasters"
        coi_dir.mkdir()
        raster_dir.mkdir()
        
        return StatisticalEnrichmentConfig(
            coi_directory=coi_dir,
            coi_pattern="*cities*",
            raster_directory=raster_dir,
            raster_pattern="*.tif",
            output_file=temp_dir / "output.geojson",
            statistics=["mean", "max"],
            skip_existing=False
        )
    
    @pytest.fixture
    def sample_coi_file(self, temp_dir):
        """Create a sample COI file for testing."""
        coi_dir = temp_dir / "coi"
        coi_dir.mkdir(exist_ok=True)
        
        # Create a simple polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'city_name': ['Test City'], 'population': [100000]},
            geometry=[polygon],
            crs="EPSG:4326"
        )
        
        coi_file = coi_dir / "test_cities.geojson"
        gdf.to_file(coi_file, driver="GeoJSON")
        
        return coi_file
    
    @pytest.fixture
    def sample_raster_file(self, temp_dir):
        """Create a sample raster file for testing."""
        raster_dir = temp_dir / "rasters"
        raster_dir.mkdir(exist_ok=True)
        
        # Create a simple raster
        width, height = 10, 10
        transform = from_bounds(0, 0, 1, 1, width, height)
        
        # Create test data
        data = np.random.rand(height, width).astype(np.float32) * 100
        
        raster_file = raster_dir / "test_data.tif"
        
        with rasterio.open(
            raster_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        return raster_file
    
    def test_processor_initialization(self, sample_config):
        """Test processor initialization."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        assert processor.enrichment_config == sample_config
        assert processor.coi_file is None  # Not set until setup
        assert processor.raster_files == []  # Not set until setup
    
    def test_config_validation_missing_libraries(self, sample_config):
        """Test validation when geospatial libraries are missing."""
        with patch('geoworkflow.processors.integration.enrichment.HAS_GEOSPATIAL_LIBS', False):
            processor = StatisticalEnrichmentProcessor(sample_config)
            validation_result = processor._validate_custom_inputs()
            
            assert not validation_result["valid"]
            assert any("geospatial libraries" in error for error in validation_result["errors"])
    
    def test_config_validation_missing_directories(self, temp_dir):
        """Test validation when directories don't exist."""
        config = StatisticalEnrichmentConfig(
            coi_directory=temp_dir / "nonexistent_coi",
            raster_directory=temp_dir / "nonexistent_rasters",
            output_file=temp_dir / "output.geojson",
            statistics=["mean"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        validation_result = processor._validate_custom_inputs()
        
        assert not validation_result["valid"]
        assert len(validation_result["errors"]) >= 2  # Both directories missing
    
    def test_coi_file_discovery_success(self, sample_config, sample_coi_file):
        """Test successful COI file discovery."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        discovered_file = processor._discover_coi_file()
        assert discovered_file == sample_coi_file
    
    def test_coi_file_discovery_no_files(self, sample_config):
        """Test COI file discovery when no files match."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        with pytest.raises(ProcessingError, match="No COI files found"):
            processor._discover_coi_file()
    
    def test_coi_file_discovery_multiple_files(self, sample_config, temp_dir):
        """Test COI file discovery when multiple files match."""
        coi_dir = temp_dir / "coi"
        
        # Create multiple matching files
        for i in range(2):
            polygon = Polygon([(i, i), (i+1, i), (i+1, i+1), (i, i+1)])
            gdf = gpd.GeoDataFrame(
                {'name': [f'City {i}']},
                geometry=[polygon],
                crs="EPSG:4326"
            )
            gdf.to_file(coi_dir / f"test_cities_{i}.geojson", driver="GeoJSON")
        
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        with pytest.raises(ProcessingError, match="Multiple COI files found"):
            processor._discover_coi_file()
    
    def test_raster_file_discovery(self, sample_config, sample_raster_file):
        """Test raster file discovery."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        raster_files = processor._discover_raster_files()
        assert len(raster_files) == 1
        assert raster_files[0] == sample_raster_file
    
    def test_dataset_name_cleaning(self, sample_config):
        """Test dataset name cleaning functionality."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        test_cases = [
            ("copernicus_land_cover.tif", "copernicus_land_cover"),
            ("some-file_with.special.chars", "some_file_with_special_chars"),
            ("123_starts_with_number", "data_123_starts_with_number"),
            ("UPPERCASE_FILE", "uppercase_file"),
            ("file___with___multiple___underscores", "file_with_multiple_underscores")
        ]
        
        for input_name, expected_output in test_cases:
            result = processor._clean_dataset_name(input_name)
            assert result == expected_output
    
    @patch('geoworkflow.processors.integration.enrichment.zonal_stats')
    def test_raster_statistics_computation(self, mock_zonal_stats, sample_config, 
                                         sample_coi_file, sample_raster_file):
        """Test raster statistics computation."""
        # Mock zonal_stats return value
        mock_zonal_stats.return_value = [{'mean': 50.0, 'max': 100.0}]
        
        processor = StatisticalEnrichmentProcessor(sample_config)
        
        # Setup processor state
        processor.coi_gdf = gpd.read_file(sample_coi_file)
        processor.enriched_gdf = processor.coi_gdf.copy()
        
        # Test computation
        success = processor._compute_raster_statistics(sample_raster_file)
        
        assert success
        assert mock_zonal_stats.called
        
        # Check that columns were added
        expected_columns = ['test_data_mean', 'test_data_max']
        for col in expected_columns:
            assert col in processor.enriched_gdf.columns
    
    def test_area_calculation(self, sample_config, sample_coi_file):
        """Test polygon area calculation."""
        processor = StatisticalEnrichmentProcessor(sample_config)
        processor.enriched_gdf = gpd.read_file(sample_coi_file)
        
        # Test km2 calculation
        processor.enrichment_config.area_units = "km2"
        processor._add_area_column()
        assert "area_km2" in processor.enriched_gdf.columns
        
        # Test m2 calculation
        processor.enrichment_config.area_units = "m2"
        processor._add_area_column()
        assert "area_m2" in processor.enriched_gdf.columns
    
    def test_full_processing_workflow(self, sample_config, sample_coi_file, sample_raster_file):
        """Test the complete processing workflow."""
        with patch('geoworkflow.processors.integration.enrichment.zonal_stats') as mock_zonal_stats:
            # Mock zonal_stats return value
            mock_zonal_stats.return_value = [{'mean': 50.0, 'max': 100.0}]
            
            processor = StatisticalEnrichmentProcessor(sample_config)
            result = processor.process()
            
            assert result.success
            assert result.processed_count == 1
            assert result.failed_count == 0
            assert sample_config.output_file.exists()
            
            # Verify output file
            enriched_gdf = gpd.read_file(sample_config.output_file)
            assert len(enriched_gdf) == 1
            assert 'test_data_mean' in enriched_gdf.columns
            assert 'test_data_max' in enriched_gdf.columns
    
    def test_skip_existing_functionality(self, sample_config, sample_coi_file, sample_raster_file):
        """Test skip existing file functionality."""
        # Create output file first
        sample_config.output_file.touch()
        sample_config.skip_existing = True
        
        processor = StatisticalEnrichmentProcessor(sample_config)
        result = processor.process()
        
        assert result.success
        assert result.skipped_count == 1
        assert "already exists" in result.message


class TestConvenienceFunction:
    """Test the convenience function for enrichment."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('geoworkflow.processors.integration.enrichment.StatisticalEnrichmentProcessor')
    def test_convenience_function_success(self, mock_processor_class, temp_dir):
        """Test successful use of convenience function."""
        from geoworkflow.processors.integration.enrichment import enrich_cities_with_statistics
        
        # Mock processor and result
        mock_processor = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_processor.process.return_value = mock_result
        mock_processor_class.return_value = mock_processor
        
        success = enrich_cities_with_statistics(
            coi_directory=temp_dir / "coi",
            raster_directory=temp_dir / "rasters",
            output_file=temp_dir / "output.geojson"
        )
        
        assert success
        assert mock_processor_class.called
        assert mock_processor.process.called
    
    @patch('geoworkflow.processors.integration.enrichment.StatisticalEnrichmentProcessor')
    def test_convenience_function_failure(self, mock_processor_class, temp_dir):
        """Test convenience function with processing failure."""
        from geoworkflow.processors.integration.enrichment import enrich_cities_with_statistics
        
        # Mock processor and result
        mock_processor = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_processor.process.return_value = mock_result
        mock_processor_class.return_value = mock_processor
        
        success = enrich_cities_with_statistics(
            coi_directory=temp_dir / "coi",
            raster_directory=temp_dir / "rasters",
            output_file=temp_dir / "output.geojson"
        )
        
        assert not success
    
    def test_convenience_function_exception(self, temp_dir):
        """Test convenience function with exception."""
        from geoworkflow.processors.integration.enrichment import enrich_cities_with_statistics
        
        with patch('geoworkflow.processors.integration.enrichment.StatisticalEnrichmentProcessor') as mock_class:
            mock_class.side_effect = Exception("Test exception")
            
            success = enrich_cities_with_statistics(
                coi_directory=temp_dir / "coi",
                raster_directory=temp_dir / "rasters",
                output_file=temp_dir / "output.geojson"
            )
            
            assert not success


class TestIntegrationScenarios:
    """Test integration scenarios and real-world use cases."""
    
    @pytest.fixture
    def comprehensive_test_setup(self):
        """Create a comprehensive test setup with multiple files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create directory structure
            coi_dir = temp_dir / "coi"
            raster_dir = temp_dir / "rasters"
            coi_dir.mkdir()
            raster_dir.mkdir()
            
            # Create COI file with multiple cities
            polygons = [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
            ]
            
            gdf = gpd.GeoDataFrame(
                {
                    'city_name': ['City A', 'City B', 'City C'],
                    'population': [100000, 50000, 200000],
                    'country': ['Angola', 'Namibia', 'Botswana']
                },
                geometry=polygons,
                crs="EPSG:4326"
            )
            
            coi_file = coi_dir / "AFRICAPOLIS_cities.geojson"
            gdf.to_file(coi_file, driver="GeoJSON")
            
            # Create multiple raster files
            raster_configs = [
                ("copernicus_landcover.tif", (0, 0, 6, 6)),
                ("odiac_emissions.tif", (0, 0, 6, 6)),
                ("pm25_concentration.tif", (0, 0, 6, 6))
            ]
            
            raster_files = []
            for filename, bounds in raster_configs:
                width, height = 20, 20
                transform = from_bounds(*bounds, width, height)
                
                # Create realistic test data
                if "landcover" in filename:
                    data = np.random.choice([10, 20, 30, 60], size=(height, width))
                elif "emissions" in filename:
                    data = np.random.exponential(scale=5, size=(height, width))
                else:  # PM2.5
                    data = np.random.lognormal(mean=2, sigma=0.5, size=(height, width))
                
                data = data.astype(np.float32)
                
                raster_file = raster_dir / filename
                
                with rasterio.open(
                    raster_file,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    dst.write(data, 1)
                
                raster_files.append(raster_file)
            
            yield {
                'temp_dir': temp_dir,
                'coi_file': coi_file,
                'raster_files': raster_files,
                'coi_dir': coi_dir,
                'raster_dir': raster_dir
            }
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_comprehensive_enrichment_workflow(self, comprehensive_test_setup):
        """Test comprehensive enrichment with multiple cities and rasters."""
        setup = comprehensive_test_setup
        
        config = StatisticalEnrichmentConfig(
            coi_directory=setup['coi_dir'],
            coi_pattern="*AFRICAPOLIS*",
            raster_directory=setup['raster_dir'],
            raster_pattern="*.tif",
            output_file=setup['temp_dir'] / "enriched_cities.geojson",
            statistics=["mean", "max", "min", "std"],
            add_area_column=True,
            area_units="km2"
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        assert result.success
        assert result.processed_count == 3  # Three raster files
        assert result.failed_count == 0
        
        # Verify output
        enriched_gdf = gpd.read_file(config.output_file)
        
        # Check original data preserved
        assert len(enriched_gdf) == 3
        assert 'city_name' in enriched_gdf.columns
        assert 'population' in enriched_gdf.columns
        
        # Check statistical columns added
        expected_columns = []
        for dataset in ['copernicus_landcover', 'odiac_emissions', 'pm25_concentration']:
            for stat in ['mean', 'max', 'min', 'std']:
                expected_columns.append(f"{dataset}_{stat}")
        
        for col in expected_columns:
            assert col in enriched_gdf.columns
        
        # Check area column
        assert 'area_km2' in enriched_gdf.columns
        
        # Verify statistics are reasonable (not all NaN)
        for col in expected_columns:
            assert not enriched_gdf[col].isna().all()
    
    def test_crs_reprojection_handling(self, comprehensive_test_setup):
        """Test handling of CRS mismatches between COI and rasters."""
        setup = comprehensive_test_setup
        
        # Create raster in different CRS
        raster_file_utm = setup['raster_dir'] / "utm_raster.tif"
        
        # Create raster in UTM (approximate for Southern Africa)
        width, height = 10, 10
        # UTM bounds (rough conversion from geographic bounds)
        transform = from_bounds(200000, 7800000, 600000, 8200000, width, height)
        data = np.random.rand(height, width).astype(np.float32) * 100
        
        with rasterio.open(
            raster_file_utm,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:32733',  # UTM Zone 33S
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        config = StatisticalEnrichmentConfig(
            coi_directory=setup['coi_dir'],
            coi_pattern="*AFRICAPOLIS*",
            raster_directory=setup['raster_dir'],
            raster_pattern="utm_*.tif",
            output_file=setup['temp_dir'] / "enriched_utm.geojson",
            statistics=["mean"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        # Should succeed despite CRS difference
        assert result.success
        assert result.processed_count == 1
    
    def test_memory_efficiency_large_dataset(self, comprehensive_test_setup):
        """Test memory efficiency with larger datasets."""
        setup = comprehensive_test_setup
        
        # Create a larger raster file
        large_raster = setup['raster_dir'] / "large_raster.tif"
        width, height = 1000, 1000  # 1M pixels
        transform = from_bounds(0, 0, 10, 10, width, height)
        
        # Create data in chunks to avoid memory issues
        with rasterio.open(
            large_raster,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            # Write data in blocks
            for i in range(0, height, 256):
                for j in range(0, width, 256):
                    block_height = min(256, height - i)
                    block_width = min(256, width - j)
                    block_data = np.random.rand(block_height, block_width).astype(np.float32)
                    
                    dst.write(block_data, 1, window=rasterio.windows.Window(j, i, block_width, block_height))
        
        config = StatisticalEnrichmentConfig(
            coi_directory=setup['coi_dir'],
            coi_pattern="*AFRICAPOLIS*",
            raster_directory=setup['raster_dir'],
            raster_pattern="large_*.tif",
            output_file=setup['temp_dir'] / "enriched_large.geojson",
            statistics=["mean", "max"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        assert result.success
        assert result.processed_count == 1
        
        # Verify results are reasonable
        enriched_gdf = gpd.read_file(config.output_file)
        assert 'large_raster_mean' in enriched_gdf.columns
        assert 'large_raster_max' in enriched_gdf.columns


class TestErrorConditions:
    """Test various error conditions and edge cases."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_corrupted_raster_handling(self, temp_dir):
        """Test handling of corrupted raster files."""
        # Create directories
        coi_dir = temp_dir / "coi"
        raster_dir = temp_dir / "rasters"
        coi_dir.mkdir()
        raster_dir.mkdir()
        
        # Create valid COI file
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[polygon], crs="EPSG:4326")
        gdf.to_file(coi_dir / "test_cities.geojson", driver="GeoJSON")
        
        # Create corrupted raster file (just empty file with .tif extension)
        corrupted_raster = raster_dir / "corrupted.tif"
        corrupted_raster.write_text("not a raster file")
        
        config = StatisticalEnrichmentConfig(
            coi_directory=coi_dir,
            coi_pattern="*cities*",
            raster_directory=raster_dir,
            raster_pattern="*.tif",
            output_file=temp_dir / "output.geojson",
            statistics=["mean"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        # Should complete but with failed files
        assert result.success  # Overall success despite failed files
        assert result.failed_count == 1
        assert result.processed_count == 0
    
    def test_empty_coi_file(self, temp_dir):
        """Test handling of empty COI file."""
        # Create directories
        coi_dir = temp_dir / "coi"
        raster_dir = temp_dir / "rasters"
        coi_dir.mkdir()
        raster_dir.mkdir()
        
        # Create empty COI file
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
        empty_gdf.to_file(coi_dir / "empty_cities.geojson", driver="GeoJSON")
        
        # Create valid raster
        width, height = 10, 10
        transform = from_bounds(0, 0, 1, 1, width, height)
        data = np.random.rand(height, width).astype(np.float32)
        
        with rasterio.open(
            raster_dir / "test.tif",
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        config = StatisticalEnrichmentConfig(
            coi_directory=coi_dir,
            coi_pattern="*cities*",
            raster_directory=raster_dir,
            raster_pattern="*.tif",
            output_file=temp_dir / "output.geojson",
            statistics=["mean"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        # Should succeed with empty output
        assert result.success
        
        # Check output file
        output_gdf = gpd.read_file(config.output_file)
        assert len(output_gdf) == 0
    
    def test_no_spatial_overlap(self, temp_dir):
        """Test case where COI and rasters don't spatially overlap."""
        # Create directories
        coi_dir = temp_dir / "coi"
        raster_dir = temp_dir / "rasters"
        coi_dir.mkdir()
        raster_dir.mkdir()
        
        # Create COI in one location
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[polygon], crs="EPSG:4326")
        gdf.to_file(coi_dir / "test_cities.geojson", driver="GeoJSON")
        
        # Create raster in completely different location
        width, height = 10, 10
        transform = from_bounds(100, 100, 101, 101, width, height)  # Far from COI
        data = np.random.rand(height, width).astype(np.float32)
        
        with rasterio.open(
            raster_dir / "distant.tif",
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        config = StatisticalEnrichmentConfig(
            coi_directory=coi_dir,
            coi_pattern="*cities*",
            raster_directory=raster_dir,
            raster_pattern="*.tif",
            output_file=temp_dir / "output.geojson",
            statistics=["mean"]
        )
        
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        # Should succeed but with NaN values for statistics
        assert result.success
        
        output_gdf = gpd.read_file(config.output_file)
        assert len(output_gdf) == 1
        assert pd.isna(output_gdf['distant_mean'].iloc[0])


# Performance and benchmark tests
class TestPerformance:
    """Test performance characteristics of the enrichment processor."""
    
    @pytest.mark.slow
    def test_processing_time_scaling(self):
        """Test that processing time scales reasonably with input size."""
        # This test would create datasets of different sizes and measure processing time
        # Skip implementation for now but structure is provided
        pass
    
    @pytest.mark.slow 
    def test_memory_usage_monitoring(self):
        """Test memory usage during processing."""
        # This test would monitor memory usage during processing
        # Skip implementation for now but structure is provided
        pass


# Integration tests with other processors
class TestProcessorIntegration:
    """Test integration with other geoworkflow processors."""
    
    def test_integration_with_clipping_processor(self):
        """Test integration with the ClippingProcessor output."""
        # This test would use actual ClippingProcessor output as input
        # Skip implementation for now but structure is provided
        pass
    
    def test_integration_with_alignment_processor(self):
        """Test integration with the AlignmentProcessor output.""" 
        # This test would use actual AlignmentProcessor output as input
        # Skip implementation for now but structure is provided
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])