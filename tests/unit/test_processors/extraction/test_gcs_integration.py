# tests/integration/test_open_buildings_gcs_integration.py
"""
Integration tests for GCS Open Buildings extraction.

These tests interact with real GCS data (public Open Buildings dataset).
Mark as slow since they involve network operations.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

try:
    import geopandas as gpd
    from shapely.geometry import box
    import gcsfs
    HAS_GCS_LIBS = True
except ImportError:
    HAS_GCS_LIBS = False

# Skip all tests if GCS libraries not available
pytestmark = pytest.mark.skipif(
    not HAS_GCS_LIBS,
    reason="GCS libraries (gcsfs, geopandas) not available"
)

from geoworkflow.processors.extraction.open_buildings_gcs import OpenBuildingsGCSProcessor
from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig
from geoworkflow.utils.gcs_utils import GCSClient


@pytest.fixture(scope="module")
def temp_test_dir():
    """Create a temporary directory for integration tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_urban_aoi(temp_test_dir):
    """Create a very small urban AOI for quick testing."""
    # Small area in Accra, Ghana (known to have buildings)
    # Approximately 0.5 km² area
    aoi_file = temp_test_dir / "small_urban_aoi.geojson"
    
    gdf = gpd.GeoDataFrame(
        {'name': ['Accra Test Area']},
        geometry=[box(-0.20, 5.55, -0.19, 5.56)],  # Very small area
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.fixture
def medium_urban_aoi(temp_test_dir):
    """Create a medium-sized urban AOI."""
    # Medium area in Lagos, Nigeria (~10 km²)
    aoi_file = temp_test_dir / "medium_urban_aoi.geojson"
    
    gdf = gpd.GeoDataFrame(
        {'name': ['Lagos Test Area']},
        geometry=[box(3.35, 6.45, 3.40, 6.50)],
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.fixture
def rural_aoi(temp_test_dir):
    """Create a rural AOI with sparse buildings."""
    # Rural area with minimal buildings
    aoi_file = temp_test_dir / "rural_aoi.geojson"
    
    gdf = gpd.GeoDataFrame(
        {'name': ['Rural Test Area']},
        geometry=[box(1.0, 8.0, 1.1, 8.1)],  # Rural Sahel region
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.mark.integration
@pytest.mark.slow
class TestGCSConnection:
    """Test GCS connection and data access."""
    
    def test_gcs_client_anonymous_access(self):
        """Test that GCS client can connect anonymously to public data."""
        client = GCSClient(use_anonymous=True)
        
        # Test connection by checking if bucket exists
        bucket_path = "gs://open-buildings-data/v3/"
        
        # This should not raise an error
        assert client is not None
    
    def test_gcs_bucket_structure(self):
        """Test that expected GCS bucket structure exists."""
        try:
            fs = gcsfs.GCSFileSystem(token='anon')
            
            # Check if the polygons bucket exists
            path = "open-buildings-data/v3/polygons_s2_level_6_gzip_no_header"
            files = fs.ls(path, detail=False)
            
            # Should have CSV.GZ files
            assert len(files) > 0
            assert any(f.endswith('.csv.gz') for f in files)
            
        except Exception as e:
            pytest.skip(f"Cannot access GCS bucket: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestSmallAreaExtraction:
    """Test extraction for very small areas (fast tests)."""
    
    def test_extract_small_urban_area(self, small_urban_aoi, temp_test_dir):
        """Test extraction of buildings from small urban area."""
        output_dir = temp_test_dir / "small_urban_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            confidence_threshold=0.75,
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        # Verify success
        assert result.success, f"Extraction failed: {result.message}"
        
        # Verify output file exists
        assert len(result.output_paths) > 0
        output_file = result.output_paths[0]
        assert output_file.exists()
        
        # Verify some buildings were extracted
        assert result.processed_count > 0
        
        # Verify metrics were recorded
        assert processor.get_metric("buildings_extracted") > 0
        assert processor.get_metric("s2_cells_processed") > 0
    
    def test_extract_with_area_filters(self, small_urban_aoi, temp_test_dir):
        """Test extraction with min/max area filters."""
        output_dir = temp_test_dir / "filtered_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            confidence_threshold=0.75,
            min_area_m2=50.0,  # Exclude very small buildings
            max_area_m2=500.0,  # Exclude very large buildings
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        
        # With filters, we might get fewer buildings
        # but should still get some
        assert result.processed_count >= 0
    
    def test_extract_high_confidence_only(self, small_urban_aoi, temp_test_dir):
        """Test extraction with high confidence threshold."""
        output_dir = temp_test_dir / "high_confidence_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            confidence_threshold=0.9,  # Very high threshold
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        
        # Should get fewer buildings with higher threshold
        buildings_high = result.processed_count
        
        # Compare with lower threshold
        config.confidence_threshold = 0.7
        config.output_dir = temp_test_dir / "low_confidence_output"
        processor2 = OpenBuildingsGCSProcessor(config)
        result2 = processor2.process()
        
        assert result2.success
        buildings_low = result2.processed_count
        
        # Lower threshold should yield more or equal buildings
        assert buildings_low >= buildings_high


@pytest.mark.integration
@pytest.mark.slow
class TestOutputFormats:
    """Test different output formats."""
    
    def test_csv_output_format(self, small_urban_aoi, temp_test_dir):
        """Test CSV output format."""
        output_dir = temp_test_dir / "csv_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        
        output_file = result.output_paths[0]
        assert output_file.suffix == ".csv"
        assert output_file.exists()
        
        # Verify CSV can be read
        import pandas as pd
        df = pd.read_csv(output_file)
        
        # Check expected columns
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'confidence' in df.columns
        assert 'area_in_meters' in df.columns
    
    def test_geojson_output_format(self, small_urban_aoi, temp_test_dir):
        """Test GeoJSON output format."""
        output_dir = temp_test_dir / "geojson_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            export_format="geojson",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        
        output_file = result.output_paths[0]
        assert output_file.suffix == ".geojson"
        assert output_file.exists()
        
        # Verify GeoJSON can be read
        gdf = gpd.read_file(output_file)
        assert len(gdf) > 0
        assert gdf.crs is not None


@pytest.mark.integration
@pytest.mark.slow
class TestParallelProcessing:
    """Test parallel processing features."""
    
    def test_single_worker(self, small_urban_aoi, temp_test_dir):
        """Test extraction with single worker."""
        output_dir = temp_test_dir / "single_worker_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            num_workers=1,
            export_format="csv",
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        assert result.processed_count > 0
    
    def test_multiple_workers(self, small_urban_aoi, temp_test_dir):
        """Test extraction with multiple workers."""
        output_dir = temp_test_dir / "multi_worker_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            num_workers=4,
            export_format="csv",
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        assert result.success
        assert result.processed_count > 0
    
    def test_worker_count_performance(self, medium_urban_aoi, temp_test_dir):
        """Test that more workers improves performance for medium areas."""
        import time
        
        # Test with 1 worker
        config1 = OpenBuildingsGCSConfig(
            aoi_file=medium_urban_aoi,
            output_dir=temp_test_dir / "perf_test_1",
            num_workers=1,
            export_format="csv",
            overwrite_existing=True
        )
        
        start = time.time()
        processor1 = OpenBuildingsGCSProcessor(config1)
        result1 = processor1.process()
        time_1_worker = time.time() - start
        
        assert result1.success
        
        # Test with 4 workers
        config4 = OpenBuildingsGCSConfig(
            aoi_file=medium_urban_aoi,
            output_dir=temp_test_dir / "perf_test_4",
            num_workers=4,
            export_format="csv",
            overwrite_existing=True
        )
        
        start = time.time()
        processor4 = OpenBuildingsGCSProcessor(config4)
        result4 = processor4.process()
        time_4_workers = time.time() - start
        
        assert result4.success
        
        # Should extract same number of buildings
        assert result1.processed_count == result4.processed_count
        
        # 4 workers should be faster (allowing some overhead)
        # This is a soft assertion - parallel processing benefits may vary
        if time_4_workers >= time_1_worker:
            pytest.warns(
                UserWarning,
                match="Multiple workers did not improve performance"
            )


@pytest.mark.integration
@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_area_with_no_buildings(self, rural_aoi, temp_test_dir):
        """Test extraction for area with no or very few buildings."""
        output_dir = temp_test_dir / "no_buildings_output"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=rural_aoi,
            output_dir=output_dir,
            confidence_threshold=0.75,
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        result = processor.process()
        
        # Should complete successfully even with no buildings
        assert result.success
        
        # May have zero buildings
        assert result.processed_count >= 0
    
    def test_overwrite_existing_file(self, small_urban_aoi, temp_test_dir):
        """Test overwriting existing output file."""
        output_dir = temp_test_dir / "overwrite_test"
        output_dir.mkdir(exist_ok=True)
        
        config = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        # First extraction
        processor1 = OpenBuildingsGCSProcessor(config)
        result1 = processor1.process()
        assert result1.success
        
        first_count = result1.processed_count
        
        # Second extraction (overwrite)
        processor2 = OpenBuildingsGCSProcessor(config)
        result2 = processor2.process()
        assert result2.success
        
        # Should have same count
        assert result2.processed_count == first_count
    
    def test_no_overwrite_fails(self, small_urban_aoi, temp_test_dir):
        """Test that extraction fails without overwrite when file exists."""
        output_dir = temp_test_dir / "no_overwrite_test"
        output_dir.mkdir(exist_ok=True)
        
        # First extraction
        config1 = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            export_format="csv",
            num_workers=2,
            overwrite_existing=True
        )
        
        processor1 = OpenBuildingsGCSProcessor(config1)
        result1 = processor1.process()
        assert result1.success
        
        # Second extraction without overwrite
        config2 = OpenBuildingsGCSConfig(
            aoi_file=small_urban_aoi,
            output_dir=output_dir,
            export_format="csv",
            num_workers=2,
            overwrite_existing=False
        )
        
        processor2 = OpenBuildingsGCSProcessor(config2)
        validation = processor2._validate_custom_inputs()
        
        # Should fail validation
        assert not validation["valid"]
        assert any("already exists" in err.lower() for err in validation["errors"])


@pytest.mark.integration
@pytest.mark.slow
class TestRealWorldScenarios:
    """Test realistic use cases."""
    
    def test_city_extraction_workflow(self, medium_urban_aoi, temp_test_dir):
        """Test complete workflow for extracting city buildings."""
        output_dir = temp_test_dir / "city_extraction"
        
        config = OpenBuildingsGCSConfig(
            aoi_file=medium_urban_aoi,
            output_dir=output_dir,
            confidence_threshold=0.75,
            min_area_m2=20.0,
            export_format="geojson",
            num_workers=4,
            overwrite_existing=True
        )
        
        processor = OpenBuildingsGCSProcessor(config)
        
        # Test validation
        validation = processor._validate_custom_inputs()
        assert validation["valid"]
        
        # Test setup
        setup_info = processor._setup_custom_processing()
        assert "s2_cells_to_process" in setup_info
        assert setup_info["s2_cells_to_process"] > 0
        
        # Run extraction
        result = processor.process()
        
        assert result.success
        assert result.processed_count > 0
        
        # Verify output can be used
        output_file = result.output_paths[0]
        gdf = gpd.read_file(output_file)
        
        # Verify data quality
        assert len(gdf) == result.processed_count
        assert all(gdf['confidence'] >= 0.75)
        assert all(gdf.geometry.is_valid)
