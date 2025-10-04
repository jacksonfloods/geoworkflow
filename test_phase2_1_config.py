#!/usr/bin/env python3
"""
Test script for OpenBuildingsGCSConfig.

This script validates the configuration class works correctly.
"""
from pathlib import Path
import sys

try:
    # Try to import from the new location
    sys.path.insert(0, 'src')
    from geoworkflow.schemas.open_buildings_gcs_config import (
        OpenBuildingsGCSConfig,
        OpenBuildingsGCSPointsConfig
    )
    print("✓ Successfully imported OpenBuildingsGCSConfig")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_basic_config():
    """Test basic configuration creation."""
    print("\n=== Testing Basic Configuration ===")
    
    # Create a dummy AOI file for testing
    test_aoi = Path("test_aoi.geojson")
    if not test_aoi.exists():
        print(f"⚠ Warning: {test_aoi} doesn't exist, creating placeholder...")
        test_aoi.write_text('{"type":"FeatureCollection","features":[]}')
    
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            confidence_threshold=0.75,
            num_workers=2
        )
        print("✓ Basic configuration created successfully")
        print(f"  - AOI file: {config.aoi_file}")
        print(f"  - Output dir: {config.output_dir}")
        print(f"  - Confidence: {config.confidence_threshold}")
        print(f"  - Workers: {config.num_workers}")
        return True
    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        return False

def test_validation():
    """Test configuration validation."""
    print("\n=== Testing Validation ===")
    
    # Test invalid confidence threshold
    test_aoi = Path("test_aoi.geojson")
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            confidence_threshold=0.3  # Too low!
        )
        print("✗ Validation failed - accepted invalid confidence")
        return False
    except ValueError:
        print("✓ Correctly rejected invalid confidence threshold")
    
    # Test invalid area range
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            min_area_m2=100.0,
            max_area_m2=50.0  # Max < Min!
        )
        print("✗ Validation failed - accepted invalid area range")
        return False
    except ValueError:
        print("✓ Correctly rejected invalid area range")
    
    return True

def test_helper_methods():
    """Test configuration helper methods."""
    print("\n=== Testing Helper Methods ===")
    
    test_aoi = Path("test_aoi.geojson")
    config = OpenBuildingsGCSConfig(
        aoi_file=test_aoi,
        output_dir=Path("./test_output/"),
        export_format="geojson"
    )
    
    # Test output path generation
    output_path = config.get_output_file_path()
    print(f"✓ Output path: {output_path}")
    
    # Test summary
    summary = config.summary()
    print("✓ Configuration summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    # Test memory estimation
    memory = config.estimate_memory_usage()
    print(f"✓ Estimated memory usage: {memory:.2f} MB")
    
    return True

def test_points_config():
    """Test points-specific configuration."""
    print("\n=== Testing Points Configuration ===")
    
    test_aoi = Path("test_aoi.geojson")
    try:
        config = OpenBuildingsGCSPointsConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/")
        )
        print("✓ Points configuration created successfully")
        print(f"  - Data type: {config.data_type}")
        print(f"  - GCS path: {config.gcs_bucket_path}")
        print(f"  - Export format: {config.export_format}")
        return True
    except Exception as e:
        print(f"✗ Failed to create points config: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OpenBuildingsGCSConfig Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_config,
        test_validation,
        test_helper_methods,
        test_points_config,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    # Cleanup test file
    test_aoi = Path("test_aoi.geojson")
    if test_aoi.exists():
        test_aoi.unlink()
        print("✓ Cleaned up test files")
    
    sys.exit(0 if all(results) else 1)
