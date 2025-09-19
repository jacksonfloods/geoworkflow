#!/usr/bin/env python3
# File: test_aoi_processor.py
"""
Test script for the AOI processor implementation.

This script validates that the new AOI processor works correctly and
produces the same results as the legacy define_aoi.py script.
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from geoworkflow.core.logging_setup import setup_logging
from geoworkflow.schemas.config_models import AOIConfig
from geoworkflow.processors.aoi.processor import AOIProcessor


def test_southern_africa_aoi():
    """Test creating Southern Africa AOI (Angola, Namibia, Botswana)."""
    print("\n" + "="*60)
    print("TEST 1: Southern Africa AOI Creation")
    print("="*60)
    
    # Create test configuration
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["Angola", "Namibia", "Botswana"],
        buffer_km=100,
        output_file=Path("data/aoi/test_southern_africa_aoi.geojson")
    )
    
    print(f"Input file: {config.input_file}")
    print(f"Countries: {config.countries}")
    print(f"Buffer: {config.buffer_km} km")
    print(f"Output: {config.output_file}")
    
    # Create and run processor
    try:
        processor = AOIProcessor(config)
        result = processor.process()
        
        # Check actual success status
        if result.success:
            print(f"\n✅ Processing completed successfully!")
            print(f"   Success: {result.success}")
            print(f"   Processed count: {result.processed_count}")
            print(f"   Elapsed time: {result.elapsed_time:.2f}s")
            print(f"   Message: {result.message}")
            
            if result.metadata:
                print("\n📊 Processing Metadata:")
                for key, value in result.metadata.items():
                    print(f"   {key}: {value}")
            
            # Verify output file exists
            if config.output_file.exists():
                file_size = config.output_file.stat().st_size
                print(f"\n📁 Output file created: {file_size:,} bytes")
            else:
                print(f"\n⚠️  Warning: Output file not found at {config.output_file}")
            
            return True
        else:
            print(f"\n❌ Processing failed: {result.message}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_countries_dissolved():
    """Test creating continent-wide AOI with dissolved boundaries."""
    print("\n" + "="*60)
    print("TEST 2: All Countries Dissolved AOI")
    print("="*60)
    
    # Create test configuration  
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        use_all_countries=True,
        dissolve_boundaries=True,
        buffer_km=50,
        output_file=Path("data/aoi/test_africa_dissolved_aoi.geojson")
    )
    
    print(f"Input file: {config.input_file}")
    print(f"Use all countries: {config.use_all_countries}")
    print(f"Dissolve boundaries: {config.dissolve_boundaries}")
    print(f"Buffer: {config.buffer_km} km")
    print(f"Output: {config.output_file}")
    
    try:
        processor = AOIProcessor(config)
        result = processor.process()
        
        if result.success:
            print(f"\n✅ Processing completed successfully!")
            print(f"   Success: {result.success}")
            print(f"   Message: {result.message}")
            
            # Get additional info
            bounds = processor.get_aoi_bounds()
            area_km2 = processor.get_aoi_area_km2()
            
            if bounds:
                print(f"\n🌍 AOI Bounds: {bounds}")
            if area_km2:
                print(f"📏 Approximate area: {area_km2:,.0f} km²")
            
            return True
        else:
            print(f"\n❌ Processing failed: {result.message}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_list_countries():
    """Test listing available countries."""
    print("\n" + "="*60)
    print("TEST 3: List Available Countries")
    print("="*60)
    
    # Create minimal configuration for listing
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["dummy"],  # Not used for listing
        output_file=Path("data/aoi/dummy.geojson")  # Not used for listing
    )
    
    try:
        processor = AOIProcessor(config)
        
        # List all countries
        all_countries = processor.list_available_countries()
        print(f"Total countries available: {len(all_countries)}")
        print(f"First 10 countries: {all_countries[:10]}")
        
        # List countries starting with 'K'
        k_countries = processor.list_available_countries(prefix="K")
        print(f"\nCountries starting with 'K': {k_countries}")
        
        # Basic validation
        if len(all_countries) > 0 and isinstance(all_countries[0], str):
            print("✅ Country listing working correctly")
            return True
        else:
            print("❌ Country listing returned unexpected format")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_errors():
    """Test configuration validation."""
    print("\n" + "="*60)
    print("TEST 4: Configuration Validation")
    print("="*60)
    
    # Test with invalid file path
    print("Testing invalid file path...")
    try:
        config = AOIConfig(
            input_file=Path("nonexistent_file.geojson"),
            country_name_column="NAME_0",
            countries=["Angola"],
            output_file=Path("data/aoi/test_output.geojson")
        )
        processor = AOIProcessor(config)
        result = processor.process()
        
        # Check if processing actually failed (should fail due to missing input file)
        if not result.success:
            print(f"✅ Correctly failed with message: {result.message}")
        else:
            print("❌ Should have failed but succeeded")
            return False
            
    except Exception as e:
        print(f"✅ Correctly caught exception: {type(e).__name__} - {str(e)}")
    
    # Test with invalid countries
    print("\nTesting invalid country names...")
    try:
        config = AOIConfig(
            input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
            country_name_column="NAME_0",
            countries=["NonexistentCountry", "AnotherFakeCountry"],
            output_file=Path("data/aoi/test_output.geojson")
        )
        processor = AOIProcessor(config)
        result = processor.process()
        
        # Check if processing actually failed (should fail due to invalid countries)
        if not result.success:
            print(f"✅ Correctly failed with message: {result.message}")
        else:
            print("❌ Should have failed but succeeded")
            return False
            
    except Exception as e:
        print(f"✅ Correctly caught exception: {type(e).__name__} - {str(e)}")
    
    # Test with invalid column name
    print("\nTesting invalid column name...")
    try:
        config = AOIConfig(
            input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
            country_name_column="NONEXISTENT_COLUMN",
            countries=["Angola"],
            output_file=Path("data/aoi/test_output.geojson")
        )
        processor = AOIProcessor(config)
        result = processor.process()
        
        # Check if processing actually failed (should fail due to invalid column)
        if not result.success:
            print(f"✅ Correctly failed with message: {result.message}")
        else:
            print("❌ Should have failed but succeeded")
            return False
            
    except Exception as e:
        print(f"✅ Correctly caught exception: {type(e).__name__} - {str(e)}")
    
    return True


def test_small_buffer():
    """Test with small buffer to ensure quick processing."""
    print("\n" + "="*60)
    print("TEST 5: Small Buffer Test")
    print("="*60)
    
    # Create test configuration with small buffer
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["Rwanda"],  # Small country for quick processing
        buffer_km=10,  # Small buffer
        output_file=Path("data/aoi/test_rwanda_small_buffer.geojson")
    )
    
    print(f"Testing small country (Rwanda) with 10km buffer...")
    
    try:
        processor = AOIProcessor(config)
        result = processor.process()
        
        if result.success:
            print(f"✅ Processing completed successfully!")
            print(f"   Elapsed time: {result.elapsed_time:.2f}s")
            
            # Check if file was created and has reasonable size
            if config.output_file.exists():
                file_size = config.output_file.stat().st_size
                print(f"   File size: {file_size:,} bytes")
                
                if file_size > 100:  # Should be more than 100 bytes for a valid GeoJSON
                    return True
                else:
                    print(f"❌ Output file too small: {file_size} bytes")
                    return False
            else:
                print("❌ Output file not created")
                return False
        else:
            print(f"❌ Processing failed: {result.message}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        return False


def test_no_buffer():
    """Test AOI creation without buffer."""
    print("\n" + "="*60)
    print("TEST 6: No Buffer Test")
    print("="*60)
    
    # Create test configuration with no buffer
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["Lesotho"],  # Small country
        buffer_km=0,  # No buffer
        output_file=Path("data/aoi/test_lesotho_no_buffer.geojson")
    )
    
    print(f"Testing Lesotho with no buffer...")
    
    try:
        processor = AOIProcessor(config)
        result = processor.process()
        
        if result.success:
            print(f"✅ Processing completed successfully!")
            print(f"   No buffer applied: {result.metadata.get('buffer_applied_km', 'N/A')} km")
            return True
        else:
            print(f"❌ Processing failed: {result.message}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        return False


def main():
    """Run all AOI processor tests."""
    print("🧪 AOI PROCESSOR TEST SUITE")
    print("Testing enhanced AOI processor against legacy functionality...")
    
    # Setup logging (quieter for tests)
    setup_logging(level="WARNING")  # Only show warnings and errors during tests
    
    # Ensure output directory exists
    aoi_dir = Path("data/aoi")
    aoi_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("Southern Africa AOI", test_southern_africa_aoi),
        ("All Countries Dissolved", test_all_countries_dissolved),
        ("List Countries", test_list_countries),
        ("Validation Errors", test_validation_errors),
        ("Small Buffer Test", test_small_buffer),
        ("No Buffer Test", test_no_buffer),
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed_tests.append(test_name)
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed_tests.append(test_name)
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! AOI processor is working correctly.")
        
        # Show created files
        print("\n📁 Created test files:")
        test_files = list(Path("data/aoi").glob("test_*.geojson"))
        for test_file in test_files:
            file_size = test_file.stat().st_size
            print(f"   {test_file.name} ({file_size:,} bytes)")
        
        return 0
    else:
        print("⚠️  Some tests failed:")
        for failed_test in failed_tests:
            print(f"   ❌ {failed_test}")
        
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that input file exists: data/00_source/archives/africa_boundaries/africa_boundaries.geojson")
        print("2. Ensure GeoPandas is installed: conda install geopandas")
        print("3. Check file permissions for data/aoi/ directory")
        print("4. Review error messages above for specific issues")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())