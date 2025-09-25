# test_phase_1_2_working.py
"""
Working test for Phase 1.2 completion that handles file validation correctly.
"""

from pathlib import Path
import tempfile
import json

def test_phase_1_2_imports():
    """Test that all Phase 1.2 imports work correctly."""
    print("Testing Phase 1.2 imports...")
    
    try:
        # Test core imports
        from geoworkflow.utils.earth_engine_utils import (
            check_earth_engine_available, 
            validate_earth_engine_setup,
            EarthEngineAuth,
            OpenBuildingsAPI
        )
        print("✅ Earth Engine utilities imported successfully")
        
        # Test configuration import
        from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
        print("✅ Configuration schema imported successfully")
        
        # Test constants import
        from geoworkflow.core.constants import (
            EARTH_ENGINE_DATASETS,
            DEFAULT_BUILDING_CONFIDENCE,
            EE_ERROR_PATTERNS
        )
        print("✅ Earth Engine constants imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_phase_1_2_configuration():
    """Test configuration with temporary files."""
    print("\nTesting Phase 1.2 configuration...")
    
    try:
        from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a temporary AOI file (simple GeoJSON)
            aoi_file = temp_path / "test_aoi.geojson"
            aoi_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [-1.0, 5.0], [-1.0, 6.0], [0.0, 6.0], [0.0, 5.0], [-1.0, 5.0]
                            ]]
                        },
                        "properties": {"name": "test_area"}
                    }
                ]
            }
            
            with open(aoi_file, 'w') as f:
                json.dump(aoi_data, f)
            
            # Create a temporary service account key file
            service_key_file = temp_path / "service_key.json"
            key_data = {
                "type": "service_account",
                "project_id": "test-project-12345",
                "private_key_id": "test-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n",
                "client_email": "test@test-project-12345.iam.gserviceaccount.com",
                "client_id": "123456789",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
            
            with open(service_key_file, 'w') as f:
                json.dump(key_data, f)
            
            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Now test configuration creation
            config = OpenBuildingsExtractionConfig(
                aoi_file=aoi_file,
                output_dir=output_dir,
                service_account_key=service_key_file,
                confidence_threshold=0.8,
                min_area_m2=15.0,
                export_format="geojson"
            )
            
            print(f"✅ Configuration created successfully!")
            print(f"   - AOI file: {config.aoi_file}")
            print(f"   - Output directory: {config.output_dir}")
            print(f"   - Confidence threshold: {config.confidence_threshold}")
            print(f"   - Export format: {config.export_format}")
            print(f"   - Min area: {config.min_area_m2} m²")
            
            # Test configuration methods
            output_file = config.get_output_file_path()
            print(f"   - Output file path: {output_file}")
            
            guidance = config.get_academic_setup_guidance()
            print(f"✅ Academic guidance available: {len(guidance)} characters")
            
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_earth_engine_availability():
    """Test Earth Engine availability check."""
    print("\nTesting Earth Engine availability...")
    
    try:
        from geoworkflow.utils.earth_engine_utils import check_earth_engine_available
        
        available = check_earth_engine_available()
        if available:
            print("✅ Earth Engine API is available")
        else:
            print("⚠️  Earth Engine API not available (expected - no installation)")
            print("   This is normal if earthengine-api is not installed")
        
        return True
        
    except Exception as e:
        print(f"❌ Earth Engine availability test failed: {e}")
        return False

def test_earth_engine_constants():
    """Test Earth Engine constants are properly defined."""
    print("\nTesting Earth Engine constants...")
    
    try:
        from geoworkflow.core.constants import (
            EARTH_ENGINE_DATASETS,
            DEFAULT_BUILDING_CONFIDENCE,
            EE_ERROR_PATTERNS,
            DEFAULT_EE_RETRY_ATTEMPTS
        )
        
        print(f"✅ Earth Engine datasets defined: {list(EARTH_ENGINE_DATASETS.keys())}")
        print(f"✅ Default confidence threshold: {DEFAULT_BUILDING_CONFIDENCE}")
        print(f"✅ Error patterns defined: {list(EE_ERROR_PATTERNS.keys())}")
        print(f"✅ Default retry attempts: {DEFAULT_EE_RETRY_ATTEMPTS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def main():
    """Run all Phase 1.2 tests."""
    print("=" * 60)
    print("PHASE 1.2 VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        test_phase_1_2_imports,
        test_phase_1_2_configuration, 
        test_earth_engine_availability,
        test_earth_engine_constants
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("PHASE 1.2 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    
    if failed == 0:
        print("\n🎉 PHASE 1.2 VALIDATION: COMPLETE!")
        print("Ready to proceed to Phase 2 (Main Processor Implementation)")
    else:
        print(f"\n⚠️  Phase 1.2 has {failed} failing components")
        print("Please address issues before proceeding to Phase 2")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)