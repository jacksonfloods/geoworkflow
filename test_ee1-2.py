#!/usr/bin/env python3
"""
Integration test for Phase 1.2: Earth Engine Utilities
Tests the complete Earth Engine utilities without actual EE calls.
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test both with and without Earth Engine available
def test_without_earth_engine():
    """Test behavior when Earth Engine is not available."""
    print("=" * 60)
    print("Testing Phase 1.2: Earth Engine NOT Available")
    print("=" * 60)
    
    with patch('geoworkflow.utils.earth_engine_utils.HAS_EARTH_ENGINE', False):
        from geoworkflow.utils.earth_engine_utils import (
            check_earth_engine_available,
            validate_earth_engine_setup
        )
        
        # Test availability check
        available = check_earth_engine_available()
        print(f"✓ Earth Engine availability check: {available}")
        assert available is False
        
        # Test validation
        validation = validate_earth_engine_setup()
        print(f"✓ Validation result: {validation['valid']}")
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        print(f"  - Errors: {validation['errors']}")

def test_with_mocked_earth_engine():
    """Test behavior with mocked Earth Engine."""
    print("\n" + "=" * 60)
    print("Testing Phase 1.2: Earth Engine Available (Mocked)")
    print("=" * 60)
    
    with patch('geoworkflow.utils.earth_engine_utils.HAS_EARTH_ENGINE', True), \
         patch('geoworkflow.utils.earth_engine_utils.ee') as mock_ee, \
         patch('geoworkflow.utils.earth_engine_utils.gpd') as mock_gpd:
        
        from geoworkflow.utils.earth_engine_utils import (
            check_earth_engine_available,
            validate_earth_engine_setup,
            EarthEngineAuth,
            OpenBuildingsAPI
        )
        
        # Test availability
        available = check_earth_engine_available()
        print(f"✓ Earth Engine availability check: {available}")
        assert available is True
        
        # Test validation with service account
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock service account key
            key_data = {
                "type": "service_account",
                "project_id": "test-academic-project",
                "private_key_id": "key123",
                "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
                "client_email": "test@test-academic-project.iam.gserviceaccount.com"
            }
            
            key_file = Path(temp_dir) / "service_account.json"
            with open(key_file, 'w') as f:
                json.dump(key_data, f)
            
            validation = validate_earth_engine_setup(service_account_key=key_file)
            print(f"✓ Validation with service account: {validation['valid']}")
            print(f"  - Available auth methods: {validation['info'].get('available_auth_methods', [])}")
            
            # Test authentication
            mock_ee.ServiceAccountCredentials.return_value = MagicMock()
            mock_ee.Initialize.return_value = None
            
            project_id = EarthEngineAuth.authenticate(service_account_key=key_file)
            print(f"✓ Authentication successful, project: {project_id}")
            assert project_id == "test-academic-project"
            
            # Test API initialization
            mock_ee.FeatureCollection.return_value = MagicMock()
            
            api = OpenBuildingsAPI(project_id, "v3")
            print(f"✓ OpenBuildingsAPI initialized for project: {api.project_id}")
            assert api.project_id == project_id
            assert api.dataset_version == "v3"

def test_configuration_integration():
    """Test integration with Phase 1.1 configuration."""
    print("\n" + "=" * 60)
    print("Testing Configuration Integration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test AOI file
        aoi_file = Path(temp_dir) / "test_aoi.geojson"
        aoi_content = {
            "type": "FeatureCollection", 
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-0.1, 5.5], [0.1, 5.5], [0.1, 5.7], [-0.1, 5.7], [-0.1, 5.5]
                        ]]
                    }
                }
            ]
        }
        
        with open(aoi_file, 'w') as f:
            json.dump(aoi_content, f)
        
        # Test configuration with utilities
        from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
        from geoworkflow.utils.earth_engine_utils import validate_earth_engine_setup
        
        config = OpenBuildingsExtractionConfig(
            aoi_file=aoi_file,
            output_dir=Path(temp_dir) / "output",
            confidence_threshold=0.8,
            min_area_m2=20.0
        )
        
        print(f"✓ Configuration created successfully")
        print(f"  - AOI file: {config.aoi_file}")
        print(f"  - Confidence threshold: {config.confidence_threshold}")
        print(f"  - Min area: {config.min_area_m2}")
        
        # Test utilities validation
        validation = validate_earth_engine_setup()
        print(f"✓ Earth Engine validation completed")
        print(f"  - Valid: {validation['valid']}")
        
        if not validation['valid']:
            print(f"  - Setup guidance: {validation['setup_guidance']}")

def test_error_handling():
    """Test comprehensive error handling."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    from geoworkflow.core.exceptions import (
        EarthEngineAuthenticationError,
        EarthEngineQuotaError, 
        EarthEngineGeometryError
    )
    
    # Test that custom exceptions exist and are properly structured
    print("✓ Earth Engine exceptions imported successfully")
    
    # Test exception hierarchy
    assert issubclass(EarthEngineAuthenticationError, Exception)
    assert issubclass(EarthEngineQuotaError, Exception)
    assert issubclass(EarthEngineGeometryError, Exception)
    
    print("✓ Exception hierarchy validated")

if __name__ == "__main__":
    print("🚀 Starting Phase 1.2 Integration Tests")
    
    try:
        test_without_earth_engine()
        test_with_mocked_earth_engine()
        test_configuration_integration()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ PHASE 1.2 INTEGRATION TESTS PASSED!")
        print("✅ Earth Engine Utilities Ready for Phase 2")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        raise