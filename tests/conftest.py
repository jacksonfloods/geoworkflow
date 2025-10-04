# tests/conftest.py
"""
Pytest configuration and shared fixtures for geoworkflow tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Try to import geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False

# Try to import core components
try:
    from geoworkflow.schemas.config_models import WorkflowConfig
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_aoi_small(temp_dir):
    """Create a small AOI for quick tests."""
    if not HAS_GEOSPATIAL:
        pytest.skip("GeoPandas not available")
    
    aoi_file = temp_dir / "small_aoi.geojson"
    gdf = gpd.GeoDataFrame(
        {'name': ['Small Test Area']},
        geometry=[box(-0.1, 5.5, -0.05, 5.55)],
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.fixture
def sample_aoi_medium(temp_dir):
    """Create a medium-sized AOI."""
    if not HAS_GEOSPATIAL:
        pytest.skip("GeoPandas not available")
    
    aoi_file = temp_dir / "medium_aoi.geojson"
    gdf = gpd.GeoDataFrame(
        {'name': ['Medium Test Area']},
        geometry=[box(-1, 5, 0, 6)],
        crs="EPSG:4326"
    )
    gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.fixture
def sample_polygon():
    """Create a sample polygon geometry."""
    if not HAS_GEOSPATIAL:
        pytest.skip("Shapely not available")
    
    return box(0, 0, 1, 1)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gcs: marks tests that require GCS access"
    )