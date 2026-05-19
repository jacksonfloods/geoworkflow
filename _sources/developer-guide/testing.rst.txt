Testing Guide
=============

GeoWorkflow uses pytest for testing.

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=geoworkflow --cov-report=html

   # Run specific test file
   pytest tests/test_aoi_processor.py

   # Run specific test
   pytest tests/test_aoi_processor.py::test_aoi_creation

   # Run with verbose output
   pytest -v

   # Run only fast tests
   pytest -m "not slow"

Test Structure
--------------

::

   tests/
   ├── conftest.py              # Shared fixtures
   ├── fixtures/                # Test data
   │   ├── rasters/
   │   ├── vectors/
   │   └── configs/
   ├── unit/                    # Unit tests
   │   ├── test_processors/
   │   ├── test_utils/
   │   └── test_schemas/
   └── integration/             # Integration tests
       └── test_pipelines/

Writing Tests
-------------

Basic Test Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from pathlib import Path
   from geoworkflow.processors.aoi.processor import AOIProcessor

   def test_aoi_processor_creates_output(tmp_path):
       """Test that AOI processor creates output file."""
       config = {
           "source_file": "tests/fixtures/boundaries.geojson",
           "output_dir": tmp_path,
           "countries": ["Ghana"],
           "output_format": "geojson"
       }
       
       processor = AOIProcessor(config)
       result = processor.process()
       
       assert result.success
       assert len(result.output_paths) > 0
       assert result.output_paths[0].exists()

Using Fixtures
~~~~~~~~~~~~~~

.. code-block:: python

   # conftest.py
   @pytest.fixture
   def sample_raster(tmp_path):
       """Create a sample raster for testing."""
       import rasterio
       import numpy as np
       
       raster_path = tmp_path / "test_raster.tif"
       data = np.random.rand(100, 100)
       
       with rasterio.open(
           raster_path, 'w',
           driver='GTiff',
           height=100, width=100,
           count=1, dtype=data.dtype,
           crs='EPSG:4326',
           transform=rasterio.transform.from_bounds(0, 0, 1, 1, 100, 100)
       ) as dst:
           dst.write(data, 1)
       
       return raster_path

   # test_file.py
   def test_with_fixture(sample_raster):
       """Test using the fixture."""
       assert sample_raster.exists()

Testing Errors
~~~~~~~~~~~~~~

.. code-block:: python

   def test_processor_invalid_config():
       """Test that invalid config raises error."""
       with pytest.raises(ValueError, match="Missing required"):
           processor = MyProcessor({})

Parametrized Tests
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.parametrize("crs,expected", [
       ("EPSG:4326", "WGS84"),
       ("EPSG:3857", "Web Mercator"),
   ])
   def test_crs_names(crs, expected):
       """Test CRS name conversion."""
       result = get_crs_name(crs)
       assert expected in result

Test Markers
------------

.. code-block:: python

   # Mark slow tests
   @pytest.mark.slow
   def test_large_dataset():
       pass

   # Mark integration tests
   @pytest.mark.integration
   def test_full_pipeline():
       pass

   # Run only unit tests
   pytest -m "not integration"

Coverage Reports
----------------

.. code-block:: bash

   # Generate HTML coverage report
   pytest --cov=geoworkflow --cov-report=html

   # Open report
   open htmlcov/index.html

Continuous Integration
----------------------

Tests run automatically on every PR via GitHub Actions.
