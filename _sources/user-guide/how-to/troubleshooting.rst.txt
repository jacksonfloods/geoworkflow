Troubleshooting
===============

Common issues and their solutions.

Installation Issues
-------------------

**Problem**: Conda environment creation fails

**Solution**: Try creating environment manually:

.. code-block:: bash

   conda create -n geoworkflow python=3.10
   conda activate geoworkflow
   conda install -c conda-forge geopandas rasterio

Processing Errors
-----------------

**Problem**: "File not found" errors

**Solution**: Check that your paths are correct and files exist:

.. code-block:: python

   from pathlib import Path
   
   # Verify files exist
   input_path = Path("data/raw/raster.tif")
   if not input_path.exists():
       print(f"File not found: {input_path}")

**Problem**: Memory errors with large rasters

**Solution**: Process in smaller chunks or increase available memory

See Also
--------

* :doc:`../../developer-guide/index` - Development troubleshooting
