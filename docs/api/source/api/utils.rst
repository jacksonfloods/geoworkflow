Utilities Module
================

The utils module provides helper functions for file operations and
common tasks used throughout GeoWorkflow.

.. automodule:: geoworkflow.utils
   :members:
   :undoc-members:
   :show-inheritance:

File Utilities
--------------

Functions for file and directory operations.

.. automodule:: geoworkflow.utils.file_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

File utilities include:

* **Archive handling**: Extract ZIP files, validate archives, manage temporary directories
* **Path operations**: Resolve paths, create directories, clean filenames
* **File discovery**: Find files by pattern, filter by extension, recursive searches
* **Validation**: Check file existence, verify formats, validate permissions

Common Use Cases
~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.utils import file_utils
   
   # Find all GeoTIFF files in a directory
   tif_files = file_utils.find_files_by_extension(
       directory="data/rasters/",
       extension=".tif"
   )
   
   # Extract archive
   extracted_files = file_utils.extract_archive(
       archive_path="data.zip",
       output_dir="extracted/"
   )
   
   # Get raster information
   info = file_utils.get_raster_info("raster.tif")
   print(f"CRS: {info['crs']}, Shape: {info['shape']}")
