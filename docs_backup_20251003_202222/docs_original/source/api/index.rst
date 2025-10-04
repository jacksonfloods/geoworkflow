API Reference
=============

This section provides detailed API documentation for all modules, classes, and functions
in the GeoWorkflow package.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   core
   processors
   schemas
   utils

Module Overview
---------------

The GeoWorkflow API is organized into four main modules:

* **core** - Foundational classes, constants, and exceptions
* **processors** - Data extraction, clipping, and alignment processors
* **schemas** - Pydantic configuration models for YAML configs
* **utils** - Helper functions for file operations

Quick Navigation
----------------

Use the module links above to explore the API, or use the search function
to find specific classes, functions, or methods.

Usage Examples
--------------

Basic Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.core.pipeline import ProcessingPipeline
   from geoworkflow.schemas.config_models import WorkflowConfig
   
   # Load configuration
   config = WorkflowConfig.from_yaml("config.yaml")
   
   # Create and run pipeline
   pipeline = ProcessingPipeline(config)
   result = pipeline.run()
   
   if result.success:
       print(f"Processed {result.total_files_processed} files")

Working with Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.schemas.config_models import (
       WorkflowConfig, 
       ClippingConfig,
       AlignmentConfig
   )
   
   # Create configuration programmatically
   config = WorkflowConfig(
       name="Ghana Processing",
       base_dir="data/",
       clipping=ClippingConfig(
           input_directory="data/01_extracted/",
           output_dir="data/02_clipped/",
           aoi_file="data/aoi/ghana.geojson"
       )
   )
   
   # Save to YAML
   config.to_yaml("config.yaml")

Using Processors Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.processors.spatial.clipper import ClippingProcessor
   from geoworkflow.schemas.config_models import ClippingConfig
   
   # Create configuration
   config = ClippingConfig(
       input_directory="data/rasters/",
       output_dir="data/clipped/",
       aoi_file="aoi.geojson"
   )
   
   # Run clipper
   processor = ClippingProcessor(config)
   result = processor.process()
   
   print(f"Clipped {result.processed_count} files")

Using File Utilities
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.utils.file_utils import (
       find_raster_files,
       extract_archive,
       get_raster_info
   )
   
   # Find all raster files
   raster_files = find_raster_files(
       directory="data/",
       recursive=True
   )
   
   # Extract archive
   extracted = extract_archive(
       archive_path="data.zip",
       output_dir="extracted/"
   )
   
   # Get raster metadata
   for raster in raster_files:
       info = get_raster_info(raster)
       print(f"{raster}: {info['width']}x{info['height']}, CRS: {info['crs']}")

See Also
--------

* :doc:`../tutorials/index` - Step-by-step tutorials (if available)
* :doc:`../schemas/index` - Configuration file documentation
* :ref:`genindex` - Complete index of all functions and classes
* :ref:`modindex` - Module index
