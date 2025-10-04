Configuration Schemas
=====================

The schemas module defines Pydantic models for validating and managing
configuration files used throughout the GeoWorkflow pipeline.

Config Models
-------------

Pydantic models for YAML configuration files.

.. automodule:: geoworkflow.schemas.config_models
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Configuration Structure
-----------------------

The configuration system uses nested Pydantic models to define:

* **Dataset configurations**: Specify data sources, formats, and processing parameters
* **Processing configurations**: Define clipping, alignment, and transformation steps
* **Output configurations**: Control output formats, locations, and naming conventions
* **Validation rules**: Ensure configuration files are valid before processing

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   name: "Ghana Urban Analysis"
   version: "1.0"
   
   base_dir: "data/"
   
   aoi:
     input_file: "data/boundaries/africa.geojson"
     countries: ["Ghana"]
     output_file: "data/aoi/ghana.geojson"
   
   processing:
     stages: ["extract", "clip", "align"]
     max_workers: 4
     log_level: "INFO"
   
   clipping:
     input_directory: "data/01_extracted/"
     output_dir: "data/02_clipped/"
     aoi_file: "data/aoi/ghana.geojson"
     recursive: true
   
   alignment:
     reference_raster: "data/02_clipped/reference.tif"
     input_directory: "data/02_clipped/"
     output_dir: "data/03_aligned/"
     resampling_method: "cubic"
     auto_detect_nodata: true

Loading Configurations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geoworkflow.schemas.config_models import WorkflowConfig
   
   # Load from YAML
   config = WorkflowConfig.from_yaml("workflow.yaml")
   
   # Access configuration
   print(f"Processing stages: {config.processing.stages}")
   print(f"AOI file: {config.clipping.aoi_file}")
   
   # Validate configuration
   if config.has_stage("clip"):
       clip_config = config.get_stage_config("clip")
       print(f"Will clip to: {clip_config.aoi_file}")
