Configuration Guide
===================

GeoWorkflow uses YAML configuration files to define workflows and processor settings.

Configuration File Structure
----------------------------

.. code-block:: yaml

   # workflow_config.yaml
   workflow:
     name: "AfricaPolis Processing"
     stages:
       - extract
       - clip
       - align
       - enrich

   paths:
     source_dir: "data/00_source"
     extracted_dir: "data/01_extracted"
     clipped_dir: "data/02_clipped"
     processed_dir: "data/03_processed"
     output_dir: "data/04_analysis_ready"

   aoi:
     countries: ["Ghana", "Togo", "Kenya", "Tanzania"]
     buffer_km: 50

   processing:
     target_crs: "EPSG:4326"
     resolution_m: 100
     resampling_method: "bilinear"

Loading Configuration
---------------------

.. code-block:: python

   from geoworkflow.core.config import ConfigManager

   # Load from file
   config = ConfigManager.load("config/workflows/my_workflow.yaml")

   # Or create programmatically
   config = {
       "stages": ["clip", "align"],
       "source_dir": "data/raw"
   }

Environment Variables
---------------------

Set these environment variables for Earth Engine integration:

.. code-block:: bash

   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   export GEE_PROJECT_ID="your-project-id"

See Also
--------

- `Processor-specific configs <../guide/concepts.md#configuration-models>`__
- `Workflow examples <../tutorials/basic-workflow.md>`__
