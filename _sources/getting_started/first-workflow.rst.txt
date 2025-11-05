Basic Workflow Tutorial
=======================

This tutorial walks through a complete data processing workflow using GeoWorkflow.

Scenario
--------

Process PM2.5 air quality data for Ghana and Togo, enriching urban boundary polygons with pollution statistics.

Prerequisites
-------------

- GeoWorkflow installed
- Sample data in ``data/00_source/``
- AFRICAPOLIS boundaries

Step 1: Create Area of Interest
-------------------------------

.. code-block:: python

   from geoworkflow.processors.aoi.processor import AOIProcessor
   from geoworkflow.schemas.config_models import AOIConfig
   from pathlib import Path

   # Configure AOI creation
   aoi_config = AOIConfig(
       source_file=Path("data/00_source/boundaries/africa_boundaries.geojson"),
       output_dir=Path("data/aoi"),
       countries=["Ghana", "Togo"],
       buffer_km=50,
       output_format="geojson"
   )

   # Create AOI
   aoi_processor = AOIProcessor(aoi_config)
   result = aoi_processor.process()

   print(f"Created AOI: {result.output_paths[0]}")

Step 2: Extract Archives
------------------------

.. code-block:: python

   from geoworkflow.processors.extraction.archive import ArchiveExtractionProcessor
   from geoworkflow.schemas.config_models import ExtractionConfig

   extract_config = ExtractionConfig(
       source_dir=Path("data/00_source/archives/pm25"),
       output_dir=Path("data/01_extracted/pm25"),
       archive_format="zip"
   )

   extractor = ArchiveExtractionProcessor(extract_config)
   result = extractor.process()

Step 3: Clip to AOI
-------------------

.. code-block:: python

   from geoworkflow.processors.spatial.clipper import ClippingProcessor
   from geoworkflow.schemas.config_models import ClippingConfig

   clip_config = ClippingConfig(
       input_dir=Path("data/01_extracted/pm25"),
       output_dir=Path("data/02_clipped/pm25"),
       boundary_file=Path("data/aoi/ghana_togo_aoi.geojson"),
       maintain_extent=False
   )

   clipper = ClippingProcessor(clip_config)
   result = clipper.process()

Step 4: Align Rasters
---------------------

.. code-block:: python

   from geoworkflow.processors.spatial.aligner import AlignmentProcessor
   from geoworkflow.schemas.config_models import AlignmentConfig

   align_config = AlignmentConfig(
       input_dir=Path("data/02_clipped/pm25"),
       output_dir=Path("data/03_processed/pm25"),
       target_crs="EPSG:4326",
       resolution=(0.001, 0.001),  # ~100m at equator
       resampling_method="bilinear"
   )

   aligner = AlignmentProcessor(align_config)
   result = aligner.process()

Step 5: Enrich Urban Boundaries
-------------------------------

.. code-block:: python

   from geoworkflow.processors.integration.enrichment import StatisticalEnrichmentProcessor
   from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig

   enrich_config = StatisticalEnrichmentConfig(
       vector_file=Path("data/01_extracted/AFRICAPOLIS2020.geojson"),
       raster_dir=Path("data/03_processed/pm25"),
       output_dir=Path("data/04_analysis_ready"),
       statistics=["mean", "std", "min", "max"],
       prefix="pm25"
   )

   enricher = StatisticalEnrichmentProcessor(enrich_config)
   result = enricher.process()

   print(f"Enriched {result.processed_count} urban areas")

Step 6: Examine Results
-----------------------

.. code-block:: python

   import geopandas as gpd

   # Load enriched data
   gdf = gpd.read_file("data/04_analysis_ready/africapolis_pm25_stats.geojson")

   # View statistics
   print(gdf[["AgglomName", "pm25_mean", "pm25_std"]].head())

   # Simple visualization
   gdf.plot(column="pm25_mean", legend=True, figsize=(12, 8))

Complete Pipeline Version
-------------------------

Or run everything at once:

.. code-block:: python

   from geoworkflow.core.pipeline import ProcessingPipeline

   pipeline_config = {
       "stages": ["extract", "clip", "align", "enrich"],
       "source_dir": Path("data/00_source"),
       "countries": ["Ghana", "Togo"],
       "target_crs": "EPSG:4326",
       "resolution_m": 100
   }

   pipeline = ProcessingPipeline(pipeline_config)
   results = pipeline.run()

   for stage, result in results.items():
       print(f"{stage}: {result.processed_count} files processed")

Troubleshooting
---------------

| **Issue**: Files not found
| **Solution**: Check that paths are correct and files exist

| **Issue**: CRS mismatch warnings
| **Solution**: This is expected - the alignment stage handles it

| **Issue**: Memory errors with large rasters
| **Solution**: Process smaller regions or use windowed reading
