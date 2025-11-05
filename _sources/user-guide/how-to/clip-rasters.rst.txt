How to Clip Rasters
===================

This guide shows you how to clip raster datasets to a specific boundary.

Prerequisites
-------------

* Installed GeoWorkflow
* Boundary file (GeoJSON or Shapefile)
* Raster files to clip

.. note::
   
   This is a placeholder. Content will be added based on existing code examples.

Basic Clipping
--------------

.. code-block:: python

   from geoworkflow.processors.spatial.clipper import ClippingProcessor
   
   config = {
       "input_dir": "data/raw",
       "output_dir": "data/clipped",
       "boundary_file": "data/boundaries/aoi.geojson"
   }
   
   processor = ClippingProcessor(config)
   result = processor.process()

See Also
--------

* :doc:`align-datasets` - Align clipped rasters to a common grid
* :doc:`../explanation/workflow-stages` - Understanding the clipping stage
