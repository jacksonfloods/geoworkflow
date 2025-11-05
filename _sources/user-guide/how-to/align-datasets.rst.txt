How to Align Datasets
=====================

This guide shows you how to align multiple raster datasets to a common grid.

Prerequisites
-------------

* Clipped raster datasets
* Reference raster for alignment

.. note::
   
   This is a placeholder. Content will be added based on existing code examples.

Basic Alignment
---------------

.. code-block:: python

   from geoworkflow.processors.spatial.aligner import AlignmentProcessor
   
   config = {
       "input_dir": "data/clipped",
       "output_dir": "data/aligned",
       "reference_raster": "data/reference/template.tif"
   }
   
   processor = AlignmentProcessor(config)
   result = processor.process()

See Also
--------

* :doc:`clip-rasters` - Clip rasters before alignment
* :doc:`../explanation/workflow-stages` - Understanding the alignment stage
