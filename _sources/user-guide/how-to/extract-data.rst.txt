How to Extract Data
===================

This guide shows you how to extract data from archives and remote sources.

.. note::
   
   This is a placeholder. Content will be added based on existing code examples.

Extract from Archives
---------------------

.. code-block:: python

   from geoworkflow.processors.extraction.archive import ArchiveExtractor
   
   config = {
       "archive_path": "data/archives/dataset.zip",
       "output_dir": "data/extracted"
   }
   
   processor = ArchiveExtractor(config)
   result = processor.process()

Extract Open Buildings from GCS
--------------------------------

.. code-block:: python

   from geoworkflow.processors.extraction.open_buildings_gcs import OpenBuildingsGCSProcessor
   
   config = {
       "aoi_file": "data/boundaries/aoi.geojson",
       "output_dir": "data/buildings"
   }
   
   processor = OpenBuildingsGCSProcessor(config)
   result = processor.process()

See Also
--------

* :doc:`../explanation/concepts` - Understanding data extraction
