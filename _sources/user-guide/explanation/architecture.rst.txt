Project Structure
=================

This page documents the organization of the GeoWorkflow codebase.

Interactive Directory Tree
--------------------------

Explore the project structure below. Click on folders to expand/collapse, and hover over items to see descriptions.

.. note::

!!! tip “How to Use” - **Click** on folder nodes (📁) to expand or collapse them - **Hover** over any node to see detailed descriptions - **Blue nodes** represent directories - **Green nodes** represent Python files

??? note “Cant see the tree?” If the interactive tree doesnt load, you can `view it directly <../../assets/directory-tree-container.html>`__ or see the text version below.

--------------

Source Code Layout
------------------

::

   src/geoworkflow/
   │   ├── __init__.py
   │   ├── __version__.py
   │   ├── visualization/
   │   │   ├── __init__.py
   │   │   ├── raster/
   │   │   │   ├── __init__.py
   │   │   │   ├── processor.py
   │   │   ├── vector/
   │   │   │   ├── __init__.py
   │   │   ├── reports/
   │   │   │   ├── __init__.py
   │   ├── core/
   │   │   ├── __init__.py
   │   │   ├── base.py
   │   │   ├── config.py
   │   │   ├── constants.py
   │   │   ├── enhanced_base.py
   │   │   ├── exceptions.py
   │   │   ├── logging_setup.py
   │   │   ├── pipeline.py
   │   │   ├── pipeline_enhancements.py
   │   ├── utils/
   │   │   ├── __init__.py
   │   │   ├── earth_engine_error_handler.py
   │   │   ├── earth_engine_utils.py
   │   │   ├── file_utils.py
   │   │   ├── gcs_utils.py
   │   │   ├── mask_utils.py
   │   │   ├── progress_utils.py
   │   │   ├── raster_utils.py
   │   │   ├── resource_utils.py
   │   │   ├── s2_utils.py
   │   │   ├── validation.py
   │   ├── cli/
   │   │   ├── __init__.py
   │   │   ├── cli_structure.py
   │   │   ├── main.py
   │   │   ├── commands/
   │   │   │   ├── __init__.py
   │   │   │   ├── aoi.py
   │   │   │   ├── extract.py
   │   │   │   ├── pipeline.py
   │   │   │   ├── process.py
   │   │   │   ├── visualize.py
   │   ├── schemas/
   │   │   ├── __init__.py
   │   │   ├── config_models.py
   │   │   ├── open_buildings_gcs_config.py
   │   ├── processors/
   │   │   ├── __init__.py
   │   │   ├── integration/
   │   │   │   ├── __init__.py
   │   │   │   ├── enrichment.py
   │   │   ├── spatial/
   │   │   │   ├── __init__.py
   │   │   │   ├── aligner.py
   │   │   │   ├── clipper.py
   │   │   │   ├── masker.py
   │   │   ├── extraction/
   │   │   │   ├── __init__.py
   │   │   │   ├── archive.py
   │   │   │   ├── gcs_utils.py
   │   │   │   ├── open_buildings.py
   │   │   │   ├── open_buildings_gcs.py
   │   │   ├── aoi/
   │   │   │   ├── __init__.py
   │   │   │   ├── processor.py

Directory Descriptions
----------------------

``core/``
~~~~~~~~~

Foundation classes, base processors, configuration, and constants

``processors/``
~~~~~~~~~~~~~~~

Specialized processors for each workflow stage

``processors/aoi/``
~~~~~~~~~~~~~~~~~~~

Area of Interest (AOI) creation and management

``processors/spatial/``
~~~~~~~~~~~~~~~~~~~~~~~

Spatial operations (clipping, alignment, reprojection)

``processors/extraction/``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Data extraction from archives and downloads

``processors/integration/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistical enrichment and data integration

``schemas/``
~~~~~~~~~~~~

Pydantic models for configuration validation

``utils/``
~~~~~~~~~~

Helper functions and common operations

``cli/``
~~~~~~~~

Command-line interface entry points

``cli/commands/``
~~~~~~~~~~~~~~~~~

CLI command implementations

``visualization/``
~~~~~~~~~~~~~~~~~~

Visualization components

``visualization/raster/``
~~~~~~~~~~~~~~~~~~~~~~~~~

Raster visualization processors

``visualization/vector/``
~~~~~~~~~~~~~~~~~~~~~~~~~

Vector visualization processors

``visualization/reports/``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Report generation utilities
