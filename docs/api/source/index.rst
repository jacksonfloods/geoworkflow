GeoWorkflow Documentation
=========================

Welcome to GeoWorkflow's documentation! This toolkit provides comprehensive geospatial 
data processing capabilities for African urbanization analysis.

**GeoWorkflow** is designed to streamline the processing of diverse geospatial datasets 
including satellite imagery, population data, building footprints, and environmental metrics.

Features
--------

* **AOI Management**: Create and manage Areas of Interest from country boundaries
* **Data Extraction**: Extract data from archives and Google Earth Engine
* **Spatial Processing**: Clip, align, and mask rasters to consistent grids
* **Enrichment**: Statistical analysis and zonal statistics
* **Visualization**: Generate publication-ready maps and figures
* **CLI Interface**: Complete command-line interface for all operations

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install -e .

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Create an Area of Interest
   geoworkflow aoi create --country "Ghana" --output data/aoi/ghana.geojson

   # Clip rasters to AOI
   geoworkflow process clip --input data/01_extracted/ --aoi data/aoi/ghana.geojson --output data/02_clipped/

   # Generate visualizations
   geoworkflow visualize rasters --input data/02_clipped/ --output outputs/visualizations/

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   schemas/index

.. toctree::
   :maxdepth: 1
   :caption: Resources

   literature/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project Documentation
=====================

For comprehensive project guides, tutorials, and structural documentation, see the
`Project Guide <https://your-mkdocs-url/>`_.

This API reference provides detailed documentation of all classes and functions.