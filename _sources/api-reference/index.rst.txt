API Reference
=============

Complete technical reference for all modules, classes, and functions in GeoWorkflow.

.. note::
   
   This is information-oriented reference material. If you're looking for tutorials
   or how-to guides, see :doc:`../getting-started/index` or :doc:`../user-guide/index`.

Module Overview
---------------

The GeoWorkflow API is organized into four main modules:

* **core** - Foundational classes, constants, and exceptions
* **processors** - Data extraction, clipping, and alignment processors
* **schemas** - Pydantic configuration models for YAML configs
* **utils** - Helper functions for file operations

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core

Processors
----------

.. toctree::
   :maxdepth: 2

   processors

Schemas
-------

.. toctree::
   :maxdepth: 2

   schemas

Utilities
---------

.. toctree::
   :maxdepth: 2

   utils

Auto-Generated API
------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   geoworkflow.core
   geoworkflow.processors
   geoworkflow.schemas
   geoworkflow.utils
   geoworkflow.cli
