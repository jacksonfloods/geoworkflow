AAP GeoWorkflow
========================

Welcome to the **AAP's GeoWorkflow** documentation! 
This is a unified geospatial data processing workflow for Cornell University's AAP Geospatial Analysis Lab.


.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Getting Started
        :link: getting_started/index
        :link-type: doc
        :text-align: center
        :class-card: sd-rounded-3

        **New to GeoWorkflow?** Start here with installation guides, quickstart
        tutorials, and your first workflow.

    .. grid-item-card:: 📖 User Guide
        :link: user_guide/index
        :link-type: doc
        :text-align: center
        :class-card: sd-rounded-3

        **Learn how to use GeoWorkflow** with how-to guides for common tasks
        and explanations of core concepts.

    .. grid-item-card:: 📚 API Reference
        :link: api_reference/index
        :link-type: doc
        :text-align: center
        :class-card: sd-rounded-3

        **Detailed technical reference** with complete API documentation for
        all modules, classes, and functions.

    .. grid-item-card:: 🛠️ Developer Guide
        :link: developer_guide/index
        :link-type: doc
        :text-align: center
        :class-card: sd-rounded-3

        **Contributing to GeoWorkflow?** Guidelines for development, testing,
        and documentation contributions.

What is GeoWorkflow?
--------------------

GeoWorkflow is a Python-based pipeline for processing geospatial data for Stephan Schmidt's research group at Cornell University's AAP Geospatial Analysis Lab.
It aims to create a streamlined, reproducible workflow for handling large geospatial datasets.

**Key Features:**

* **Multi-stage Processing Pipeline** - Extract, clip, align, enrich, visualize
* **Modular Processors** - Extensible architecture for custom processors
* **Configuration-driven** - YAML-based workflow definitions
* **Progress Tracking** - Rich console output with detailed metrics
* **Error Handling** - Comprehensive error reporting and recovery

Quick Example
-------------

.. code-block:: python

    from geoworkflow.core.pipeline import ProcessingPipeline

    # Define workflow configuration
    config = {
        "stages": ["clip", "align", "enrich"],
        "source_dir": "data/raw",
        "output_dir": "data/processed"
    }

    # Run pipeline
    pipeline = ProcessingPipeline(config)
    results = pipeline.run()

Installation
------------

.. code-block:: bash

    # Using conda (recommended)
    conda env create -f environment.yml
    conda activate geoworkflow
    pip install -e .

See the :doc:`getting_started/installation` guide for detailed instructions.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   getting_started/index
   user_guide/index
   api_reference/index
   developer_guide/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Links

   GitHub Repository <https://github.com/jacksonfloods/geoworkflow>
   Issue Tracker <https://github.com/jacksonfloods/geoworkflow/issues>
   Cornell AAP Research Group <https://labs.aap.cornell.edu/stephan-schmidts-research-group/team>
