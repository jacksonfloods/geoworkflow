Documentation Guidelines
========================

How to build and contribute to GeoWorkflow documentation.

Building Docs Locally
----------------------

.. code-block:: bash

   cd docs
   make html
   
   # Serve locally
   python -m http.server -d _build/html

Live Reload During Development
-------------------------------

.. code-block:: bash

   cd docs
   make livehtml

Documentation Structure
-----------------------

GeoWorkflow follows the Diátaxis framework:

* **Getting Started** - Learning-oriented tutorials
* **User Guide** - Problem-solving how-tos and explanations
* **API Reference** - Technical reference material
* **Developer Guide** - Contributing guidelines

Writing Guidelines
------------------

* Use NumPy-style docstrings in code
* Write in ReStructuredText (.rst) format
* Include code examples where appropriate
* Keep tutorials hands-on and goal-oriented

See Also
--------

* `Diátaxis Documentation Framework <https://diataxis.fr/>`_
* `Sphinx Documentation <https://www.sphinx-doc.org/>`_
* `PyData Theme Documentation <https://pydata-sphinx-theme.readthedocs.io/>`_
