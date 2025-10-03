#!/bin/bash
# Setup script for Sphinx documentation structure
# Run from the geoworkflow/ root directory

set -e  # Exit on error

echo "🚀 Setting up Sphinx documentation structure..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p docs/source/_static
mkdir -p docs/source/_templates
mkdir -p docs/source/api
mkdir -p docs/source/tutorials
mkdir -p docs/source/schemas
mkdir -p docs/source/literature
mkdir -p docs/build/html

echo -e "${GREEN}✓ Directories created${NC}"

# Create conf.py
echo -e "${BLUE}Creating conf.py...${NC}"
cat > docs/source/conf.py << 'EOF'
# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

# -- Project information -----------------------------------------------------
project = 'GeoWorkflow'
copyright = '2025, AfricaPolis Team'
author = 'AfricaPolis Team'

# The full version, including alpha/beta/rc tags
try:
    from geoworkflow import __version__
    release = __version__
except ImportError:
    release = '0.1.0'

version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate API docs from docstrings
    'sphinx.ext.napoleon',          # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other project docs
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx_autodoc_typehints',     # Better type hint support
    'sphinx_click',                 # Document Click CLI commands
    'nbsphinx',                     # Jupyter notebook integration
    'nbsphinx_link',                # Link to notebooks without copying
    'myst_parser',                  # Markdown support
    'sphinx_copybutton',            # Copy button for code blocks
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True

# NBSphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue on notebook errors
nbsphinx_kernel_name = 'python3'

# Templates path
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "text": "GeoWorkflow Documentation",
    },
    "github_url": "https://github.com/jacksonfloods/geoworkflow",  # Update with your repo
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "footer_items": ["copyright", "sphinx-version"],
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Favicon and logo (uncomment when you add them)
# html_favicon = '_static/favicon.ico'
# html_logo = '_static/logo.png'

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
}

# -- Markdown support --------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST settings for advanced markdown features
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]
EOF

echo -e "${GREEN}✓ conf.py created${NC}"

# Create index.rst
echo -e "${BLUE}Creating index.rst...${NC}"
cat > docs/source/index.rst << 'EOF'
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
EOF

echo -e "${GREEN}✓ index.rst created${NC}"

# Create API index
echo -e "${BLUE}Creating api/index.rst...${NC}"
cat > docs/source/api/index.rst << 'EOF'
API Reference
=============

This section contains the complete API reference for GeoWorkflow.

.. toctree::
   :maxdepth: 2
   :caption: Modules

Core Modules
------------

Coming soon: Core functionality documentation

Processors
----------

Coming soon: Data processors documentation

Utilities
---------

Coming soon: Utility functions documentation

CLI Reference
-------------

Coming soon: Command-line interface documentation
EOF

echo -e "${GREEN}✓ api/index.rst created${NC}"

# Create tutorials index
echo -e "${BLUE}Creating tutorials/index.rst...${NC}"
cat > docs/source/tutorials/index.rst << 'EOF'
Tutorials
=========

Learn how to use GeoWorkflow through practical examples.

.. toctree::
   :maxdepth: 1
   :caption: Available Tutorials

Coming soon: Tutorial notebooks will be linked here.
EOF

echo -e "${GREEN}✓ tutorials/index.rst created${NC}"

# Create schemas index
echo -e "${BLUE}Creating schemas/index.rst...${NC}"
cat > docs/source/schemas/index.rst << 'EOF'
Configuration Schemas
=====================

Documentation for YAML configuration files used in GeoWorkflow.

Coming soon: Configuration schema documentation
EOF

echo -e "${GREEN}✓ schemas/index.rst created${NC}"

# Create literature index
echo -e "${BLUE}Creating literature/index.rst...${NC}"
cat > docs/source/literature/index.rst << 'EOF'
Literature & References
=======================

Key papers and resources related to African urbanization analysis.

Coming soon: Links to literature PDFs and references
EOF

echo -e "${GREEN}✓ literature/index.rst created${NC}"

# Create custom.css
echo -e "${BLUE}Creating _static/custom.css...${NC}"
cat > docs/source/_static/custom.css << 'EOF'
/* Custom CSS for GeoWorkflow documentation */

/* Adjust code block styling */
div.highlight {
    border-radius: 4px;
}

/* Make tables more readable */
table.docutils {
    border: none;
    border-collapse: collapse;
}

table.docutils td, table.docutils th {
    border: 1px solid #e1e4e5;
    padding: 8px;
}

/* Improve sidebar navigation */
.bd-sidebar {
    font-size: 0.9rem;
}
EOF

echo -e "${GREEN}✓ custom.css created${NC}"

# Create Makefile
echo -e "${BLUE}Creating Makefile...${NC}"
cat > docs/Makefile << 'EOF'
# Minimal makefile for Sphinx documentation

# You can set these variables from the command line
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
EOF

echo -e "${GREEN}✓ Makefile created${NC}"

# Create make.bat for Windows users
echo -e "${BLUE}Creating make.bat...${NC}"
cat > docs/make.bat << 'EOF'
@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
EOF

echo -e "${GREEN}✓ make.bat created${NC}"

# Create .gitignore additions
echo -e "${BLUE}Creating docs/.gitignore...${NC}"
cat > docs/.gitignore << 'EOF'
# Sphinx build artifacts
build/*
!build/html/
build/html/.buildinfo
build/html/.doctrees/

# Jupyter notebook checkpoints (if any get created)
.ipynb_checkpoints/
EOF

echo -e "${GREEN}✓ .gitignore created${NC}"

# Make scripts executable
chmod +x docs/Makefile

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Sphinx documentation structure created!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Install dependencies:"
echo "   pip install sphinx pydata-sphinx-theme sphinx-autodoc-typehints sphinx-click nbsphinx nbsphinx-link myst-parser sphinx-copybutton"
echo ""
echo "2. Build the documentation:"
echo "   cd docs"
echo "   make html"
echo ""
echo "3. View the documentation:"
echo "   open build/html/index.html"
echo "   # Or serve it:"
echo "   cd build/html && python -m http.server 8000"
echo ""
echo "4. For live rebuilding during development:"
echo "   pip install sphinx-autobuild"
echo "   make livehtml"
echo ""
EOF

echo -e "${GREEN}✓ setup_sphinx_docs.sh created${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
