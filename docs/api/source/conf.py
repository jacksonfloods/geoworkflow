# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parents[3] / 'src'))

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
