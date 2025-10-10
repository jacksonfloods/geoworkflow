# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

# -- Project information -----------------------------------------------------
project = 'AfricaPolis GeoWorkflow'
copyright = '2025, AfricaPolis Team'
author = 'AfricaPolis Team'

try:
    from geoworkflow import __version__
    release = __version__
except ImportError:
    release = '0.1.0'

version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_click',
    'nbsphinx',
    'nbsphinx_link',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',  # For cards and grids
]

# Napoleon settings (NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary
autosummary_generate = True

# MyST Parser settings (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]

# Templates and static files
templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "text": "AfricaPolis GeoWorkflow",
        "image_light": "_static/images/logo.png",  # Optional
        "image_dark": "_static/images/logo.png",   # Optional
    },
    "github_url": "https://github.com/jacksonfloods/geoworkflow",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jacksonfloods/geoworkflow",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "jacksonfloods",
    "github_repo": "geoworkflow",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_title = f"{project} {version}"
html_short_title = project
html_css_files = ["css/custom.css"]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
}