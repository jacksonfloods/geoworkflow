# File: src/geoworkflow/processors/extraction/__init__.py
"""
Extraction processors for the geoworkflow package.

This module provides processors for extracting geospatial data from various sources:
- ArchiveExtractionProcessor: Extract data from ZIP archives
- OpenBuildingsExtractionProcessor: Extract building footprints from Google Open Buildings dataset
"""

from .archive import ArchiveExtractionProcessor
from .open_buildings import OpenBuildingsExtractionProcessor
from .open_buildings_gcs import OpenBuildingsGCSProcessor

__all__ = [
    "ArchiveExtractionProcessor",
    "OpenBuildingsExtractionProcessor",
    "OpenBuildingsGCSProcessor"
]