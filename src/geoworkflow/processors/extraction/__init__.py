# File: src/geoworkflow/processors/extraction/__init__.py
"""
Extraction processors for the geoworkflow package.

This module provides processors for extracting geospatial data from various sources:
- ArchiveExtractionProcessor: Extract data from ZIP archives
- GEERasterExportProcessor: Declarative Earth Engine raster downloads (any image/collection)
- OpenBuildingsExtractionProcessor: Extract building footprints from Google Open Buildings dataset
- SatelliteImageryProcessor: Extract Sentinel-2 RGB imagery from Google Earth Engine
"""

from .archive import ArchiveExtractionProcessor
from .gee_raster_export import GEERasterExportProcessor, export_gee_rasters
from .open_buildings import OpenBuildingsExtractionProcessor
from .open_buildings_gcs import OpenBuildingsGCSProcessor
from .satellite_imagery import SatelliteImageryProcessor

__all__ = [
    "ArchiveExtractionProcessor",
    "GEERasterExportProcessor",
    "export_gee_rasters",
    "OpenBuildingsExtractionProcessor",
    "OpenBuildingsGCSProcessor",
    "SatelliteImageryProcessor"
]
