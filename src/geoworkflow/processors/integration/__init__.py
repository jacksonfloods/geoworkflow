
# File: src/geoworkflow/processors/integration/__init__.py
"""
Integration processors for the geoworkflow package.

This module provides processors for integrating and enriching geospatial data
with statistical analysis and cross-dataset computations.
"""

from .enrichment import StatisticalEnrichmentProcessor, enrich_cities_with_statistics
from .grid_statistics import GridStatisticsProcessor, compute_grid_statistics
from .netcdf_builder import GridToNetCDFProcessor, build_grid_netcdf

__all__ = [
    "StatisticalEnrichmentProcessor",
    "enrich_cities_with_statistics",
    "GridStatisticsProcessor",
    "compute_grid_statistics",
    "GridToNetCDFProcessor",
    "build_grid_netcdf",
]