# File: src/geoworkflow/processors/__init__.py  
"""Geoworkflow processors."""

from .aoi.processor import AOIProcessor
from .spatial.clipper import ClippingProcessor
from .spatial.aligner import AlignmentProcessor
from .spatial.masker import MaskingProcessor
from .spatial.hexgrid import HexGridProcessor, create_hex_grid_processor
from .integration.grid_statistics import GridStatisticsProcessor, compute_grid_statistics
from .extraction.archive import ArchiveExtractionProcessor
from .extraction.open_buildings import OpenBuildingsExtractionProcessor

__all__ = [
    "AOIProcessor",
    "ClippingProcessor",
    "AlignmentProcessor",
    "MaskingProcessor",
    "HexGridProcessor",
    "create_hex_grid_processor",
    "GridStatisticsProcessor",
    "compute_grid_statistics",
    "ArchiveExtractionProcessor",
    "OpenBuildingsExtractionProcessor",
]