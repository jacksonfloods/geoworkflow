# File: src/geoworkflow/processors/__init__.py  
"""Geoworkflow processors."""

from .aoi.processor import AOIProcessor
from .spatial.clipper import ClippingProcessor
from .spatial.aligner import AlignmentProcessor
from .spatial.masker import MaskingProcessor
from .extraction.archive import ArchiveExtractionProcessor
from .extraction.open_buildings import OpenBuildingsExtractionProcessor

__all__ = [
    "AOIProcessor", 
    "ClippingProcessor", 
    "AlignmentProcessor", 
    "MaskingProcessor",
    "ArchiveExtractionProcessor",
    "OpenBuildingsExtractionProcessor"
]