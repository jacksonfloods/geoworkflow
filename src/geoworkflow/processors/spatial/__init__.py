# File: src/geoworkflow/processors/spatial/__init__.py
"""
Spatial processing processors for the geoworkflow package.

This module provides processors for spatial operations including:
- ClippingProcessor: Clip raster and vector data to AOI
- AlignmentProcessor: Align rasters to reference grid
"""

from .clipper import ClippingProcessor, clip_data
from .aligner import AlignmentProcessor, align_rasters
from .masker import MaskingProcessor

__all__ = [
    "ClippingProcessor", 
    "clip_data",
    "AlignmentProcessor",
    "align_rasters",
    "MaskingProcessor"
]