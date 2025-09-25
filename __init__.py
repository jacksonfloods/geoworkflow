# File: src/geoworkflow/__init__.py
"""
Geoworkflow: Comprehensive geospatial data processing workflow for African geospatial analysis.

This package provides a unified framework for processing geospatial data including:
- Area of Interest (AOI) creation and management
- Raster and vector data processing
- Data extraction from archives
- Spatial clipping and alignment
- Advanced visualization capabilities
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geoworkflow")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports for convenience
from geoworkflow.core.config import load_config, WorkflowConfig
from geoworkflow.core.pipeline import ProcessingPipeline
from geoworkflow.core.exceptions import GeoWorkflowError

from .schemas.config_models import (
    AOIConfig,
    ExtractionConfig, 
    ClippingConfig,
    AlignmentConfig,
    VisualizationConfig,
    WorkflowConfig,
    OpenBuildingsExtractionConfig  # Add this line
)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "load_config",
    "WorkflowConfig", 
    "ProcessingPipeline",
    "GeoWorkflowError",
    "OpenBuildingsExtractionConfig",  # Add this line
]