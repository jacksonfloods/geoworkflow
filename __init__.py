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
try:
    from geoworkflow.core.config import ConfigManager  # Changed: load_config doesn't exist as direct import
    from geoworkflow.core.pipeline import ProcessingPipeline
    from geoworkflow.core.exceptions import GeoWorkflowError
except ImportError:
    # Core modules not available yet
    ConfigManager = None
    ProcessingPipeline = None
    GeoWorkflowError = Exception

# Schema imports
try:
    from .schemas.config_models import (
        AOIConfig,
        ExtractionConfig, 
        ClippingConfig,
        AlignmentConfig,
        VisualizationConfig,
        WorkflowConfig,
        OpenBuildingsExtractionConfig
    )
except ImportError:
    # Schemas not available yet
    AOIConfig = None
    ExtractionConfig = None
    ClippingConfig = None
    AlignmentConfig = None
    VisualizationConfig = None
    WorkflowConfig = None
    OpenBuildingsExtractionConfig = None

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "ConfigManager",
    "WorkflowConfig", 
    "ProcessingPipeline",
    "GeoWorkflowError",
    "OpenBuildingsExtractionConfig",
    "AOIConfig",
    "ExtractionConfig",
    "ClippingConfig",
    "AlignmentConfig",
    "VisualizationConfig",
]