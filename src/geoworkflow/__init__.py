"""
GeoWorkflow - Geoworkflow: A unified geospatial data processing workflow for Cornell University's AAP Geospatial Analysis Lab.

A unified toolkit for processing, aligning, and visualizing geospatial data
with focus on African datasets including Copernicus, ODIAC, PM2.5, and AFRICAPOLIS.
"""

from .__version__ import __version__, __author__, __email__, __description__

# Core imports
from .core.base import BaseProcessor, BaseVisualizer, ProcessingResult, Pipeline
from .core.exceptions import (
    GeoWorkflowError, 
    ConfigurationError, 
    ProcessingError, 
    ValidationError,
    FileOperationError,
    GeospatialError
)
from .core.constants import ProcessingStage, DataType, DataSource, CommonCRS

# Configuration imports
from .schemas.config_models import (
    AOIConfig,
    ExtractionConfig, 
    ClippingConfig,
    AlignmentConfig,
    VisualizationConfig,
    WorkflowConfig
)

__all__ = [
    # Version info
    '__version__',
    '__author__', 
    '__email__',
    '__description__',
    
    # Core classes
    'BaseProcessor',
    'BaseVisualizer', 
    'ProcessingResult',
    'Pipeline',
    
    # Exceptions
    'GeoWorkflowError',
    'ConfigurationError',
    'ProcessingError',
    'ValidationError', 
    'FileOperationError',
    'GeospatialError',
    
    # Constants
    'ProcessingStage',
    'DataType',
    'DataSource', 
    'CommonCRS',
    
    # Configuration models
    'AOIConfig',
    'ExtractionConfig',
    'ClippingConfig',
    'AlignmentConfig',
    'VisualizationConfig',
    'WorkflowConfig',

    # Hex database (the primary interface) — lazily resolved (see __getattr__)
    'open_hexdb',
    'GeoHexDB',
    'HexDBConfig',
    'HexDBRecipe',
]


# The hex database is geoworkflow's headline interface, exposed at the top level as
# `from geoworkflow import open_hexdb`. Imported lazily (PEP 562) so a plain
# `import geoworkflow` doesn't pull DuckDB/GeoPandas until the store is used.
_STORE_EXPORTS = {"open_hexdb", "GeoHexDB", "HexDBConfig", "HexDBRecipe"}


def __getattr__(name):
    if name in _STORE_EXPORTS:
        import importlib
        store = importlib.import_module("geoworkflow.store")
        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
