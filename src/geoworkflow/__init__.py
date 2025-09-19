"""
GeoWorkflow - Comprehensive geospatial data processing for African analysis.

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
]
