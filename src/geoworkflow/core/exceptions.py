"""
Custom exceptions for the geoworkflow package.

This module defines the exception hierarchy used throughout the geoworkflow
package to provide clear error handling and debugging information.
"""

from typing import Optional, Any, Dict
import traceback


class GeoWorkflowError(Exception):
    """Base exception for all geoworkflow errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(GeoWorkflowError):
    """Raised when there are configuration validation errors."""
    pass


class ProcessingError(GeoWorkflowError):
    """Raised when data processing operations fail."""
    pass


class ValidationError(GeoWorkflowError):
    """Raised when data validation fails."""
    pass


class FileOperationError(GeoWorkflowError):
    """Raised when file operations fail."""
    pass


class GeospatialError(GeoWorkflowError):
    """Raised when geospatial operations fail."""
    pass


class AlignmentError(GeospatialError):
    """Raised when raster alignment operations fail."""
    pass


class ClippingError(GeospatialError):
    """Raised when spatial clipping operations fail."""
    pass


class ExtractionError(ProcessingError):
    """Raised when data extraction operations fail."""
    pass


class VisualizationError(GeoWorkflowError):
    """Raised when visualization operations fail."""
    pass

class ClippingError(GeospatialError):
    """Raised when spatial clipping operations fail."""
    pass


class PipelineError(GeoWorkflowError):
    """Raised when pipeline execution fails."""
    
    def __init__(self, message: str, stage: Optional[str] = None, 
                 failed_processor: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.stage = stage
        self.failed_processor = failed_processor


def handle_processing_error(func):
    """
    Decorator to handle common processing errors and convert them to GeoWorkflowError.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GeoWorkflowError:
            # Re-raise our own exceptions
            raise
        except FileNotFoundError as e:
            raise FileOperationError(f"File not found: {e}")
        except PermissionError as e:
            raise FileOperationError(f"Permission denied: {e}")
        except ValueError as e:
            raise ValidationError(f"Invalid value: {e}")
        except Exception as e:
            # Generic error with traceback
            tb = traceback.format_exc()
            raise ProcessingError(
                f"Unexpected error in {func.__name__}: {e}",
                details={'traceback': tb}
            )
    return wrapper
