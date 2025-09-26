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

class EarthEngineError(GeoWorkflowError):
    """Base exception for Earth Engine operations."""
    pass


class EarthEngineAuthenticationError(EarthEngineError):
    """Earth Engine authentication failed."""
    pass


class EarthEngineQuotaError(EarthEngineError):
    """Earth Engine quota exceeded."""
    pass


class EarthEngineTimeoutError(EarthEngineError):
    """Earth Engine operation timed out."""
    pass


class EarthEngineGeometryError(EarthEngineError):
    """Invalid geometry for Earth Engine operations."""
    pass

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

class EarthEngineError(GeoWorkflowError):
    """Base exception for Earth Engine operations."""
    pass

class EarthEngineAuthenticationError(EarthEngineError):
    """Earth Engine authentication failed."""
    pass

class EarthEngineQuotaError(EarthEngineError):
    """Earth Engine quota exceeded."""
    pass

class EarthEngineTimeoutError(EarthEngineError):
    """Earth Engine operation timed out."""
    pass

class EarthEngineGeometryError(EarthEngineError):
    """Earth Engine geometry operation failed."""
    pass

class EarthEngineExportError(EarthEngineError):
    """Earth Engine export operation failed."""
    pass

def classify_earth_engine_error(error_message: str) -> str:
    """
    Classify Earth Engine error and provide category.
    
    Args:
        error_message: Error message from Earth Engine
        
    Returns:
        Error category string
    """
    error_lower = error_message.lower()
    
    for category, patterns in EE_ERROR_PATTERNS.items():
        if any(pattern in error_lower for pattern in patterns):
            return category
    
    return 'unknown'

def get_academic_friendly_error_message(error: Exception) -> str:
    """
    Convert technical Earth Engine error to academic-friendly message.
    
    Args:
        error: Original Earth Engine exception
        
    Returns:
        User-friendly error message with guidance
    """
    error_message = str(error).lower()
    error_category = classify_earth_engine_error(error_message)
    
    if error_category == 'authentication':
        return (
            "Earth Engine authentication failed. For academic access:\n"
            "1. Sign up at: https://earthengine.google.com/signup/\n"
            "2. Create service account: https://developers.google.com/earth-engine/guides/service_account\n"
            "3. Set service_account_key in config or GOOGLE_APPLICATION_CREDENTIALS env var\n"
            f"Technical details: {error}"
        )
    elif error_category == 'quota':
        return (
            "Earth Engine quota exceeded. Try:\n"
            "1. Reduce AOI size or increase confidence_threshold\n"
            "2. Set max_features to limit results\n"
            "3. Wait and retry (quotas reset daily)\n"
            f"Technical details: {error}"
        )
    elif error_category == 'timeout':
        return (
            "Earth Engine operation timed out. Try:\n"
            "1. Reduce the area of interest\n"
            "2. Increase confidence threshold to get fewer buildings\n"
            "3. Set max_features to limit results\n"
            f"Technical details: {error}"
        )
    elif error_category == 'geometry':
        return (
            "Geometry processing failed. Try:\n"
            "1. Simplify the AOI geometry\n"
            "2. Ensure geometry is valid (no self-intersections)\n"
            "3. Use a smaller AOI\n"
            f"Technical details: {error}"
        )
    else:
        return (
            "Earth Engine operation failed.\n"
            "Common solutions:\n"
            "1. Check your internet connection\n"
            "2. Verify AOI file is valid\n"
            "3. Try with a smaller area\n"
            f"Technical details: {error}"
        )

# ============================================================================
# Retry Decorator for Earth Engine Operations
# ============================================================================

def retry_earth_engine_operation(max_attempts: int = 3, delay_seconds: float = 5.0):
    """
    Decorator to retry Earth Engine operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_category = classify_earth_engine_error(str(e))
                    
                    # Don't retry authentication errors
                    if error_category == 'authentication':
                        raise
                    
                    # Don't retry on final attempt
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Earth Engine operation failed (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying in {delay} seconds: {e}"
                    )
                    time.sleep(delay)
            
        return wrapper
    return decorator

class EarthEngineError(GeoWorkflowError):
    """Base exception for Earth Engine operations."""
    pass


class EarthEngineAuthenticationError(EarthEngineError):
    """Earth Engine authentication failed."""
    pass


class EarthEngineQuotaError(EarthEngineError):
    """Earth Engine quota exceeded."""
    pass


class EarthEngineTimeoutError(EarthEngineError):
    """Earth Engine operation timed out."""
    pass


class EarthEngineGeometryError(EarthEngineError):
    """Invalid geometry for Earth Engine operations."""
    pass


class EarthEngineExportError(EarthEngineError):
    """Earth Engine export operation failed."""
    pass


# Error classification patterns for academic-friendly messages
EE_ERROR_PATTERNS = {
    'authentication': ['authentication', 'unauthorized', 'credentials', 'permission denied'],
    'quota': ['quota', 'limit', 'rate limit', 'too many requests', 'usage limit'],
    'timeout': ['timeout', 'deadline', 'cancelled', 'timed out'],
    'geometry': ['invalid geometry', 'self-intersection', 'too complex', 'geometry error']
}


def classify_earth_engine_error(error_message: str) -> str:
    """
    Classify Earth Engine error and provide category.
    
    Args:
        error_message: Error message from Earth Engine
        
    Returns:
        Error category string
    """
    error_lower = error_message.lower()
    
    for category, patterns in EE_ERROR_PATTERNS.items():
        if any(pattern in error_lower for pattern in patterns):
            return category
    
    return 'unknown'


def get_academic_friendly_error_message(error: Exception) -> str:
    """
    Convert technical Earth Engine error to academic-friendly message.
    
    Args:
        error: Original Earth Engine exception
        
    Returns:
        User-friendly error message with guidance
    """
    error_message = str(error).lower()
    error_category = classify_earth_engine_error(error_message)
    
    if error_category == 'authentication':
        return (
            "Earth Engine authentication failed. For academic access:\n"
            "1. Sign up at: https://earthengine.google.com/signup/\n"
            "2. Create service account: https://developers.google.com/earth-engine/guides/service_account\n"
            "3. Set service_account_key in config or GOOGLE_APPLICATION_CREDENTIALS env var\n"
            f"Technical details: {error}"
        )
    elif error_category == 'quota':
        return (
            "Earth Engine quota exceeded. Try:\n"
            "1. Reduce AOI size or increase confidence_threshold\n"
            "2. Set max_features to limit results\n"
            "3. Wait and retry (quotas reset daily)\n"
            f"Technical details: {error}"
        )
    elif error_category == 'timeout':
        return (
            "Earth Engine operation timed out. Try:\n"
            "1. Reduce the area of interest\n"
            "2. Increase confidence threshold to get fewer buildings\n"
            "3. Set max_features to limit results\n"
            f"Technical details: {error}"
        )
    elif error_category == 'geometry':
        return (
            "Geometry processing failed. Try:\n"
            "1. Simplify the AOI geometry\n"
            "2. Ensure geometry is valid (no self-intersections)\n"
            "3. Use a smaller AOI\n"
            f"Technical details: {error}"
        )
    else:
        return (
            "Earth Engine operation failed.\n"
            "Common solutions:\n"
            "1. Check your internet connection\n"
            "2. Verify AOI file is valid\n"
            "3. Try with a smaller area\n"
            f"Technical details: {error}"
        )


# Retry decorator for Earth Engine operations
def retry_earth_engine_operation(max_attempts: int = 3, delay_seconds: float = 5.0):
    """
    Decorator to retry Earth Engine operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
    """
    import time
    import logging
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_category = classify_earth_engine_error(str(e))
                    
                    # Don't retry authentication errors
                    if error_category == 'authentication':
                        raise
                    
                    # Don't retry on final attempt
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Earth Engine operation failed (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying in {delay} seconds: {e}"
                    )
                    time.sleep(delay)
            
        return wrapper
    return decorator