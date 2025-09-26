#/scr/geoworkflow/utils/earth_engine_error_handler.py
"""
Earth Engine error handling utilities.

This module provides academic-friendly error handling for Earth Engine operations,
with specific guidance for common authentication, quota, and processing issues.
"""

import logging
from typing import Dict, Any, Optional, Type
from pathlib import Path

from geoworkflow.core.exceptions import (
    EarthEngineError,
    EarthEngineAuthenticationError, 
    EarthEngineQuotaError,
    EarthEngineTimeoutError,
    EarthEngineGeometryError,
    EarthEngineExportError,
    get_academic_friendly_error_message
)

logger = logging.getLogger(__name__)


class EarthEngineErrorHandler:
    """Handle Earth Engine errors with academic-friendly messaging."""
    
    @staticmethod
    def handle_authentication_error(error: Exception, config: Dict[str, Any]) -> EarthEngineAuthenticationError:
        """Handle authentication-related errors."""
        friendly_message = get_academic_friendly_error_message(error)
        
        # Add specific guidance based on configuration
        if not config.get('service_account_key') and not config.get('project_id'):
            friendly_message += (
                "\n\nConfiguration suggestions:\n"
                "- Add service_account_key: '/path/to/key.json' to your config\n"
                "- Or set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "- Or add project_id for user credential authentication"
            )
        
        return EarthEngineAuthenticationError(friendly_message)
    
    @staticmethod
    def handle_quota_error(error: Exception, config: Dict[str, Any]) -> EarthEngineQuotaError:
        """Handle quota-related errors with optimization suggestions."""
        friendly_message = get_academic_friendly_error_message(error)
        
        # Add specific optimization suggestions
        optimization_tips = []
        
        if config.get('confidence_threshold', 0.75) < 0.8:
            optimization_tips.append(f"Increase confidence_threshold from {config.get('confidence_threshold')} to 0.8+")
            
        if not config.get('max_features'):
            optimization_tips.append("Set max_features (e.g., 10000) to limit results")
            
        if config.get('min_area_m2', 10) < 25:
            optimization_tips.append(f"Increase min_area_m2 from {config.get('min_area_m2')} to 25+")
        
        if optimization_tips:
            friendly_message += f"\n\nOptimization suggestions:\n" + "\n".join(f"- {tip}" for tip in optimization_tips)
        
        return EarthEngineQuotaError(friendly_message)
    
    @staticmethod
    def handle_timeout_error(error: Exception, config: Dict[str, Any]) -> EarthEngineTimeoutError:
        """Handle timeout-related errors."""
        friendly_message = get_academic_friendly_error_message(error)
        
        # Suggest chunking for large areas
        if not config.get('max_features'):
            friendly_message += (
                "\n\nFor large areas, try:\n"
                "- Set max_features to 5000-10000\n"
                "- Process in smaller geographic chunks\n" 
                "- Increase confidence_threshold to reduce data volume"
            )
        
        return EarthEngineTimeoutError(friendly_message)
    
    @staticmethod
    def handle_geometry_error(error: Exception, aoi_file: Optional[Path]) -> EarthEngineGeometryError:
        """Handle geometry-related errors."""
        friendly_message = get_academic_friendly_error_message(error)
        
        if aoi_file:
            friendly_message += (
                f"\n\nAOI file troubleshooting for {aoi_file}:\n"
                "- Check geometry is valid (no self-intersections)\n"
                "- Try simplifying complex boundaries\n"
                "- Ensure coordinate system is WGS84 (EPSG:4326)\n"
                "- Verify file is not corrupted"
            )
        
        return EarthEngineGeometryError(friendly_message)
    
    @staticmethod
    def classify_and_handle_error(error: Exception, config: Dict[str, Any]) -> EarthEngineError:
        """
        Classify Earth Engine error and return appropriate exception type.
        
        Args:
            error: Original Earth Engine exception
            config: Processing configuration for context
            
        Returns:
            Appropriate Earth Engine exception with academic guidance
        """
        error_str = str(error).lower()
        
        # Authentication errors
        if any(pattern in error_str for pattern in ['authentication', 'unauthorized', 'credentials']):
            return EarthEngineErrorHandler.handle_authentication_error(error, config)
        
        # Quota errors  
        elif any(pattern in error_str for pattern in ['quota', 'limit', 'rate limit']):
            return EarthEngineErrorHandler.handle_quota_error(error, config)
        
        # Timeout errors
        elif any(pattern in error_str for pattern in ['timeout', 'deadline', 'cancelled']):
            return EarthEngineErrorHandler.handle_timeout_error(error, config)
        
        # Geometry errors
        elif any(pattern in error_str for pattern in ['geometry', 'self-intersection', 'invalid']):
            return EarthEngineErrorHandler.handle_geometry_error(error, config.get('aoi_file'))
        
        # Export errors
        elif any(pattern in error_str for pattern in ['export', 'download', 'task']):
            return EarthEngineExportError(get_academic_friendly_error_message(error))
        
        # Generic Earth Engine error
        else:
            return EarthEngineError(get_academic_friendly_error_message(error))


def validate_earth_engine_prerequisites() -> Dict[str, Any]:
    """
    Validate Earth Engine setup and return validation results.
    
    Returns:
        Dictionary with validation status and guidance
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "setup_guidance": []
    }
    
    # Check if Earth Engine is available
    try:
        import ee
        validation_result["ee_available"] = True
    except ImportError:
        validation_result["valid"] = False
        validation_result["errors"].append("Earth Engine API not installed")
        validation_result["setup_guidance"].append(
            "Install Earth Engine: pip install earthengine-api google-auth google-cloud-storage"
        )
        return validation_result
    
    # Check authentication
    try:
        ee.Initialize()
        validation_result["authenticated"] = True
    except Exception as e:
        validation_result["authenticated"] = False
        validation_result["warnings"].append(f"Authentication issue: {e}")
        validation_result["setup_guidance"].extend([
            "Get Earth Engine access: https://earthengine.google.com/signup/",
            "Set up authentication: earthengine authenticate",
            "Or use service account key file"
        ])
    
    return validation_result