# File: src/geoworkflow/utils/validation.py
"""
Data validation utilities for the geoworkflow package.

This module provides functions for validating geospatial data formats,
coordinate reference systems, and data integrity.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

try:
    import rasterio
    import geopandas as gpd
    import pyproj
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.exceptions import ValidationError
from geoworkflow.core.constants import RASTER_EXTENSIONS, VECTOR_EXTENSIONS


logger = logging.getLogger(__name__)


def validate_file_format(file_path: Union[str, Path], expected_type: str) -> bool:
    """
    Validate that a file has the expected format.
    
    Args:
        file_path: Path to the file
        expected_type: Expected file type ('raster' or 'vector')
        
    Returns:
        True if file format is valid
        
    Raises:
        ValidationError: If file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    extension = file_path.suffix.lower()
    
    if expected_type == "raster":
        if extension not in RASTER_EXTENSIONS:
            raise ValidationError(
                f"Invalid raster file format: {extension}. "
                f"Expected one of: {', '.join(RASTER_EXTENSIONS)}"
            )
    elif expected_type == "vector":
        if extension not in VECTOR_EXTENSIONS:
            raise ValidationError(
                f"Invalid vector file format: {extension}. "
                f"Expected one of: {', '.join(VECTOR_EXTENSIONS)}"
            )
    else:
        raise ValidationError(f"Unknown expected type: {expected_type}")
    
    return True


def validate_crs(crs_string: str) -> bool:
    """
    Validate a coordinate reference system string.
    
    Args:
        crs_string: CRS string (e.g., "EPSG:4326", "ESRI:102022")
        
    Returns:
        True if CRS is valid
        
    Raises:
        ValidationError: If CRS is invalid
    """
    if not HAS_GEOSPATIAL_LIBS:
        logger.warning("Geospatial libraries not available, skipping CRS validation")
        return True
    
    try:
        # Try to create a CRS object
        crs = pyproj.CRS(crs_string)
        if not crs.is_valid:
            raise ValidationError(f"Invalid CRS: {crs_string}")
        return True
    except Exception as e:
        raise ValidationError(f"Invalid CRS '{crs_string}': {str(e)}")


def validate_raster_file(file_path: Union[str, Path], 
                        detect_nodata: bool = True,
                        force_redetection: bool = False) -> Dict[str, Any]:
    """
    Validate a raster file and return metadata with enhanced nodata detection.
    
    Args:
        file_path: Path to raster file
        detect_nodata: Whether to automatically detect and cache nodata values
        force_redetection: Force re-detection even if metadata exists
        
    Returns:
        Dictionary with raster metadata including enhanced nodata information
        
    Raises:
        ValidationError: If raster file is invalid
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise ValidationError("Rasterio not available for raster validation")
    
    file_path = Path(file_path)
    validate_file_format(file_path, "raster")
    
    try:
        with rasterio.open(file_path) as src:
            metadata = {
                "path": str(file_path),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else None,
                "bounds": src.bounds,
                "transform": src.transform,
                "nodata": src.nodata,
            }
            
            # Validate that the raster has spatial information
            if src.crs is None:
                logger.warning(f"Raster has no CRS defined: {file_path}")
            
            # Enhanced nodata detection and caching
            if detect_nodata:
                # Import here to avoid circular imports
                from ..utils.raster_utils import detect_and_update_nodata, _get_cached_nodata_detection
                
                # Check if we have cached results first
                cached_nodata = _get_cached_nodata_detection(file_path)
                detection_method = "cached" if cached_nodata is not None and not force_redetection else "computed"
                
                # Run detection with caching
                detected_nodata = detect_and_update_nodata(
                    file_path, 
                    force_redetection=force_redetection,
                    update_metadata=True
                )
                
                # Add enhanced nodata information to metadata
                metadata["enhanced_nodata"] = {
                    "original_nodata": src.nodata,
                    "detected_nodata": detected_nodata,
                    "detection_method": detection_method,
                    "detection_successful": detected_nodata is not None,
                    "metadata_updated": detected_nodata is not None and detection_method == "computed"
                }
                
                # Log results
                if detected_nodata is not None:
                    if detection_method == "cached":
                        logger.debug(f"Using cached nodata value {detected_nodata} for {file_path.name}")
                    else:
                        logger.info(f"Detected and cached nodata value {detected_nodata} for {file_path.name}")
                else:
                    logger.debug(f"No nodata value detected for {file_path.name}")
            
            return metadata
            
    except Exception as e:
        raise ValidationError(f"Failed to validate raster file '{file_path}': {str(e)}")


def validate_vector_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a vector file and return metadata.
    
    Args:
        file_path: Path to vector file
        
    Returns:
        Dictionary with vector metadata
        
    Raises:
        ValidationError: If vector file is invalid
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise ValidationError("GeoPandas not available for vector validation")
    
    file_path = Path(file_path)
    validate_file_format(file_path, "vector")
    
    try:
        gdf = gpd.read_file(file_path)
        
        metadata = {
            "path": str(file_path),
            "feature_count": len(gdf),
            "columns": list(gdf.columns),
            "geometry_types": gdf.geometry.geom_type.unique().tolist(),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
        }
        
        # Validate that the vector has spatial information
        if gdf.crs is None:
            logger.warning(f"Vector has no CRS defined: {file_path}")
        
        if gdf.empty:
            logger.warning(f"Vector file is empty: {file_path}")
        
        return metadata
        
    except Exception as e:
        raise ValidationError(f"Failed to validate vector file '{file_path}': {str(e)}")


def validate_buffer_distance(buffer_km: float) -> bool:
    """
    Validate buffer distance parameter.
    
    Args:
        buffer_km: Buffer distance in kilometers
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If buffer distance is invalid
    """
    if not isinstance(buffer_km, (int, float)):
        raise ValidationError("Buffer distance must be a number")
    
    if buffer_km < 0:
        raise ValidationError("Buffer distance must be non-negative")
    
    if buffer_km > 1000:  # Reasonable upper limit
        raise ValidationError("Buffer distance must be less than 1000 km")
    
    return True


def validate_coordinate_pair(lat: float, lon: float) -> bool:
    """
    Validate latitude/longitude coordinate pair.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationError("Coordinates must be numeric")
    
    if not (-90 <= lat <= 90):
        raise ValidationError(f"Latitude {lat} must be between -90 and 90")
    
    if not (-180 <= lon <= 180):
        raise ValidationError(f"Longitude {lon} must be between -180 and 180")
    
    return True


def validate_epsg_code(epsg_code: int) -> bool:
    """
    Validate EPSG code.
    
    Args:
        epsg_code: EPSG code to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If EPSG code is invalid
    """
    if not isinstance(epsg_code, int):
        raise ValidationError("EPSG code must be an integer")
    
    if epsg_code < 1000 or epsg_code > 99999:
        raise ValidationError("EPSG code must be between 1000 and 99999")
    
    # Try to validate with pyproj if available
    if HAS_GEOSPATIAL_LIBS:
        try:
            crs = pyproj.CRS.from_epsg(epsg_code)
            if not crs.is_valid:
                raise ValidationError(f"Invalid EPSG code: {epsg_code}")
        except Exception:
            raise ValidationError(f"Unknown EPSG code: {epsg_code}")
    
    return True