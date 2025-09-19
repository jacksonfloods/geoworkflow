# ---

# File: src/geoworkflow/utils/file_utils.py
"""
File and directory utilities for the geoworkflow package.

This module provides functions for file operations, temporary directory
management, and file system utilities.
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Union, Iterator
import logging

from geoworkflow.core.exceptions import GeoWorkflowError
from geoworkflow.core.constants import RASTER_EXTENSIONS, VECTOR_EXTENSIONS, ARCHIVE_EXTENSIONS

logger = logging.getLogger(__name__)


def create_temp_directory(prefix: str = "geoworkflow_") -> Path:
    """
    Create a temporary directory.
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Returns:
        Path to the created temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_dir: Path) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Path to temporary directory to remove
    """
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned up temporary directory: {temp_dir}")


def ensure_directory_exists(directory: Union[str, Path], create: bool = True) -> Path:
    """
    Ensure that a directory exists.
    
    Args:
        directory: Directory path
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Path object for the directory
        
    Raises:
        GeoWorkflowError: If directory doesn't exist and create=False
    """
    directory = Path(directory)
    
    if not directory.exists():
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        else:
            raise GeoWorkflowError(f"Directory does not exist: {directory}")
    
    return directory


def find_files_by_extension(
    directory: Union[str, Path], 
    extensions: Union[str, List[str]], 
    recursive: bool = True
) -> List[Path]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: File extension(s) to search for (with or without leading dot)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    # Normalize extensions
    if isinstance(extensions, str):
        extensions = [extensions]
    
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    extensions.extend([ext.upper() for ext in extensions])  # Add uppercase variants
    
    files = []
    search_method = directory.rglob if recursive else directory.glob
    
    for ext in extensions:
        pattern = f"*{ext}"
        files.extend(search_method(pattern))
    
    return sorted(list(set(files)))  # Remove duplicates and sort


def find_raster_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Find raster files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of raster file paths
    """
    extensions = [ext.lstrip('.') for ext in RASTER_EXTENSIONS]
    return find_files_by_extension(directory, extensions, recursive)


def find_vector_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Find vector files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of vector file paths
    """
    extensions = [ext.lstrip('.') for ext in VECTOR_EXTENSIONS]
    return find_files_by_extension(directory, extensions, recursive)


def find_archive_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Find archive files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of archive file paths
    """
    extensions = [ext.lstrip('.') for ext in ARCHIVE_EXTENSIONS]
    return find_files_by_extension(directory, extensions, recursive)


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def extract_archive(
    archive_path: Union[str, Path], 
    extract_dir: Union[str, Path],
    file_pattern: Optional[str] = None
) -> List[Path]:
    """
    Extract files from an archive.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract files to
        file_pattern: Optional pattern to filter extracted files
        
    Returns:
        List of extracted file paths
        
    Raises:
        GeoWorkflowError: If extraction fails
    """
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    
    if not archive_path.exists():
        raise GeoWorkflowError(f"Archive file does not exist: {archive_path}")
    
    ensure_directory_exists(extract_dir, create=True)
    
    try:
        if archive_path.suffix.lower() == '.zip':
            return _extract_zip(archive_path, extract_dir, file_pattern)
        else:
            raise GeoWorkflowError(f"Unsupported archive format: {archive_path.suffix}")
            
    except Exception as e:
        raise GeoWorkflowError(f"Failed to extract archive '{archive_path}': {str(e)}")


def _extract_zip(
    zip_path: Path, 
    extract_dir: Path, 
    file_pattern: Optional[str] = None
) -> List[Path]:
    """Extract files from a ZIP archive."""
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        
        # Filter files if pattern is provided
        if file_pattern:
            import fnmatch
            files_to_extract = [f for f in all_files if fnmatch.fnmatch(f, file_pattern)]
        else:
            files_to_extract = all_files
        
        for file_name in files_to_extract:
            # Skip directories
            if file_name.endswith('/'):
                continue
                
            zip_ref.extract(file_name, extract_dir)
            extracted_path = extract_dir / file_name
            extracted_files.append(extracted_path)
            logger.debug(f"Extracted: {file_name}")
    
    return extracted_files


def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> Path:
    """
    Copy a file to a new location.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        Path to the copied file
        
    Raises:
        GeoWorkflowError: If copy operation fails
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise GeoWorkflowError(f"Source file does not exist: {source}")
    
    # Ensure destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(source, destination)
        logger.debug(f"Copied file: {source} -> {destination}")
        return destination
    except Exception as e:
        raise GeoWorkflowError(f"Failed to copy file '{source}' to '{destination}': {str(e)}")


def move_file(source: Union[str, Path], destination: Union[str, Path]) -> Path:
    """
    Move a file to a new location.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        Path to the moved file
        
    Raises:
        GeoWorkflowError: If move operation fails
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise GeoWorkflowError(f"Source file does not exist: {source}")
    
    # Ensure destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(source), str(destination))
        logger.debug(f"Moved file: {source} -> {destination}")
        return destination
    except Exception as e:
        raise GeoWorkflowError(f"Failed to move file '{source}' to '{destination}': {str(e)}")

# ---

# File: src/geoworkflow/utils/spatial_utils.py
"""
Spatial utilities for the geoworkflow package.

This module provides functions for spatial operations, coordinate transformations,
and geospatial data analysis.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

try:
    import rasterio
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    import pyproj
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.exceptions import GeoWorkflowError
from geoworkflow.core.constants import DEFAULT_CRS

logger = logging.getLogger(__name__)


def get_raster_info(raster_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a raster file.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Dictionary with raster information
        
    Raises:
        GeoWorkflowError: If raster cannot be read
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise GeoWorkflowError("Rasterio not available for raster operations")
    
    raster_path = Path(raster_path)
    
    try:
        with rasterio.open(raster_path) as src:
            info = {
                "path": str(raster_path),
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtypes": [str(dtype) for dtype in src.dtypes],
                "crs": str(src.crs) if src.crs else None,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "compression": src.compression,
                "tiled": src.is_tiled,
                "blockxsize": src.block_shapes[0][1] if src.block_shapes else None,
                "blockysize": src.block_shapes[0][0] if src.block_shapes else None,
                "overviews": [src.overviews(i) for i in range(1, src.count + 1)],
            }
            
            # Calculate pixel size
            transform = src.transform
            info["pixel_size_x"] = abs(transform.a)
            info["pixel_size_y"] = abs(transform.e)
            
            # Calculate area in square kilometers (if CRS is geographic)
            if src.crs and src.crs.is_geographic:
                bounds = src.bounds
                # Rough calculation for geographic coordinates
                width_deg = bounds.right - bounds.left
                height_deg = bounds.top - bounds.bottom
                # Approximate area (not accurate for large areas)
                area_deg2 = width_deg * height_deg
                # Convert to km² (very rough approximation)
                area_km2 = area_deg2 * 111.32 * 111.32  # 1 degree ≈ 111.32 km
                info["approximate_area_km2"] = area_km2
            
            return info
            
    except Exception as e:
        raise GeoWorkflowError(f"Failed to read raster information from '{raster_path}': {str(e)}")


def get_vector_info(vector_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a vector file.
    
    Args:
        vector_path: Path to vector file
        
    Returns:
        Dictionary with vector information
        
    Raises:
        GeoWorkflowError: If vector cannot be read
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise GeoWorkflowError("GeoPandas not available for vector operations")
    
    vector_path = Path(vector_path)
    
    try:
        gdf = gpd.read_file(vector_path)
        
        info = {
            "path": str(vector_path),
            "feature_count": len(gdf),
            "columns": list(gdf.columns),
            "geometry_column": gdf.geometry.name,
            "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
        }
        
        # Calculate area if possible
        if not gdf.empty and gdf.crs:
            try:
                if gdf.crs.is_geographic:
                    # Reproject to equal-area projection for area calculation
                    gdf_projected = gdf.to_crs(DEFAULT_CRS)
                    total_area_m2 = gdf_projected.geometry.area.sum()
                    info["total_area_km2"] = total_area_m2 / 1_000_000
                else:
                    # Assume already in projected coordinates (meters)
                    total_area_m2 = gdf.geometry.area.sum()
                    info["total_area_km2"] = total_area_m2 / 1_000_000
            except Exception as e:
                logger.warning(f"Could not calculate area: {str(e)}")
                info["total_area_km2"] = None
        
        return info
        
    except Exception as e:
        raise GeoWorkflowError(f"Failed to read vector information from '{vector_path}': {str(e)}")


def calculate_bounds_intersection(bounds1: Tuple[float, float, float, float], 
                                bounds2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate the intersection of two bounding boxes.
    
    Args:
        bounds1: First bounding box (left, bottom, right, top)
        bounds2: Second bounding box (left, bottom, right, top)
        
    Returns:
        Intersection bounds or None if no intersection
    """
    left1, bottom1, right1, top1 = bounds1
    left2, bottom2, right2, top2 = bounds2
    
    # Calculate intersection
    left = max(left1, left2)
    bottom = max(bottom1, bottom2)
    right = min(right1, right2)
    top = min(top1, top2)
    
    # Check if there's a valid intersection
    if left >= right or bottom >= top:
        return None
    
    return (left, bottom, right, top)


def reproject_bounds(bounds: Tuple[float, float, float, float], 
                    src_crs: str, dst_crs: str) -> Tuple[float, float, float, float]:
    """
    Reproject bounding box coordinates.
    
    Args:
        bounds: Bounding box (left, bottom, right, top)
        src_crs: Source CRS
        dst_crs: Destination CRS
        
    Returns:
        Reprojected bounds
        
    Raises:
        GeoWorkflowError: If reprojection fails
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise GeoWorkflowError("PyProj not available for coordinate transformation")
    
    try:
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        
        left, bottom, right, top = bounds
        
        # Transform corner points
        x_coords = [left, right, left, right]
        y_coords = [bottom, bottom, top, top]
        
        transformed_x, transformed_y = transformer.transform(x_coords, y_coords)
        
        # Calculate new bounds
        new_left = min(transformed_x)
        new_right = max(transformed_x)
        new_bottom = min(transformed_y)
        new_top = max(transformed_y)
        
        return (new_left, new_bottom, new_right, new_top)
        
    except Exception as e:
        raise GeoWorkflowError(f"Failed to reproject bounds: {str(e)}")


def create_bounding_box_geometry(bounds: Tuple[float, float, float, float], crs: str = DEFAULT_CRS):
    """
    Create a bounding box geometry from bounds.
    
    Args:
        bounds: Bounding box (left, bottom, right, top)
        crs: Coordinate reference system
        
    Returns:
        GeoDataFrame with bounding box geometry
        
    Raises:
        GeoWorkflowError: If geometry creation fails
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise GeoWorkflowError("GeoPandas not available for geometry operations")
    
    try:
        left, bottom, right, top = bounds
        bbox_geom = box(left, bottom, right, top)
        
        gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=crs)
        return gdf
        
    except Exception as e:
        raise GeoWorkflowError(f"Failed to create bounding box geometry: {str(e)}")


def buffer_geometry(geometry_gdf: 'gpd.GeoDataFrame', buffer_distance_km: float) -> 'gpd.GeoDataFrame':
    """
    Buffer geometries by a specified distance in kilometers.
    
    For WGS84 coordinates, this uses a simple degree-based approximation.
    1 degree ≈ 111 km at the equator (less accurate at poles).
    
    Args:
        geometry_gdf: GeoDataFrame with geometries to buffer
        buffer_distance_km: Buffer distance in kilometers
        
    Returns:
        GeoDataFrame with buffered geometries
        
    Raises:
        GeoWorkflowError: If buffering fails
    """
    if not HAS_GEOSPATIAL_LIBS:
        raise GeoWorkflowError("GeoPandas not available for geometry operations")
    
    try:
        buffered_geom = geometry_gdf.copy()
        
        if geometry_gdf.crs and geometry_gdf.crs.is_geographic:
            # Simple degree-based buffer for geographic coordinates
            # 1 degree ≈ 111 km (rough approximation)
            buffer_distance_deg = buffer_distance_km / 111.0
            buffered_geom.geometry = geometry_gdf.geometry.buffer(buffer_distance_deg)
        else:
            # Assume projected coordinates in meters
            buffer_distance_m = buffer_distance_km * 1000
            buffered_geom.geometry = geometry_gdf.geometry.buffer(buffer_distance_m)
        
        return buffered_geom
        
    except Exception as e:
        raise GeoWorkflowError(f"Failed to buffer geometry: {str(e)}")


def calculate_pixel_size(transform) -> Tuple[float, float]:
    """
    Calculate pixel size from raster transform.
    
    Args:
        transform: Rasterio transform object
        
    Returns:
        Tuple of (x_size, y_size) in CRS units
    """
    x_size = abs(transform.a)
    y_size = abs(transform.e)
    return x_size, y_size


def align_bounds_to_grid(bounds: Tuple[float, float, float, float], 
                        pixel_size: Tuple[float, float],
                        origin: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """
    Align bounds to a pixel grid.
    
    Args:
        bounds: Bounding box (left, bottom, right, top)
        pixel_size: Pixel size (x_size, y_size)
        origin: Grid origin (x_origin, y_origin)
        
    Returns:
        Grid-aligned bounds
    """
    left, bottom, right, top = bounds
    x_size, y_size = pixel_size
    x_origin, y_origin = origin
    
    # Calculate grid-aligned bounds
    aligned_left = x_origin + round((left - x_origin) / x_size) * x_size
    aligned_right = x_origin + round((right - x_origin) / x_size) * x_size
    aligned_bottom = y_origin + round((bottom - y_origin) / y_size) * y_size
    aligned_top = y_origin + round((top - y_origin) / y_size) * y_size
    
    return (aligned_left, aligned_bottom, aligned_right, aligned_top)