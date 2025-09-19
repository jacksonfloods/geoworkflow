# File: src/geoworkflow/utils/raster_utils.py
"""
Raster utilities for the geoworkflow package.

This module provides utilities for raster data operations including:
- Automatic nodata value detection
- Nodata mask application and validation
- Raster data quality assessment
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
import warnings

try:
    import numpy as np
    import rasterio
    HAS_RASTER_LIBS = True
except ImportError:
    HAS_RASTER_LIBS = False

from ..core.exceptions import GeoWorkflowError

GEOWORKFLOW_NODATA_DETECTED = "GEOWORKFLOW_NODATA_DETECTED"
GEOWORKFLOW_NODATA_VALUE = "GEOWORKFLOW_NODATA_VALUE" 
GEOWORKFLOW_NODATA_METHOD = "GEOWORKFLOW_NODATA_METHOD"
GEOWORKFLOW_NODATA_CONFIDENCE = "GEOWORKFLOW_NODATA_CONFIDENCE"
GEOWORKFLOW_NODATA_TIMESTAMP = "GEOWORKFLOW_NODATA_TIMESTAMP"

logger = logging.getLogger(__name__)

# Enhanced COMMON_NODATA_VALUES with real-world usage patterns
COMMON_NODATA_VALUES = [
    # Standard negative sentinels (most common)
    -9999, -999, -99, -9, -1,
    
    # Integer type extremes and near-extremes
    -32768, -32767,  # Int16 min and near-min
    32767,           # Int16 max
    65535,           # UInt16 max  
    255,             # UInt8 max
    -2147483648, -2147483647,  # Int32 min and near-min
    2147483647,      # Int32 max
    4294967295,      # UInt32 max
    
    # Float32 extremes (IEEE 754)
    -3.4028235e+38, 3.4028235e+38,     # Float32 min/max
    -3.4e+38, 3.4e+38,                 # Common shortened versions
    
    # Float64 extremes
    -1.7976931348623157e+308,          # Float64 min
    
    # Special satellite/remote sensing values
    0,              # Common for masked areas
    -999999,        # Extended precision version
    
    # SAGA GIS defaults
    -99999,
    
    # NASA/Earth science common values
    -32766, -32765  # Near Int16 min values
]

# Thresholds for statistical outlier detection
DEFAULT_Z_THRESHOLD = 4.0  # Values beyond 4 standard deviations
DEFAULT_PERCENTILE_THRESHOLD = 0.001  # 0.1% extreme values


def detect_nodata_value(
    raster_path: Union[str, Path], 
    band: int = 1,
    method: str = "auto",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    sample_size: Optional[int] = None
) -> Optional[float]:
    """
    Automatically detect the nodata value for a raster band.
    
    This function uses multiple approaches to detect potential nodata values:
    1. Check if nodata is already properly set in metadata
    2. Look for common nodata sentinel values
    3. Use statistical outlier detection for extreme values
    4. Validate spatial coherence of detected values
    
    Args:
        raster_path: Path to the raster file
        band: Band number to analyze (1-based indexing)
        method: Detection method ('auto', 'statistical', 'common_values', 'metadata_only')
        z_threshold: Z-score threshold for statistical outlier detection
        percentile_threshold: Percentile threshold for extreme value detection
        sample_size: Optional sample size for large rasters (None = use full raster)
        
    Returns:
        Detected nodata value, or None if no reliable nodata value found
        
    Raises:
        GeoWorkflowError: If raster cannot be read or analysis fails
    """
    if not HAS_RASTER_LIBS:
        raise GeoWorkflowError("Rasterio and numpy are required for nodata detection")
    
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise GeoWorkflowError(f"Raster file not found: {raster_path}")
    
    try:
        with rasterio.open(raster_path) as src:
            
            # Method 1: Check existing metadata first
            if method in ["auto", "metadata_only"] and src.nodata is not None:
                logger.debug(f"Using existing nodata value from metadata: {src.nodata}")
                return float(src.nodata)
            
            if method == "metadata_only":
                logger.debug("No nodata value found in metadata")
                return None
            
            # Read data for analysis
            if sample_size and src.width * src.height > sample_size:
                # Sample the raster for large files
                data = _sample_raster_data(src, band, sample_size)
                logger.debug(f"Using sample of {len(data)} pixels for analysis")
            else:
                # Read full raster
                data = src.read(band)
                logger.debug(f"Analyzing {data.size} pixels")
            
            # Flatten for analysis
            data_flat = data.flatten()
            
            # Remove any existing NaN values from analysis
            valid_mask = ~np.isnan(data_flat.astype(float))
            if not valid_mask.any():
                logger.warning("All pixels are NaN - cannot detect nodata value")
                return None
            
            data_clean = data_flat[valid_mask]
            
            # Method 2: Check for common nodata values
            if method in ["auto", "common_values"]:
                detected_common = _detect_common_nodata_values(data_clean, src.dtypes[band-1])
                if detected_common is not None:
                    logger.debug(f"Detected common nodata value: {detected_common}")
                    return detected_common
            
            # Method 3: Statistical outlier detection
            if method in ["auto", "statistical"]:
                detected_statistical = _detect_statistical_outliers(
                    data_clean, z_threshold, percentile_threshold, src.dtypes[band-1]
                )
                if detected_statistical is not None:
                    logger.debug(f"Detected statistical outlier as nodata: {detected_statistical}")
                    return detected_statistical
            
            logger.debug("No reliable nodata value detected")
            return None
            
    except Exception as e:
        raise GeoWorkflowError(f"Failed to detect nodata value for {raster_path}: {str(e)}")


def _sample_raster_data(src, band: int, sample_size: int) -> np.ndarray:
    """Sample raster data for large files."""
    height, width = src.height, src.width
    total_pixels = height * width
    
    if sample_size >= total_pixels:
        return src.read(band)
    
    # Calculate sampling strategy
    sample_ratio = np.sqrt(sample_size / total_pixels)
    sample_height = max(1, int(height * sample_ratio))
    sample_width = max(1, int(width * sample_ratio))
    
    # Read a representative sample from the center and edges
    data_samples = []
    
    # Center sample
    row_start = (height - sample_height) // 2
    col_start = (width - sample_width) // 2
    window = rasterio.windows.Window(col_start, row_start, sample_width, sample_height)
    data_samples.append(src.read(band, window=window))
    
    # Corner samples for edge effects
    corners = [
        (0, 0),  # top-left
        (0, max(0, width - sample_width)),  # top-right
        (max(0, height - sample_height), 0),  # bottom-left
        (max(0, height - sample_height), max(0, width - sample_width))  # bottom-right
    ]
    
    for row, col in corners:
        if len(np.concatenate(data_samples).flatten()) < sample_size:
            window = rasterio.windows.Window(col, row, 
                                           min(sample_width, width - col),
                                           min(sample_height, height - row))
            data_samples.append(src.read(band, window=window))
    
    return np.concatenate([d.flatten() for d in data_samples])


def _detect_common_nodata_values(data: np.ndarray, dtype) -> Optional[float]:
    """Detect common nodata sentinel values."""
    
    # Get dtype-appropriate common values
    dtype_str = str(dtype)
    relevant_values = []
    
    for val in COMMON_NODATA_VALUES:
        try:
            # Check if value can be represented in this dtype
            typed_val = np.array([val], dtype=dtype)[0]
            if typed_val == val:  # No precision loss
                relevant_values.append(val)
        except (OverflowError, ValueError):
            continue
    
    # Check for exact matches with significant frequency
    unique_vals, counts = np.unique(data, return_counts=True)
    total_pixels = len(data)
    
    for val in relevant_values:
        if val in unique_vals:
            val_count = counts[unique_vals == val][0]
            frequency = val_count / total_pixels
            
            # If a common nodata value appears in >0.1% of pixels, it's likely nodata
            if frequency > 0.001:
                logger.debug(f"Common nodata value {val} found in {frequency:.1%} of pixels")
                return float(val)
    
    return None


def _detect_statistical_outliers(
    data: np.ndarray, 
    z_threshold: float, 
    percentile_threshold: float,
    dtype
) -> Optional[float]:
    """Detect nodata values using statistical outlier analysis."""
    
    if len(data) < 100:  # Need reasonable sample size
        return None
    
    # Calculate basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:  # Constant raster
        return None
    
    # Method 1: Z-score based detection for extreme outliers
    z_scores = np.abs((data - mean_val) / std_val)
    extreme_outliers = data[z_scores > z_threshold]
    
    if len(extreme_outliers) > 0:
        # Check if outliers cluster around specific values
        outlier_candidate = _find_clustered_outlier(extreme_outliers, data)
        if outlier_candidate is not None:
            return outlier_candidate
    
    # Method 2: Percentile-based detection
    lower_percentile = percentile_threshold * 100
    upper_percentile = (1 - percentile_threshold) * 100
    
    lower_threshold = np.percentile(data, lower_percentile)
    upper_threshold = np.percentile(data, upper_percentile)
    
    # Check for large gaps that might indicate nodata
    data_sorted = np.sort(data)
    
    # Look for large gaps at the extremes
    if len(data_sorted) > 10:
        # Check bottom extreme
        bottom_gap = data_sorted[int(len(data_sorted) * 0.01)] - data_sorted[0]
        normal_range = data_sorted[-1] - data_sorted[0]
        
        if bottom_gap > 0.1 * normal_range and data_sorted[0] < lower_threshold:
            # Large gap at bottom suggests nodata
            unique_bottom = np.unique(data_sorted[:int(len(data_sorted) * 0.01)])
            if len(unique_bottom) == 1:  # All bottom values are the same
                return float(unique_bottom[0])
        
        # Check top extreme  
        top_gap = data_sorted[-1] - data_sorted[int(len(data_sorted) * 0.99)]
        
        if top_gap > 0.1 * normal_range and data_sorted[-1] > upper_threshold:
            # Large gap at top suggests nodata
            unique_top = np.unique(data_sorted[int(len(data_sorted) * 0.99):])
            if len(unique_top) == 1:  # All top values are the same
                return float(unique_top[0])
    
    return None


def _find_clustered_outlier(outliers: np.ndarray, all_data: np.ndarray) -> Optional[float]:
    """Find if outliers cluster around a specific value that could be nodata."""
    
    unique_outliers, counts = np.unique(outliers, return_counts=True)
    total_outliers = len(outliers)
    total_data = len(all_data)
    
    # Look for a single value that represents most outliers
    for val, count in zip(unique_outliers, counts):
        outlier_dominance = count / total_outliers
        data_frequency = count / total_data
        
        # If one value represents >50% of outliers and >0.1% of data
        if outlier_dominance > 0.5 and data_frequency > 0.001:
            logger.debug(f"Clustered outlier value {val}: {outlier_dominance:.1%} of outliers, "
                        f"{data_frequency:.1%} of data")
            return float(val)
    
    return None


def apply_nodata_mask(
    data: np.ndarray, 
    nodata_value: float, 
    tolerance: float = 1e-6
) -> np.ma.MaskedArray:
    """
    Apply nodata masking to a numpy array.
    
    Args:
        data: Input numpy array
        nodata_value: Value to mask as nodata
        tolerance: Tolerance for floating-point comparison
        
    Returns:
        Masked array with nodata values masked
    """
    if np.isnan(nodata_value):
        mask = np.isnan(data.astype(float))
    else:
        if np.issubdtype(data.dtype, np.floating):
            mask = np.abs(data - nodata_value) <= tolerance
        else:
            mask = data == nodata_value
    
    return np.ma.masked_array(data, mask=mask)


def validate_nodata_detection(
    raster_path: Union[str, Path],
    detected_value: float,
    band: int = 1,
    spatial_coherence_check: bool = True
) -> Dict[str, Any]:
    """
    Validate a detected nodata value by checking spatial and statistical properties.
    
    Args:
        raster_path: Path to raster file
        detected_value: The detected nodata value to validate
        band: Band number to check
        spatial_coherence_check: Whether to check spatial clustering of nodata pixels
        
    Returns:
        Dictionary with validation results and metrics
    """
    if not HAS_RASTER_LIBS:
        raise GeoWorkflowError("Rasterio required for validation")
    
    validation_results = {
        "is_valid": False,
        "confidence": 0.0,
        "metrics": {},
        "warnings": []
    }
    
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(band)
            
            # Calculate basic metrics
            mask = (data == detected_value)
            nodata_count = np.sum(mask)
            total_pixels = data.size
            nodata_percentage = (nodata_count / total_pixels) * 100
            
            validation_results["metrics"] = {
                "nodata_pixel_count": int(nodata_count),
                "total_pixels": int(total_pixels),
                "nodata_percentage": float(nodata_percentage),
                "detected_value": float(detected_value)
            }
            
            # Validation criteria
            confidence_factors = []
            
            # 1. Reasonable percentage (not too high, not too low)
            if 0.1 <= nodata_percentage <= 50:
                confidence_factors.append(0.3)
            elif nodata_percentage > 50:
                validation_results["warnings"].append(
                    f"High nodata percentage ({nodata_percentage:.1f}%) - verify detection"
                )
            
            # 2. Check if value is at data extremes
            valid_data = data[~mask]
            if len(valid_data) > 0:
                data_min, data_max = np.min(valid_data), np.max(valid_data)
                if detected_value < data_min or detected_value > data_max:
                    confidence_factors.append(0.4)  # Good - nodata is outside valid range
                
                # 3. Check for large gap between nodata and valid data
                if detected_value < data_min:
                    gap = data_min - detected_value
                    data_range = data_max - data_min
                    if data_range > 0 and gap > 0.1 * data_range:
                        confidence_factors.append(0.2)
            
            # 4. Spatial coherence check
            if spatial_coherence_check and nodata_count > 0:
                coherence_score = _check_spatial_coherence(mask)
                if coherence_score > 0.5:  # Nodata pixels are spatially clustered
                    confidence_factors.append(0.1)
                validation_results["metrics"]["spatial_coherence"] = coherence_score
            
            # Calculate overall confidence
            confidence = min(1.0, sum(confidence_factors))
            validation_results["confidence"] = confidence
            validation_results["is_valid"] = confidence > 0.6
            
            if confidence < 0.3:
                validation_results["warnings"].append(
                    "Low confidence in nodata detection - manual review recommended"
                )
    
    except Exception as e:
        validation_results["warnings"].append(f"Validation failed: {str(e)}")
    
    return validation_results


def _check_spatial_coherence(nodata_mask: np.ndarray) -> float:
    """
    Check spatial coherence of nodata pixels (are they clustered or scattered?).
    
    Returns a score from 0 (completely scattered) to 1 (highly clustered).
    """
    if nodata_mask.ndim != 2:
        return 0.0
    
    try:
        from scipy import ndimage
        # Count connected components of nodata regions
        labeled_array, num_features = ndimage.label(nodata_mask)
        
        if num_features == 0:
            return 0.0
        
        total_nodata = np.sum(nodata_mask)
        
        # Calculate size distribution of connected components
        component_sizes = []
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_array == i)
            component_sizes.append(component_size)
        
        # Higher coherence if fewer, larger components
        largest_component = max(component_sizes)
        coherence = largest_component / total_nodata
        
        # Adjust for number of components (fewer components = higher coherence)
        component_factor = 1.0 / (1.0 + np.log(num_features))
        
        return min(1.0, coherence * component_factor)
        
    except ImportError:
        # Fallback without scipy - simple edge detection
        if nodata_mask.size < 4:
            return 1.0
        
        # Count transitions between nodata and valid pixels
        horizontal_transitions = np.sum(nodata_mask[:, :-1] != nodata_mask[:, 1:])
        vertical_transitions = np.sum(nodata_mask[:-1, :] != nodata_mask[1:, :])
        total_transitions = horizontal_transitions + vertical_transitions
        
        # Fewer transitions relative to nodata pixels suggests clustering
        nodata_count = np.sum(nodata_mask)
        if nodata_count == 0:
            return 0.0
        
        # Normalize by perimeter estimate
        max_possible_transitions = 4 * nodata_count  # Very rough upper bound
        coherence = 1.0 - (total_transitions / max_possible_transitions)
        
        return max(0.0, min(1.0, coherence))


def get_enhanced_raster_info(raster_path: Union[str, Path], band: int = 1) -> Dict[str, Any]:
    """
    Get comprehensive information about a raster including nodata detection results.
    
    Args:
        raster_path: Path to raster file
        band: Band number to analyze
        
    Returns:
        Dictionary with raster information and nodata analysis
    """
    if not HAS_RASTER_LIBS:
        raise GeoWorkflowError("Rasterio required for raster analysis")
    
    info = {}
    
    try:
        with rasterio.open(raster_path) as src:
            # Basic raster info
            info.update({
                "path": str(raster_path),
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[band-1]) if band <= src.count else None,
                "crs": str(src.crs) if src.crs else None,
                "bounds": list(src.bounds),
                "transform": list(src.transform)[:6],
                "metadata_nodata": src.nodata
            })
            
            if band > src.count:
                info["error"] = f"Band {band} not available (raster has {src.count} bands)"
                return info
            
            # Read data for analysis
            data = src.read(band)
            data_flat = data.flatten()
            
            # Basic statistics (excluding existing nodata)
            if src.nodata is not None:
                valid_data = data_flat[data_flat != src.nodata]
            else:
                valid_data = data_flat[~np.isnan(data_flat.astype(float))]
            
            if len(valid_data) > 0:
                info["statistics"] = {
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "valid_pixels": len(valid_data),
                    "total_pixels": len(data_flat)
                }
            
            # Nodata detection
            detected_nodata = detect_nodata_value(raster_path, band, method="auto")
            info["detected_nodata"] = detected_nodata
            
            if detected_nodata is not None:
                validation = validate_nodata_detection(raster_path, detected_nodata, band)
                info["nodata_validation"] = validation
            
    except Exception as e:
        info["error"] = str(e)
    
    return info

def _get_cached_nodata_detection(raster_path: Path) -> Optional[float]:
    """Check if no-data has already been detected for this raster."""
    
    try:
        with rasterio.open(raster_path) as src:
            # Check our custom metadata first
            metadata = src.tags()
            
            if metadata.get(GEOWORKFLOW_NODATA_DETECTED) == "true":
                cached_value = metadata.get(GEOWORKFLOW_NODATA_VALUE)
                if cached_value:
                    return float(cached_value)
            
            # Fall back to standard nodata field if it exists
            if src.nodata is not None:
                return float(src.nodata)
                
    except Exception as e:
        logger.debug(f"Could not read cached no-data for {raster_path}: {e}")
    
    return None

def detect_and_update_nodata(raster_path: Union[str, Path], 
                           force_redetection: bool = False,
                           update_metadata: bool = True) -> Optional[float]:
    """
    Detect no-data value and persist to metadata (direct update only).
    
    Args:
        raster_path: Path to raster file
        force_redetection: Force re-detection even if metadata exists
        update_metadata: Whether to update file metadata with results
        
    Returns:
        Detected/cached no-data value or None
    """
    
    # Step 1: Check if we've already detected no-data for this file
    if not force_redetection:
        cached_result = _get_cached_nodata_detection(raster_path)
        if cached_result is not None:
            logger.debug(f"Using cached no-data detection for {raster_path}")
            return cached_result
    
    # Step 2: Run detection algorithms
    detected_nodata = detect_nodata_value(raster_path, method="auto")
    
    # Step 3: Try to persist results directly to raster metadata
    if update_metadata and detected_nodata is not None:
        success = _update_raster_metadata_directly(raster_path, detected_nodata)
        if not success:
            logger.debug(f"Could not update metadata for {raster_path} - likely read-only format")
    
    return detected_nodata

def _update_raster_metadata_directly(raster_path: Path, detected_nodata: float) -> bool:
    """Update raster metadata directly for writable formats."""
    
    try:
        with rasterio.open(raster_path, 'r+') as src:
            # Update the nodata value if not already set
            if src.nodata is None:
                src.nodata = detected_nodata
                logger.info(f"Set nodata value to {detected_nodata} for {raster_path}")
            
            # Add our custom metadata
            metadata = src.tags()
            metadata.update({
                GEOWORKFLOW_NODATA_DETECTED: "true",
                GEOWORKFLOW_NODATA_VALUE: str(detected_nodata),
                GEOWORKFLOW_NODATA_METHOD: "auto",
                GEOWORKFLOW_NODATA_TIMESTAMP: datetime.now().isoformat()
            })
            src.update_tags(**metadata)
            
        return True
        
    except (PermissionError, rasterio.errors.RasterioIOError):
        # File is read-only or format doesn't support metadata updates
        return False
    except Exception as e:
        logger.warning(f"Failed to update metadata for {raster_path}: {e}")
        return False