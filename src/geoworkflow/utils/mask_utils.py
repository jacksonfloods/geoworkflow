"""
Mask utilities for rasterio datasets.

This module provides functions for masking specified values in rasterio datasets
with nodata values, useful for filtering out unwanted data values.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def mask_values_to_nodata(
    input_raster: Union[str, Path], 
    output_raster: Union[str, Path],
    mask_values: Union[float, int, List[Union[float, int]]],
    nodata_value: Optional[Union[float, int]] = None,
    bands: Optional[List[int]] = None,
    comparison_operator: str = "equal"
) -> bool:
    """
    Mask specified values in a raster by setting them to nodata.
    
    Args:
        input_raster: Path to input raster file
        output_raster: Path to output raster file
        mask_values: Single value or list of values to mask
        nodata_value: NoData value to use (if None, uses existing or -9999)
        bands: List of band indices to process (1-based, if None processes all)
        comparison_operator: How to compare values ('equal', 'greater', 'less', 'greater_equal', 'less_equal')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure mask_values is a list
        if not isinstance(mask_values, list):
            mask_values = [mask_values]
            
        with rasterio.open(input_raster) as src:
            # Determine nodata value
            if nodata_value is None:
                nodata_value = src.nodata if src.nodata is not None else -9999
                
            # Determine which bands to process
            if bands is None:
                bands = list(range(1, src.count + 1))
            else:
                bands = [b for b in bands if 1 <= b <= src.count]
                
            # Update metadata
            output_meta = src.meta.copy()
            output_meta.update(nodata=nodata_value)
            
            with rasterio.open(output_raster, 'w', **output_meta) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(band_idx)
                    
                    if band_idx in bands:
                        # Apply masking to this band
                        masked_data = _apply_value_mask(
                            data, mask_values, nodata_value, comparison_operator
                        )
                        dst.write(masked_data, band_idx)
                    else:
                        # Copy band as-is
                        dst.write(data, band_idx)
        
        logger.debug(f"Successfully masked values in {input_raster} -> {output_raster}")
        return True
        
    except Exception as e:
        logger.error(f"Error masking values in {input_raster}: {e}")
        return False


def _apply_value_mask(
    data: np.ndarray,
    mask_values: List[Union[float, int]],
    nodata_value: Union[float, int],
    comparison_operator: str
) -> np.ndarray:
    """Apply value masking to a data array."""
    masked_data = data.copy()
    
    for value in mask_values:
        if comparison_operator == "equal":
            mask = (data == value)
        elif comparison_operator == "greater":
            mask = (data > value)
        elif comparison_operator == "less":
            mask = (data < value)
        elif comparison_operator == "greater_equal":
            mask = (data >= value)
        elif comparison_operator == "less_equal":
            mask = (data <= value)
        else:
            raise ValueError(f"Unknown comparison operator: {comparison_operator}")
            
        masked_data[mask] = nodata_value
        
    return masked_data


def mask_by_ranges(
    input_raster: Union[str, Path],
    output_raster: Union[str, Path],
    value_ranges: List[Tuple[Union[float, int], Union[float, int]]],
    nodata_value: Optional[Union[float, int]] = None,
    bands: Optional[List[int]] = None,
    mask_inside_ranges: bool = True
) -> bool:
    """
    Mask values within specified ranges.
    
    Args:
        input_raster: Path to input raster file
        output_raster: Path to output raster file
        value_ranges: List of (min, max) tuples defining ranges to mask
        nodata_value: NoData value to use (if None, uses existing or -9999)
        bands: List of band indices to process (1-based, if None processes all)
        mask_inside_ranges: If True, mask values inside ranges; if False, mask outside
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with rasterio.open(input_raster) as src:
            if nodata_value is None:
                nodata_value = src.nodata if src.nodata is not None else -9999
                
            if bands is None:
                bands = list(range(1, src.count + 1))
            else:
                bands = [b for b in bands if 1 <= b <= src.count]
                
            output_meta = src.meta.copy()
            output_meta.update(nodata=nodata_value)
            
            with rasterio.open(output_raster, 'w', **output_meta) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(band_idx)
                    
                    if band_idx in bands:
                        masked_data = data.copy()
                        
                        # Create combined mask for all ranges
                        combined_mask = np.zeros_like(data, dtype=bool)
                        for min_val, max_val in value_ranges:
                            range_mask = (data >= min_val) & (data <= max_val)
                            combined_mask = combined_mask | range_mask
                            
                        # Apply mask based on mask_inside_ranges parameter
                        if mask_inside_ranges:
                            masked_data[combined_mask] = nodata_value
                        else:
                            masked_data[~combined_mask] = nodata_value
                            
                        dst.write(masked_data, band_idx)
                    else:
                        dst.write(data, band_idx)
        
        logger.debug(f"Successfully applied range masking: {input_raster} -> {output_raster}")
        return True
        
    except Exception as e:
        logger.error(f"Error applying range masking to {input_raster}: {e}")
        return False