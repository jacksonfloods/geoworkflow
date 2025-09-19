"""
Value masking processor for the geoworkflow package.

This processor provides functionality to mask specified values in raster datasets.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
import logging

from ...core.base import BaseProcessor, ProcessingResult
from ...core.exceptions import ProcessingError
from ...utils.mask_utils import mask_values_to_nodata, mask_by_ranges
from ...utils.resource_utils import ProcessingMetrics


class MaskingProcessor(BaseProcessor):
    """
    Processor for masking specified values in raster datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.masking_config = config
        
    def process_file(self, input_path: Path, output_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Process a single raster file to mask specified values.
        
        Args:
            input_path: Path to input raster
            output_path: Path for output raster
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            mask_type = self.masking_config.get('mask_type', 'values')
            
            if mask_type == 'values':
                success = self._mask_specific_values(input_path, output_path)
            elif mask_type == 'ranges':
                success = self._mask_value_ranges(input_path, output_path)
            else:
                raise ProcessingError(f"Unknown mask type: {mask_type}")
                
            return success, output_path if success else None
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {e}")
            return False, None
    
    def _mask_specific_values(self, input_path: Path, output_path: Path) -> bool:
        """Mask specific values from the raster."""
        mask_values = self.masking_config['mask_values']
        nodata_value = self.masking_config.get('nodata_value')
        bands = self.masking_config.get('bands')
        comparison_operator = self.masking_config.get('comparison_operator', 'equal')
        
        return mask_values_to_nodata(
            input_path, output_path, mask_values,
            nodata_value, bands, comparison_operator
        )
    
    def _mask_value_ranges(self, input_path: Path, output_path: Path) -> bool:
        """Mask value ranges from the raster."""
        value_ranges = self.masking_config['value_ranges']
        nodata_value = self.masking_config.get('nodata_value')
        bands = self.masking_config.get('bands')
        mask_inside_ranges = self.masking_config.get('mask_inside_ranges', True)
        
        return mask_by_ranges(
            input_path, output_path, value_ranges,
            nodata_value, bands, mask_inside_ranges
        )