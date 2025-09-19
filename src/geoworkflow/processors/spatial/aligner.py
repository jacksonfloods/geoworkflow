# File: src/geoworkflow/processors/spatial/aligner.py
"""
Enhanced raster alignment processor for standardizing geospatial raster datasets.

This processor aligns multiple raster datasets to a reference raster's grid, CRS, and extent,
ensuring all rasters have consistent spatial properties for analysis. Now includes enhanced
nodata detection and caching to prevent interpolation artifacts.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging

try:
    import rasterio
    import rasterio.warp
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import numpy as np
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ProcessingError, ValidationError, GeospatialError
from geoworkflow.schemas.config_models import AlignmentConfig, ResamplingMethod
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory

from datetime import datetime

# Custom metadata keys for nodata detection caching
GEOWORKFLOW_NODATA_DETECTED = "GEOWORKFLOW_NODATA_DETECTED"
GEOWORKFLOW_NODATA_VALUE = "GEOWORKFLOW_NODATA_VALUE" 
GEOWORKFLOW_NODATA_METHOD = "GEOWORKFLOW_NODATA_METHOD"
GEOWORKFLOW_NODATA_CONFIDENCE = "GEOWORKFLOW_NODATA_CONFIDENCE"
GEOWORKFLOW_NODATA_TIMESTAMP = "GEOWORKFLOW_NODATA_TIMESTAMP"

# Enhanced common nodata sentinel values used across different systems
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

# Mapping from ResamplingMethod enum to rasterio Resampling enum
RESAMPLING_METHOD_MAP = {
    ResamplingMethod.NEAREST: Resampling.nearest,
    ResamplingMethod.BILINEAR: Resampling.bilinear,
    ResamplingMethod.CUBIC: Resampling.cubic,
    ResamplingMethod.CUBIC_SPLINE: Resampling.cubic_spline,
    ResamplingMethod.LANCZOS: Resampling.lanczos,
    ResamplingMethod.AVERAGE: Resampling.average,
    ResamplingMethod.MODE: Resampling.mode
}

# File extensions for raster files
RASTER_EXTENSIONS = {'.tif', '.tiff', '.geotif', '.geotiff'}


class AlignmentProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced raster alignment processor for standardizing datasets.
    
    This processor:
    - Loads reference raster metadata (CRS, transform, bounds)
    - Discovers and validates input raster files  
    - Calculates intersection bounds between reference and input rasters
    - Aligns rasters to reference grid with appropriate resampling
    - Preserves all bands and handles different data types
    - Automatically detects and caches nodata values to prevent artifacts
    - Automatically selects resampling method based on data characteristics
    """
    def __init__(self, config: Union[AlignmentConfig, Dict[str, Any]], 
             logger: Optional[logging.Logger] = None):
        """
        Initialize alignment processor.
        
        Args:
            config: Alignment configuration object or dictionary
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        if isinstance(config, AlignmentConfig):
            config_dict = config.model_dump()
            self.alignment_config = config
        else:
            config_dict = config
            self.alignment_config = AlignmentConfig(**config_dict)
        
        super().__init__(config_dict, logger)
            
        if not HAS_GEOSPATIAL_LIBS:
            raise GeospatialError("Rasterio is required for alignment operations")
        
        # Initialize reference raster properties
        self.reference_meta = None
        self.reference_bounds = None
        self.reference_crs = None
        self.reference_transform = None
        
        self.logger.info("AlignmentProcessor initialized")
        self.logger.info(f"Enhanced NoData Detection: {self.alignment_config.auto_detect_nodata}")
        self.logger.info(f"NoData Method: {self.alignment_config.nodata_detection_method}")

    def process_data(self) -> ProcessingResult:
        """
        Execute the main alignment processing logic.
        
        Returns:
            ProcessingResult with processing outcomes
        """
        try:
            self.logger.info("Starting raster alignment processing")
            
            # Load reference raster
            if not self._load_reference_raster():
                return ProcessingResult(
                    success=False,
                    message="Failed to load reference raster"
                )
            
            # Discover input files
            input_files = self._discover_input_files()
            if not input_files:
                return ProcessingResult(
                    success=False,
                    message="No input raster files found"
                )
            
            self.logger.info(f"Found {len(input_files)} raster files to align")
            
            # Process files with progress tracking
            results = {
                "total_files": len(input_files),
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "processed_files": [],
                "failed_files": [],
                "skipped_files": []
            }
            
            for i, input_file in enumerate(track_progress(input_files, "Aligning rasters")):
                try:
                    success, output_path = self._align_single_raster(input_file)
                    
                    if success:
                        results["successful"] += 1
                        results["processed_files"].append({
                            "input": str(input_file),
                            "output": str(output_path),
                            "status": "success"
                        })
                        self.logger.debug(f"Successfully aligned: {input_file.name}")
                    else:
                        results["failed"] += 1
                        results["failed_files"].append({
                            "input": str(input_file),
                            "error": "Alignment failed"
                        })
                        
                except Exception as e:
                    results["failed"] += 1
                    results["failed_files"].append({
                        "input": str(input_file),
                        "error": str(e)
                    })
                    self.logger.error(f"Error processing {input_file}: {e}")
            
            # Determine overall success
            success = results["successful"] > 0
            message = f"Aligned {results['successful']}/{results['total_files']} rasters successfully"
            
            if results["failed"] > 0:
                message += f", {results['failed']} failed"
            
            self.logger.info(message)
            
            return ProcessingResult(
                success=success,
                message=message
            )
            
        except Exception as e:
            self.logger.error(f"Alignment processing failed: {e}")
            return ProcessingResult(
                success=False,
                message=f"Processing failed: {str(e)}"
            )
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """
        Clean up processor resources.
        
        Returns:
            Cleanup information
        """
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Clear data from memory
        self.reference_meta = None
        self.reference_bounds = None
        self.reference_crs = None
        self.reference_transform = None
        cleanup_info["reference_data_cleared"] = True
        
        cleanup_info["alignment_cleanup"] = "completed"
        
        return cleanup_info
    
    def _load_reference_raster(self) -> bool:
        """
        Load reference raster metadata and properties.
        
        Returns:
            True if reference raster loaded successfully
        """
        try:
            with rasterio.open(self.alignment_config.reference_raster) as src:
                self.reference_meta = src.meta.copy()
                self.reference_bounds = src.bounds
                self.reference_crs = str(src.crs) if src.crs else None
                self.reference_transform = src.transform
                
                self.logger.info(f"Reference raster loaded: {self.alignment_config.reference_raster}")
                self.logger.info(f"  CRS: {self.reference_crs}")
                self.logger.info(f"  Shape: {src.height} x {src.width}")
                self.logger.info(f"  Resolution: {abs(src.transform.a):.2f} x {abs(src.transform.e):.2f}")
                self.logger.info(f"  Bounds: {self.reference_bounds}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading reference raster: {str(e)}")
            return False
    
    def _discover_input_files(self) -> List[Path]:
        """
        Discover all input raster files to process.
        
        Returns:
            List of input raster file paths
        """
        input_files = []
        
        if self.alignment_config.input_directory:
            # Search for raster files in input directory
            search_pattern = "**/*" if self.alignment_config.recursive else "*"
            
            for extension in self.alignment_config.file_extensions:
                pattern = f"{search_pattern}{extension}"
                files = list(self.alignment_config.input_directory.glob(pattern))
                input_files.extend(files)
                
                # Also search for uppercase extensions
                pattern_upper = f"{search_pattern}{extension.upper()}"
                files_upper = list(self.alignment_config.input_directory.glob(pattern_upper))
                input_files.extend(files_upper)
        
        # Remove duplicates and filter out reference raster
        input_files = list(set(input_files))
        input_files = [f for f in input_files if f != self.alignment_config.reference_raster]
        
        return sorted(input_files)
    
    def _align_single_raster(self, input_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Align a single raster with enhanced nodata handling.
        
        Args:
            input_path: Path to the input raster file
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            # Create output path
            output_path = self._create_output_path(input_path)
            
            # Skip if file exists and skip_existing is True
            if self.alignment_config.skip_existing and output_path.exists():
                self.logger.debug(f"Skipping existing output: {output_path}")
                return True, output_path
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(input_path) as src:
                self.logger.debug(f"Aligning: {input_path.name}")
                self.logger.debug(f"  Input CRS: {src.crs}")
                self.logger.debug(f"  Input shape: {src.height} x {src.width}")
                self.logger.debug(f"  Input bounds: {src.bounds}")
                
                # Enhanced nodata detection and handling
                effective_nodata = self._get_effective_nodata(input_path, src)
                
                # Calculate intersection bounds
                try:
                    intersection_bounds = self._calculate_intersection_bounds(src.bounds)
                    self.logger.debug(f"  Intersection bounds: {intersection_bounds}")
                except ValueError as e:
                    self.logger.error(f"  {str(e)}")
                    return False, None
                
                # Calculate aligned transform and dimensions
                aligned_transform, width, height = self._calculate_aligned_grid(intersection_bounds)
                
                self.logger.debug(f"  Output shape: {height} x {width}")
                self.logger.debug(f"  Effective nodata: {effective_nodata}")
                
                # Prepare output metadata
                output_meta = self.reference_meta.copy()
                output_meta.update({
                    'height': height,
                    'width': width,
                    'transform': aligned_transform,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'nodata': effective_nodata
                })
                
                # Get resampling method
                resampling_method = self._get_resampling_method(input_path)
                
                # Perform alignment/reprojection
                with rasterio.open(output_path, 'w', **output_meta) as dst:
                    for band_idx in range(1, src.count + 1):
                        source_data = src.read(band_idx)
                        
                        # Create destination array
                        dest_data = np.zeros((height, width), dtype=output_meta['dtype'])
                        
                        # Reproject/align the data with enhanced nodata handling
                        reproject(
                            source=source_data,
                            destination=dest_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=aligned_transform,
                            dst_crs=rasterio.crs.CRS.from_string(self.reference_crs),
                            resampling=resampling_method,
                            src_nodata=effective_nodata,  # Use our enhanced detection
                            dst_nodata=effective_nodata   # Preserve the detected value
                        )
                        
                        # Write the aligned data
                        dst.write(dest_data, band_idx)
                
                self.logger.debug(f"  Successfully aligned: {output_path.name}")
                return True, output_path
                
        except Exception as e:
            self.logger.error(f"Error aligning raster {input_path}: {str(e)}")
            return False, None
    
    def _get_effective_nodata(self, input_path: Path, src: rasterio.DatasetReader) -> Optional[float]:
        """
        Get effective nodata value using enhanced detection and caching.
        
        Args:
            input_path: Path to the input raster
            src: Open rasterio dataset reader
            
        Returns:
            Effective nodata value or None
        """
        effective_nodata = src.nodata
        
        # Auto-detect nodata if metadata is missing and auto-detection is enabled
        if (effective_nodata is None and self.alignment_config.auto_detect_nodata):
            
            self.logger.debug(f"Auto-detecting nodata for {input_path.name}")
            
            # Import here to avoid circular imports
            from geoworkflow.utils.raster_utils import detect_and_update_nodata, validate_nodata_detection
            
            detected_nodata = detect_and_update_nodata(
                input_path,
                force_redetection=self.alignment_config.force_nodata_redetection,
                update_metadata=self.alignment_config.nodata_cache_metadata
            )
            
            if detected_nodata is not None:
                # Validate detection if enabled
                if self.alignment_config.nodata_validation:
                    validation = validate_nodata_detection(input_path, detected_nodata)
                    if validation["is_valid"] and validation["confidence"] > 0.7:
                        effective_nodata = detected_nodata
                        self.logger.info(f"Auto-detected nodata value {detected_nodata} for {input_path.name} "
                                       f"(confidence: {validation['confidence']:.1%})")
                    else:
                        self.logger.warning(f"Low confidence nodata detection for {input_path.name}, "
                                          f"using metadata value")
                else:
                    effective_nodata = detected_nodata
                    self.logger.info(f"Auto-detected nodata value {detected_nodata} for {input_path.name}")
        
        return effective_nodata
    
    def _calculate_intersection_bounds(self, input_bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Calculate intersection bounds between reference and input raster.
        
        Args:
            input_bounds: Bounds of the input raster (left, bottom, right, top)
            
        Returns:
            Intersection bounds (left, bottom, right, top)
            
        Raises:
            ValueError: If no intersection exists
        """
        ref_left, ref_bottom, ref_right, ref_top = self.reference_bounds
        in_left, in_bottom, in_right, in_top = input_bounds
        
        # Calculate intersection
        left = max(ref_left, in_left)
        bottom = max(ref_bottom, in_bottom)
        right = min(ref_right, in_right)
        top = min(ref_top, in_top)
        
        # Check if there's actually an intersection
        if left >= right or bottom >= top:
            raise ValueError("No intersection between reference and input raster bounds")
        
        return (left, bottom, right, top)
    
    def _calculate_aligned_grid(self, intersection_bounds: Tuple[float, float, float, float]) -> Tuple[rasterio.Affine, int, int]:
        """
        Calculate aligned grid transform and dimensions.
        
        Args:
            intersection_bounds: Intersection bounds (left, bottom, right, top)
            
        Returns:
            Tuple of (aligned_transform, width, height)
        """
        left, bottom, right, top = intersection_bounds
        
        # Get reference pixel sizes
        ref_pixel_width = abs(self.reference_transform.a)
        ref_pixel_height = abs(self.reference_transform.e)
        
        # Align to reference grid
        ref_left, ref_top = self.reference_bounds[0], self.reference_bounds[3]
        
        # Calculate grid-aligned bounds
        col_off = round((left - ref_left) / ref_pixel_width)
        row_off = round((ref_top - top) / ref_pixel_height)
        
        aligned_left = ref_left + col_off * ref_pixel_width
        aligned_top = ref_top - row_off * ref_pixel_height
        
        # Calculate dimensions
        width = int(round((right - aligned_left) / ref_pixel_width))
        height = int(round((aligned_top - bottom) / ref_pixel_height))
        
        # Create aligned transform
        aligned_transform = from_bounds(
            aligned_left, 
            aligned_top - height * ref_pixel_height,
            aligned_left + width * ref_pixel_width,
            aligned_top,
            width, 
            height
        )
        
        return aligned_transform, width, height
    
    def _get_resampling_method(self, raster_path: Path) -> Resampling:
        """
        Determine appropriate resampling method based on raster type.
        
        Args:
            raster_path: Path to the raster file
            
        Returns:
            Appropriate resampling method
        """
        # Use configured method if explicitly set
        configured_method = RESAMPLING_METHOD_MAP.get(
            self.alignment_config.resampling_method, 
            Resampling.cubic
        )
        
        # Auto-detect based on file path/name if using CUBIC (default)
        if self.alignment_config.resampling_method == ResamplingMethod.CUBIC:
            path_str = str(raster_path).lower()
            
            # Categorical data - use nearest neighbor
            if any(keyword in path_str for keyword in ['land_cover', 'landcover', 'lc', 'class', 'category']):
                self.logger.debug(f"Using nearest neighbor resampling for categorical data: {raster_path.name}")
                return Resampling.nearest
        
        self.logger.debug(f"Using {configured_method.name} resampling for: {raster_path.name}")
        return configured_method
    
    def _create_output_path(self, input_path: Path) -> Path:
        """
        Create output path for aligned raster.
        
        Args:
            input_path: Path to the input raster file
            
        Returns:
            Path for the output aligned raster
        """
        if self.alignment_config.preserve_directory_structure and self.alignment_config.input_directory:
            # Preserve the relative directory structure
            try:
                rel_path = input_path.relative_to(self.alignment_config.input_directory)
                output_path = self.alignment_config.output_dir / rel_path
            except ValueError:
                # If input_path is not relative to input_directory, just use filename
                output_path = self.alignment_config.output_dir / input_path.name
        else:
            # Just use the filename
            output_path = self.alignment_config.output_dir / input_path.name
        
        return output_path


# Convenience function for easy integration
def align_rasters(reference_raster: Union[str, Path], 
                 input_directory: Union[str, Path], 
                 output_directory: Union[str, Path],
                 resampling_method: str = "cubic",
                 recursive: bool = True,
                 auto_detect_nodata: bool = True) -> bool:
    """
    Convenience function to align raster datasets to a reference grid.
    
    Args:
        reference_raster: Path to reference raster file
        input_directory: Directory containing rasters to align
        output_directory: Directory to save aligned rasters
        resampling_method: Resampling method ('nearest', 'bilinear', 'cubic')
        recursive: Whether to search input directory recursively
        auto_detect_nodata: Whether to automatically detect nodata values
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Map string to enum
        method_map = {
            'nearest': ResamplingMethod.NEAREST,
            'bilinear': ResamplingMethod.BILINEAR,
            'cubic': ResamplingMethod.CUBIC,
            'cubic_spline': ResamplingMethod.CUBIC_SPLINE,
            'lanczos': ResamplingMethod.LANCZOS,
            'average': ResamplingMethod.AVERAGE,
            'mode': ResamplingMethod.MODE
        }
        
        resampling_enum = method_map.get(resampling_method.lower(), ResamplingMethod.CUBIC)
        
        # Create alignment configuration
        config = AlignmentConfig(
            reference_raster=Path(reference_raster),
            input_directory=Path(input_directory),
            output_dir=Path(output_directory),
            resampling_method=resampling_enum,
            recursive=recursive,
            skip_existing=True,
            auto_detect_nodata=auto_detect_nodata
        )
        
        # Create and run processor
        processor = AlignmentProcessor(config)
        result = processor.process()
        
        return result.success
        
    except Exception as e:
        logging.getLogger('geoworkflow.alignment').error(f"Alignment failed: {e}")
        return False
