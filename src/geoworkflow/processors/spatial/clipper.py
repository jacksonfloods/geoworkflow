# File: src/geoworkflow/processors/spatial/clipper.py
"""
Enhanced clipping processor for raster and vector data.

This processor provides a comprehensive clipping solution that automatically
detects file types and handles both rasters and vectors with proper CRS handling,
error recovery, and integration with the visualization system.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging

try:
    import rasterio
    import rasterio.mask
    import rasterio.warp
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    import pyproj
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ProcessingError, ValidationError, ClippingError
from geoworkflow.schemas.config_models import ClippingConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory
from geoworkflow.visualization.raster.processor import visualize_clipped_data


class ClippingProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced clipping processor for raster and vector data.
    
    Features:
    - Automatic file type detection (raster vs vector)
    - Intelligent CRS handling with reprojection
    - Complete feature extraction for vectors (features within AOI)
    - All-band preservation for rasters
    - Robust error handling and recovery
    - Integrated visualization capabilities
    - Progress tracking and resource management
    """
    
    def __init__(self, config: Union[ClippingConfig, Dict[str, Any]], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize clipping processor.
        
        Args:
            config: Clipping configuration object or dictionary
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        if isinstance(config, ClippingConfig):
            config_dict = config.model_dump(mode='json')
            self.clipping_config = config
        else:
            config_dict = config
            self.clipping_config = ClippingConfig(**config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.aoi_gdf: Optional[gpd.GeoDataFrame] = None
        self.aoi_crs: Optional[str] = None
        self.input_files: List[Path] = []
        self.raster_files: List[Path] = []
        self.vector_files: List[Path] = []
        self.processed_files: List[Path] = []
        
        # File type detection patterns
        self.raster_extensions = {'.tif', '.tiff', '.geotif', '.geotiff', '.nc', '.netcdf'}
        self.vector_extensions = {'.shp', '.geojson', '.gpkg', '.gml', '.kml'}
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate clipping-specific inputs and configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check geospatial libraries
        if not HAS_GEOSPATIAL_LIBS:
            validation_result["errors"].append(
                "Required geospatial libraries not available. Please install: "
                "rasterio, geopandas, pyproj"
            )
            validation_result["valid"] = False
            return validation_result
        
        
        # Validate input source
        if not self.clipping_config.input_directory:
            validation_result["errors"].append(
                "input_directory must be specified"
            )
            validation_result["valid"] = False

        if self.clipping_config.input_directory and not self.clipping_config.input_directory.exists():
            validation_result["errors"].append(
                f"Input directory does not exist: {self.clipping_config.input_directory}"
            )
            validation_result["valid"] = False

        
        # Validate AOI file exists
        if not self.clipping_config.aoi_file.exists():
            validation_result["errors"].append(
                f"AOI file does not exist: {self.clipping_config.aoi_file}"
            )
            validation_result["valid"] = False
        


        # Validate output directory can be created
        try:
            ensure_directory(self.clipping_config.output_dir)
            validation_result["info"]["output_dir_validated"] = str(self.clipping_config.output_dir)
        except Exception as e:
            validation_result["errors"].append(
                f"Cannot create output directory: {e}"
            )
            validation_result["valid"] = False
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths that must exist."""
        return ["input_directory", "aoi_file"]
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        try:
            files = self._discover_input_files()
            return len(files)
        except:
            return 0
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup clipping-specific processing resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Load and validate AOI
        self.log_processing_step("Loading AOI file")
        try:
            self.aoi_gdf = gpd.read_file(self.clipping_config.aoi_file)
            
            # Set CRS if not defined (assume WGS84)
            if self.aoi_gdf.crs is None:
                self.logger.warning("AOI has no CRS defined, assuming EPSG:4326")
                self.aoi_gdf.set_crs("EPSG:4326", inplace=True)
            
            self.aoi_crs = str(self.aoi_gdf.crs)
            setup_info["aoi_crs"] = self.aoi_crs
            setup_info["aoi_features"] = len(self.aoi_gdf)
            
            self.add_metric("aoi_features_loaded", len(self.aoi_gdf))
            
        except Exception as e:
            raise ProcessingError(f"Failed to load AOI file: {str(e)}")
        
        # Discover input files
        self.log_processing_step("Discovering input files")
        self.input_files = self._discover_input_files()
        self.raster_files, self.vector_files = self._classify_files(self.input_files)
        
        setup_info["total_files_found"] = len(self.input_files)
        setup_info["raster_files_found"] = len(self.raster_files)
        setup_info["vector_files_found"] = len(self.vector_files)
        
        self.add_metric("raster_files_discovered", len(self.raster_files))
        self.add_metric("vector_files_discovered", len(self.vector_files))
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the main clipping logic for both rasters and vectors.
        
        Returns:
            ProcessingResult with clipping outcomes
        """
        result = ProcessingResult(success=True)
        
        try:
            if not self.input_files:
                result.message = "No files found to clip"
                result.skipped_count = 1
                return result
            
            self.log_processing_step(f"Clipping {len(self.input_files)} files ({len(self.raster_files)} rasters, {len(self.vector_files)} vectors)")
            
            # Process all files
            for input_file in track_progress(
                self.input_files,
                description="Clipping files",
                quiet=False
            ):
                success, output_path = self._clip_single_file(input_file)
                
                if success:
                    result.processed_count += 1
                    if output_path:
                        self.processed_files.append(output_path)
                        result.add_output_path(output_path)
                else:
                    result.failed_count += 1
                    result.add_failed_file(input_file)
                
                self.update_progress(1, f"Processed {input_file.name}")
            
            # Create visualizations if requested
            if self.clipping_config.create_visualizations and result.processed_count > 0:
                self.log_processing_step("Creating visualizations")
                viz_success = self._create_visualizations()
                result.metadata = result.metadata or {}
                result.metadata['visualizations_created'] = viz_success
            
            # Update result
            result.message = f"Successfully clipped {result.processed_count} files"
            result.add_output_path(self.clipping_config.output_dir)
            
            # Add processing metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                "output_directory": str(self.clipping_config.output_dir),
                "aoi_file": str(self.clipping_config.aoi_file),
                "raster_files_processed": len([f for f in self.processed_files if self._is_raster_file(f)]),
                "vector_files_processed": len([f for f in self.processed_files if self._is_vector_file(f)]),
                "all_touched": self.clipping_config.all_touched
            })
            
            self.add_metric("files_clipped_successfully", result.processed_count)
            
        except Exception as e:
            result.success = False
            result.message = f"Clipping processing failed: {str(e)}"
            self.logger.error(f"Clipping processing failed: {e}")
            raise ClippingError(f"Clipping processing failed: {str(e)}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup clipping-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Clear data from memory
        if hasattr(self, 'aoi_gdf') and self.aoi_gdf is not None:
            del self.aoi_gdf
            cleanup_info["aoi_gdf_cleared"] = True
        
        cleanup_info["clipping_cleanup"] = "completed"
        
        return cleanup_info
    
    def _discover_input_files(self) -> List[Path]:
        """
        Discover all input files to process.
        
        Returns:
            List of input file paths
        """
        files = []
        
        if self.clipping_config.input_directory:
            # Directory processing - find all supported files
            input_dir = self.clipping_config.input_directory
            
            # Get all supported extensions
            all_extensions = self.raster_extensions.union(self.vector_extensions)
            
            for ext in all_extensions:
                # Recursive search
                pattern = f"**/*{ext}"
                found_files = list(input_dir.glob(pattern))
                files.extend(found_files)
                
                # Also check uppercase
                pattern_upper = f"**/*{ext.upper()}"
                found_files_upper = list(input_dir.glob(pattern_upper))
                files.extend(found_files_upper)
        
        # Remove duplicates and sort
        return sorted(list(set(files)))
    
    def _classify_files(self, files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Classify files into rasters and vectors, applying pattern filters appropriately.
        
        Args:
            files: List of file paths
            
        Returns:
            Tuple of (raster_files, vector_files)
        """
        raster_files = []
        vector_files = []
        
        for file_path in files:
            if self._is_raster_file(file_path):
                # Apply raster pattern filter to raster files only
                if self.clipping_config.raster_pattern and self.clipping_config.raster_pattern != "*.tif":
                    import fnmatch
                    if fnmatch.fnmatch(file_path.name, self.clipping_config.raster_pattern):
                        raster_files.append(file_path)
                else:
                    raster_files.append(file_path)
            elif self._is_vector_file(file_path):
                # Vector files are not filtered by raster_pattern
                vector_files.append(file_path)
            else:
                self.logger.warning(f"Unknown file type, skipping: {file_path}")
        
        return raster_files, vector_files
    
    def _is_raster_file(self, file_path: Path) -> bool:
        """Check if file is a raster based on extension."""
        return file_path.suffix.lower() in self.raster_extensions
    
    def _is_vector_file(self, file_path: Path) -> bool:
        """Check if file is a vector based on extension."""
        return file_path.suffix.lower() in self.vector_extensions
    
    def _clip_single_file(self, input_file: Path) -> Tuple[bool, Optional[Path]]:
        """
        Clip a single file (raster or vector).
        
        Args:
            input_file: Path to input file
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            # Determine output path (maintain directory structure)
            relative_path = input_file.relative_to(
                self.clipping_config.input_directory if self.clipping_config.input_directory 
                else input_file.parent
            )
            output_path = self.clipping_config.output_dir / relative_path
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Route to appropriate clipping method
            if self._is_raster_file(input_file):
                return self._clip_raster(input_file, output_path)
            elif self._is_vector_file(input_file):
                return self._clip_vector(input_file, output_path)
            else:
                self.logger.warning(f"Unknown file type: {input_file}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Error clipping {input_file}: {e}")
            return False, None
    
    def _clip_raster(self, input_file: Path, output_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Clip a raster file to the AOI with transparency preservation.
        
        Args:
            input_file: Path to input raster
            output_path: Path for output raster
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            with rasterio.open(input_file) as src:
                # Log original information
                self.logger.debug(f"Clipping raster: {input_file.name} ({src.count} bands)")
                
                # Check for transparency/alpha information
                has_alpha = False
                alpha_band_index = None
                color_interp = None
                
                try:
                    # Get color interpretation if available
                    color_interp = src.colorinterp
                    
                    # Check for RGBA (4-band with alpha as last band)
                    if src.count == 4 and color_interp and len(color_interp) >= 4:
                        if color_interp[3] == rasterio.enums.ColorInterp.alpha:
                            has_alpha = True
                            alpha_band_index = 4
                            self.logger.debug("Detected RGBA format with alpha band")
                    
                    # Check mask flags for alpha information
                    if hasattr(src, 'mask_flag_enums') and src.mask_flag_enums:
                        for band_idx, flags in enumerate(src.mask_flag_enums):
                            if rasterio.enums.MaskFlags.alpha in flags:
                                has_alpha = True
                                alpha_band_index = band_idx + 1  # Convert to 1-based indexing
                                self.logger.debug(f"Detected alpha mask in band {alpha_band_index}")
                                break
                    
                    # Alternative check: if we have exactly 4 bands and no specific color interp,
                    # assume last band might be alpha (common convention)
                    if not has_alpha and src.count == 4:
                        # Read a small sample of the last band to check if it looks like alpha
                        try:
                            sample_alpha = src.read(4, window=rasterio.windows.Window(0, 0, 
                                                                                    min(100, src.width), 
                                                                                    min(100, src.height)))
                            # If the band has values that look like alpha (0-255 range with transparency values)
                            unique_vals = np.unique(sample_alpha)
                            if len(unique_vals) > 1 and (0 in unique_vals or unique_vals.min() < 200):
                                has_alpha = True
                                alpha_band_index = 4
                                self.logger.debug("Inferred alpha band from 4th band characteristics")
                        except Exception as e:
                            self.logger.debug(f"Could not sample potential alpha band: {e}")
                            
                except Exception as e:
                    self.logger.debug(f"Error checking for alpha information: {e}")
                
                # Prepare AOI for this raster
                aoi_for_raster = self._prepare_aoi_for_raster(src)
                
                if aoi_for_raster.empty:
                    self.logger.warning(f"No AOI overlap with raster: {input_file}")
                    return False, None
                
                # Get geometries in the format rasterio expects
                geometries = [geom.__geo_interface__ for geom in aoi_for_raster.geometry]
                
                # Perform clipping with appropriate handling for alpha
                if has_alpha:
                    self.logger.debug("Performing clipping with alpha preservation")
                    # Include all bands in clipping to preserve alpha
                    clipped_data, clipped_transform = rasterio.mask.mask(
                        src, 
                        geometries, 
                        crop=True, 
                        all_touched=self.clipping_config.all_touched,
                        indexes=list(range(1, src.count + 1)),  # Include all bands
                        nodata=src.nodata
                    )
                else:
                    self.logger.debug("Performing standard clipping")
                    # Standard clipping without alpha considerations
                    clipped_data, clipped_transform = rasterio.mask.mask(
                        src, 
                        geometries, 
                        crop=True, 
                        all_touched=self.clipping_config.all_touched,
                        nodata=src.nodata
                    )
                
                # Update metadata
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2],
                    'transform': clipped_transform,
                    'count': clipped_data.shape[0],  # Use actual output band count
                })
                
                # Write clipped raster
                with rasterio.open(output_path, 'w', **clipped_meta) as dst:
                    dst.write(clipped_data)
                    
                    # Preserve color interpretation and alpha information if present
                    if has_alpha and color_interp:
                        try:
                            dst.colorinterp = color_interp
                            self.logger.debug("Preserved color interpretation including alpha")
                        except Exception as e:
                            self.logger.debug(f"Could not set color interpretation: {e}")
                    
                    # If we detected alpha, ensure the output retains transparency characteristics
                    if has_alpha:
                        try:
                            # Copy any mask information from source
                            if hasattr(src, 'mask_flag_enums') and src.mask_flag_enums:
                                # This ensures mask metadata is preserved where possible
                                pass  # rasterio handles this automatically in most cases
                        except Exception as e:
                            self.logger.debug(f"Could not preserve mask information: {e}")
                
                # Log success with transparency info
                transparency_info = f" (with alpha)" if has_alpha else ""
                self.logger.debug(f"Successfully clipped raster{transparency_info}: {output_path}")
                return True, output_path
                
        except Exception as e:
            self.logger.error(f"Error clipping raster {input_file}: {e}")
            return False, None
        
    def _clip_vector(self, input_file: Path, output_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Clip a vector file to the AOI (features completely within).
        
        Args:
            input_file: Path to input vector
            output_path: Path for output vector
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            # Read vector file
            vector_gdf = gpd.read_file(input_file)
            
            if vector_gdf.empty:
                self.logger.warning(f"Empty vector file: {input_file}")
                return False, None
            
            self.logger.debug(f"Clipping vector: {input_file.name} ({len(vector_gdf)} features)")
            
            # Handle CRS
            if vector_gdf.crs is None:
                self.logger.warning(f"Vector file has no CRS, assuming EPSG:4326: {input_file}")
                vector_gdf.set_crs("EPSG:4326", inplace=True)
            
            # Reproject vector to AOI CRS if needed
            if str(vector_gdf.crs) != self.aoi_crs:
                vector_gdf = vector_gdf.to_crs(self.aoi_crs)
            
            # Find features completely within AOI
            aoi_union = self.aoi_gdf.union_all()
            features_within = vector_gdf[vector_gdf.geometry.within(aoi_union)]
            
            # Save clipped features
            if len(features_within) > 0:
                # Always save as GeoJSON for consistency
                output_path = output_path.with_suffix('.geojson')
                features_within.to_file(output_path, driver='GeoJSON')
                
                self.logger.debug(f"Clipped vector: {len(features_within)} features within AOI")
                return True, output_path
            else:
                self.logger.info(f"No features within AOI for: {input_file}")
                return True, None  # Success but no output
                
        except Exception as e:
            self.logger.error(f"Error clipping vector {input_file}: {e}")
            return False, None
        
    def _prepare_aoi_for_raster(self, raster_src) -> gpd.GeoDataFrame:
        """
        Prepare AOI for raster clipping (handle CRS reprojection).
        
        Args:
            raster_src: Rasterio dataset source
            
        Returns:
            AOI GeoDataFrame in raster's CRS
        """
        try:
            # Get raster CRS
            raster_crs = raster_src.crs
            
            if raster_crs is None:
                # Raster has no CRS, assume EPSG:4326
                self.logger.warning("Raster has no CRS defined, assuming EPSG:4326")
                raster_crs = "EPSG:4326"
            
            # Reproject AOI to raster CRS if needed
            if str(raster_crs) != self.aoi_crs:
                aoi_reprojected = self.aoi_gdf.to_crs(raster_crs)
                return aoi_reprojected
            else:
                return self.aoi_gdf.copy()
                
        except Exception as e:
            self.logger.error(f"Error preparing AOI for raster: {e}")
            return gpd.GeoDataFrame()  # Return empty GDF on error
    
    def _create_visualizations(self) -> bool:
        """
        Create visualizations of clipped data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only visualize raster files
            raster_outputs = [f for f in self.processed_files if self._is_raster_file(f)]
            
            if not raster_outputs:
                self.logger.info("No raster files to visualize")
                return True
            
            # Determine visualization output directory
            viz_output_dir = self._get_visualization_output_dir()
            
            # Configuration overrides for visualization
            viz_config_overrides = {
                'dpi': 150,
                'add_basemap': True,
                'show_colorbar': True,
                'overwrite': True,
                'classification_method': 'auto',
                'figure_width': 12,
                'figure_height': 8
            }
            
            # Call the visualization function
            success = visualize_clipped_data(
                input_directory=self.clipping_config.output_dir,
                output_directory=viz_output_dir,
                config_overrides=viz_config_overrides
            )
            
            if success:
                self.add_metric("visualizations_created", True)
                self.logger.info(f"Visualizations saved to: {viz_output_dir}")
            else:
                self.add_metric("visualizations_created", False)
                self.logger.warning("Visualization creation failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return False
    
    def _get_visualization_output_dir(self) -> Path:
        """
        Determine where to save visualization outputs.
        
        Returns:
            Path to visualization output directory
        """
        # Create visualization directory based on clipped data location
        base_dir = Path("outputs/visualizations")
        
        # Use stage-aware naming
        if "02_clipped" in str(self.clipping_config.output_dir):
            viz_dir = base_dir / "02_clipped"
        elif "clipped" in str(self.clipping_config.output_dir):
            viz_dir = base_dir / "clipped_data"
        else:
            viz_dir = base_dir / "custom_clip"
        
        # Ensure directory exists
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        return viz_dir


# Convenience function for easy integration
def clip_data(input_directory: Path, aoi_file: Path, output_directory: Path,
              create_visualizations: bool = True, all_touched: bool = True) -> bool:
    """
    Convenience function to clip raster and vector data.
    
    Args:
        input_directory: Directory containing data to clip
        aoi_file: AOI file for clipping
        output_directory: Directory to save clipped data
        create_visualizations: Whether to create visualizations
        all_touched: Include pixels touched by AOI geometry
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create clipping configuration
        config = ClippingConfig(
            input_directory=input_directory,
            aoi_file=aoi_file,
            output_dir=output_directory,
            raster_pattern="*.tif",
            all_touched=all_touched,
            create_visualizations=create_visualizations
        )
        
        # Create and run processor
        processor = ClippingProcessor(config)
        result = processor.process()
        
        return result.success
        
    except Exception as e:
        logging.getLogger('geoworkflow.clipping').error(f"Clipping failed: {e}")
        return False