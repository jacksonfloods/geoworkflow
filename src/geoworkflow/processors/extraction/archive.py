# File: src/geoworkflow/processors/extraction/archive.py
"""
Enhanced archive extraction processor that extracts and clips geospatial data.

This processor combines archive extraction with automatic clipping using the
existing ClippingProcessor, providing a unified workflow from ZIP archives
to analysis-ready clipped data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging
import zipfile
import tempfile
import shutil

try:
    import geopandas as gpd
    import rasterio
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ProcessingError, ValidationError, ExtractionError
from geoworkflow.schemas.config_models import ExtractionConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory, temp_directory
from geoworkflow.processors.spatial.clipper import ClippingProcessor, ClippingConfig
from geoworkflow.visualization.raster.processor import visualize_clipped_data


class ArchiveExtractionProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced archive extraction processor with integrated clipping and visualization.
    
    This processor:
    - Extracts files from ZIP archives (both single file and batch)
    - Automatically detects raster vs vector data
    - Clips extracted data to AOI using ClippingProcessor
    - Creates visualizations of clipped outputs
    - Manages temporary files and comprehensive error handling
    """
    
    def __init__(self, config: Union[ExtractionConfig, Dict[str, Any]], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize archive extraction processor.
        
        Args:
            config: Extraction configuration object or dictionary
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        if isinstance(config, ExtractionConfig):
            config_dict = config.model_dump(mode='json')
            self.extraction_config = config
        else:
            config_dict = config
            self.extraction_config = ExtractionConfig.from_dict(config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.aoi_gdf: Optional[gpd.GeoDataFrame] = None
        self.archive_files: List[Path] = []
        self.extracted_files: List[Path] = []
        self.clipped_files: List[Path] = []
        
        # File type detection patterns
        self.raster_extensions = {'.tif', '.tiff', '.geotif', '.geotiff', '.nc', '.netcdf'}
        self.vector_extensions = {'.shp', '.geojson', '.gpkg', '.gml', '.kml'}
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate extraction-specific inputs and configuration.
        
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
                "geopandas, rasterio"
            )
            validation_result["valid"] = False
            return validation_result
        
        # Validate input source
        if not self.extraction_config.zip_file and not self.extraction_config.zip_folder:
            validation_result["errors"].append(
                "Either zip_file or zip_folder must be specified"
            )
            validation_result["valid"] = False
        
        # Validate input paths exist
        if self.extraction_config.zip_file and not self.extraction_config.zip_file.exists():
            validation_result["errors"].append(
                f"ZIP file does not exist: {self.extraction_config.zip_file}"
            )
            validation_result["valid"] = False
        
        if self.extraction_config.zip_folder and not self.extraction_config.zip_folder.exists():
            validation_result["errors"].append(
                f"ZIP folder does not exist: {self.extraction_config.zip_folder}"
            )
            validation_result["valid"] = False
        
        # Validate AOI file exists
        if not self.extraction_config.aoi_file.exists():
            validation_result["errors"].append(
                f"AOI file does not exist: {self.extraction_config.aoi_file}"
            )
            validation_result["valid"] = False
        
        # Validate output directory can be created
        try:
            ensure_directory(self.extraction_config.output_dir)
            validation_result["info"]["output_dir_validated"] = str(self.extraction_config.output_dir)
        except Exception as e:
            validation_result["errors"].append(
                f"Cannot create output directory: {e}"
            )
            validation_result["valid"] = False
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths that must exist."""
        return ["zip_file", "zip_folder", "aoi_file"]
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        try:
            archives = self._discover_archive_files()
            # Estimate files per archive (rough approximation)
            estimated_files_per_archive = 10
            return len(archives) * estimated_files_per_archive
        except:
            return 0
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup extraction-specific processing resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Load and validate AOI
        self.log_processing_step("Loading AOI file")
        try:
            self.aoi_gdf = gpd.read_file(self.extraction_config.aoi_file)
            
            # Set CRS if not defined (assume WGS84)
            if self.aoi_gdf.crs is None:
                self.logger.warning("AOI has no CRS defined, assuming EPSG:4326")
                self.aoi_gdf.set_crs("EPSG:4326", inplace=True)
            
            setup_info["aoi_crs"] = str(self.aoi_gdf.crs)
            setup_info["aoi_features"] = len(self.aoi_gdf)
            
            self.add_metric("aoi_features_loaded", len(self.aoi_gdf))
            
        except Exception as e:
            raise ProcessingError(f"Failed to load AOI file: {str(e)}")
        
        # Discover archive files
        self.log_processing_step("Discovering archive files")
        self.archive_files = self._discover_archive_files()
        
        setup_info["archive_files_found"] = len(self.archive_files)
        self.add_metric("archive_files_discovered", len(self.archive_files))
        
        # Note: We don't set up the clipping processor here anymore
        # since we need the temporary extraction directory first
        setup_info["clipping_setup"] = "deferred_to_processing"
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the main extraction and clipping logic.
        
        Returns:
            ProcessingResult with processing outcomes
        """
        result = ProcessingResult(success=True)
        
        try:
            if not self.archive_files:
                result.message = "No archive files found to process"
                result.skipped_count = 1
                return result
            
            self.log_processing_step(f"Processing {len(self.archive_files)} archive files")
            
            # Process each archive
            for archive_file in track_progress(
                self.archive_files,
                description="Processing archives",
                quiet=False
            ):
                success, extracted_count, clipped_count = self._process_single_archive(archive_file)
                
                if success:
                    result.processed_count += clipped_count
                    self.add_metric("archives_processed_successfully", 1)
                else:
                    result.failed_count += 1
                    result.add_failed_file(archive_file)
                
                self.update_progress(1, f"Processed {archive_file.name}")
            
            # Create visualizations if requested
            if self.extraction_config.create_visualizations and result.processed_count > 0:
                self.log_processing_step("Creating visualizations")
                viz_success = self._create_visualizations()
                result.metadata = result.metadata or {}
                result.metadata['visualizations_created'] = viz_success
            
            # Update result
            result.message = f"Successfully processed {len(self.archive_files)} archives, clipped {result.processed_count} files"
            result.add_output_path(self.extraction_config.output_dir)
            
            # Add processing metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                "archives_processed": len(self.archive_files),
                "output_directory": str(self.extraction_config.output_dir),
                "aoi_file": str(self.extraction_config.aoi_file),
                "total_files_extracted": len(self.extracted_files),
                "total_files_clipped": len(self.clipped_files)
            })
            
            self.add_metric("total_files_clipped", len(self.clipped_files))
            
        except Exception as e:
            result.success = False
            result.message = f"Archive processing failed: {str(e)}"
            self.logger.error(f"Archive processing failed: {e}")
            raise ExtractionError(f"Archive processing failed: {str(e)}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup extraction-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Clear data from memory
        if hasattr(self, 'aoi_gdf') and self.aoi_gdf is not None:
            del self.aoi_gdf
            cleanup_info["aoi_gdf_cleared"] = True
        
        # Clear file lists
        self.extracted_files.clear()
        self.clipped_files.clear()
        cleanup_info["file_lists_cleared"] = True
        
        cleanup_info["extraction_cleanup"] = "completed"
        
        return cleanup_info
    
    def _discover_archive_files(self) -> List[Path]:
        """
        Discover all archive files to process.
        
        Returns:
            List of archive file paths
        """
        archive_files = []
        
        if self.extraction_config.zip_file and self.extraction_config.zip_file.is_file():
            # Single file
            archive_files.append(self.extraction_config.zip_file)
        elif self.extraction_config.zip_folder:
            # Directory processing - find all ZIP files
            archive_files.extend(self.extraction_config.zip_folder.rglob("*.zip"))
            archive_files.extend(self.extraction_config.zip_folder.rglob("*.ZIP"))
        
        return sorted(archive_files)
    
    def _process_single_archive(self, archive_file: Path) -> Tuple[bool, int, int]:
        """
        Process a single archive file: extract and clip.
        
        Args:
            archive_file: Path to archive file
            
        Returns:
            Tuple of (success, extracted_count, clipped_count)
        """
        extracted_count = 0
        clipped_count = 0
        
        try:
            self.logger.info(f"Processing archive: {archive_file.name}")
            
            # Create temporary directory for extraction
            with temp_directory(suffix=f"_{archive_file.stem}") as temp_dir:
                # Extract archive
                extracted_files = self._extract_archive(archive_file, temp_dir)
                extracted_count = len(extracted_files)
                
                if not extracted_files:
                    self.logger.warning(f"No files extracted from {archive_file}")
                    return True, 0, 0
                
                # Classify files
                raster_files, vector_files = self._classify_extracted_files(extracted_files)
                
                self.logger.debug(f"Extracted {len(raster_files)} raster files, {len(vector_files)} vector files")
                
                # Create clipping configuration for this specific extraction
                temp_clipping_config = ClippingConfig(
                    input_directory=temp_dir,  # Now we have the actual temp directory
                    aoi_file=self.extraction_config.aoi_file,
                    output_dir=self._get_archive_output_dir(archive_file),
                    raster_pattern=self.extraction_config.raster_pattern,
                    all_touched=True,
                    create_visualizations=False
                )
                
                # Create temporary clipping processor
                temp_clipper = ClippingProcessor(temp_clipping_config, self.logger)
                
                # Execute clipping
                clipping_result = temp_clipper.process()
                
                if clipping_result.success:
                    clipped_count = clipping_result.processed_count
                    self.clipped_files.extend(clipping_result.output_paths or [])
                    self.logger.info(f"Successfully clipped {clipped_count} files from {archive_file.name}")
                else:
                    self.logger.error(f"Clipping failed for {archive_file.name}: {clipping_result.message}")
                    return False, extracted_count, 0
            
            return True, extracted_count, clipped_count
            
        except Exception as e:
            self.logger.error(f"Error processing archive {archive_file}: {e}")
            return False, extracted_count, clipped_count
    
    def _extract_archive(self, archive_file: Path, extract_dir: Path) -> List[Path]:
        """
        Extract files from archive.
        
        Args:
            archive_file: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            List of extracted file paths
        """
        extracted_files = []
        
        try:
            with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                # Get list of files to extract
                all_files = zip_ref.namelist()
                
                # Filter for geospatial files
                geospatial_files = [
                    f for f in all_files 
                    if any(f.lower().endswith(ext) for ext in self.raster_extensions.union(self.vector_extensions))
                ]
                
                # Extract geospatial files
                for file_name in geospatial_files:
                    if not file_name.endswith('/'):  # Skip directories
                        try:
                            zip_ref.extract(file_name, extract_dir)
                            extracted_path = extract_dir / file_name
                            extracted_files.append(extracted_path)
                            self.logger.debug(f"Extracted: {file_name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to extract {file_name}: {e}")
                
                # Handle shapefile components
                extracted_files.extend(self._extract_shapefile_components(zip_ref, extract_dir))
                
        except Exception as e:
            self.logger.error(f"Error extracting archive {archive_file}: {e}")
            
        return extracted_files
    
    def _extract_shapefile_components(self, zip_ref: zipfile.ZipFile, extract_dir: Path) -> List[Path]:
        """Extract all components of shapefiles."""
        shapefile_components = []
        all_files = zip_ref.namelist()
        
        # Find .shp files
        shp_files = [f for f in all_files if f.lower().endswith('.shp')]
        
        for shp_file in shp_files:
            base_name = Path(shp_file).stem
            base_dir = str(Path(shp_file).parent)
            
            # Shapefile extensions
            shapefile_extensions = ['.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']
            
            # Extract all related files
            for file_path in all_files:
                file_path_obj = Path(file_path)
                if (str(file_path_obj.parent) == base_dir and 
                    file_path_obj.stem == base_name and 
                    file_path_obj.suffix.lower() in shapefile_extensions):
                    
                    try:
                        zip_ref.extract(file_path, extract_dir)
                        shapefile_components.append(extract_dir / file_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract shapefile component {file_path}: {e}")
        
        return shapefile_components
    
    def _classify_extracted_files(self, files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Classify extracted files into raster and vector.
        
        Args:
            files: List of extracted file paths
            
        Returns:
            Tuple of (raster_files, vector_files)
        """
        raster_files = []
        vector_files = []
        
        for file_path in files:
            if file_path.suffix.lower() in self.raster_extensions:
                raster_files.append(file_path)
            elif file_path.suffix.lower() in self.vector_extensions:
                vector_files.append(file_path)
        
        return raster_files, vector_files
    
    def _get_archive_output_dir(self, archive_file: Path) -> Path:
        """
        Get output directory for a specific archive.
        
        Args:
            archive_file: Path to archive file
            
        Returns:
            Output directory path
        """
        # Create subdirectory based on archive name
        archive_name = archive_file.stem
        output_dir = self.extraction_config.output_dir / archive_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _create_visualizations(self) -> bool:
        """
        Create visualizations of clipped data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.clipped_files:
                self.logger.info("No clipped files to visualize")
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
                input_directory=self.extraction_config.output_dir,
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
        # Create visualization directory
        base_dir = Path("outputs/visualizations")
        viz_dir = base_dir / "extracted_archives"
        
        # Ensure directory exists
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        return viz_dir


# Convenience function for easy integration
def extract_and_clip_archives(
    zip_source: Union[Path, str], 
    aoi_file: Union[Path, str], 
    output_directory: Union[Path, str],
    create_visualizations: bool = True
) -> bool:
    """
    Convenience function to extract and clip archive data.
    
    Args:
        zip_source: ZIP file or directory containing ZIP files
        aoi_file: AOI file for clipping
        output_directory: Directory to save clipped data
        create_visualizations: Whether to create visualizations
        
    Returns:
        True if successful, False otherwise
    """
    try:
        zip_path = Path(zip_source)
        
        # Determine if it's a file or folder
        if zip_path.is_file():
            config = ExtractionConfig(
                zip_file=zip_path,
                aoi_file=Path(aoi_file),
                output_dir=Path(output_directory),
                create_visualizations=create_visualizations
            )
        else:
            config = ExtractionConfig(
                zip_folder=zip_path,
                aoi_file=Path(aoi_file),
                output_dir=Path(output_directory),
                create_visualizations=create_visualizations
            )
        
        # Create and run processor
        processor = ArchiveExtractionProcessor(config)
        result = processor.process()
        
        return result.success
        
    except Exception as e:
        logging.getLogger('geoworkflow.extraction').error(f"Archive extraction failed: {e}")
        return False