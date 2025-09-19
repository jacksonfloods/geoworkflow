# File: src/geoworkflow/processors/integration/enrichment.py
"""
Statistical enrichment processor for the geoworkflow package.

This processor enriches city vector data (Cities of Interest - COI) with 
statistical summaries computed from raster datasets using zonal statistics.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging
import fnmatch

try:
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import rasterio
    from rasterstats import zonal_stats
    import pyproj
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ProcessingError, ValidationError, GeospatialError
from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory


class StatisticalEnrichmentProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced statistical enrichment processor for COI data.
    
    This processor:
    - Identifies a single Cities of Interest (COI) vector file
    - Computes zonal statistics from multiple raster datasets
    - Enriches the COI data with new statistical columns
    - Handles coordinate system compatibility and data validation
    - Provides comprehensive error handling and progress tracking
    """
    
    def __init__(self, config: Union[StatisticalEnrichmentConfig, Dict[str, Any]], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize statistical enrichment processor.
        
        Args:
            config: Enrichment configuration object or dictionary
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        if isinstance(config, StatisticalEnrichmentConfig):
            config_dict = config.model_dump(mode='json')
            self.enrichment_config = config
        else:
            config_dict = config
            self.enrichment_config = StatisticalEnrichmentConfig.from_dict(config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.coi_file: Optional[Path] = None
        self.coi_gdf: Optional[gpd.GeoDataFrame] = None
        self.raster_files: List[Path] = []
        self.statistics_computed: Dict[str, Dict[str, Any]] = {}
        self.enriched_gdf: Optional[gpd.GeoDataFrame] = None
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate enrichment-specific inputs and configuration.
        
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
                "geopandas, rasterio, rasterstats"
            )
            validation_result["valid"] = False
            return validation_result
        
        # Validate COI directory exists
        if not self.enrichment_config.coi_directory.exists():
            validation_result["errors"].append(
                f"COI directory does not exist: {self.enrichment_config.coi_directory}"
            )
            validation_result["valid"] = False
        
        # Validate raster directory exists
        if not self.enrichment_config.raster_directory.exists():
            validation_result["errors"].append(
                f"Raster directory does not exist: {self.enrichment_config.raster_directory}"
            )
            validation_result["valid"] = False
        
        # Validate output directory can be created
        try:
            ensure_directory(self.enrichment_config.output_file.parent)
            validation_result["info"]["output_dir_validated"] = str(self.enrichment_config.output_file.parent)
        except Exception as e:
            validation_result["errors"].append(
                f"Cannot create output directory: {e}"
            )
            validation_result["valid"] = False
        
        # Check if output file exists and skip_existing is True
        if (self.enrichment_config.skip_existing and 
            self.enrichment_config.output_file.exists()):
            validation_result["warnings"].append(
                f"Output file already exists and will be skipped: {self.enrichment_config.output_file}"
            )
            validation_result["info"]["will_skip"] = True
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths that must exist."""
        return ["coi_directory", "raster_directory"]
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        try:
            # COI discovery + raster discovery + statistics computation
            base_items = 2
            
            # Try to count raster files for better estimation
            raster_files = self._discover_raster_files()
            return base_items + len(raster_files)
        except:
            return 10  # Fallback estimate
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup enrichment-specific processing resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Discover and validate COI file
        self.log_processing_step("Discovering COI file")
        try:
            self.coi_file = self._discover_coi_file()
            setup_info["coi_file_found"] = str(self.coi_file)
            
            # Load COI data
            self.coi_gdf = gpd.read_file(self.coi_file)
            
            # Set CRS if not defined
            if self.coi_gdf.crs is None:
                self.logger.warning("COI file has no CRS defined, assuming EPSG:4326")
                self.coi_gdf.set_crs("EPSG:4326", inplace=True)
            
            setup_info["coi_crs"] = str(self.coi_gdf.crs)
            setup_info["coi_features"] = len(self.coi_gdf)
            
            self.add_metric("coi_features_loaded", len(self.coi_gdf))
            
        except Exception as e:
            raise ProcessingError(f"Failed to discover or load COI file: {str(e)}")
        
        # Discover raster files
        self.log_processing_step("Discovering raster files")
        self.raster_files = self._discover_raster_files()
        
        if not self.raster_files:
            raise ProcessingError("No raster files found matching the specified pattern")
        
        setup_info["raster_files_found"] = len(self.raster_files)
        self.add_metric("raster_files_discovered", len(self.raster_files))
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the main statistical enrichment logic.
        
        Returns:
            ProcessingResult with enrichment outcomes
        """
        result = ProcessingResult(success=True)
        
        try:
            # Check if we should skip processing
            if (self.enrichment_config.skip_existing and 
                self.enrichment_config.output_file.exists()):
                result.message = f"Skipping - output file already exists: {self.enrichment_config.output_file}"
                result.skipped_count = 1
                return result
            
            # Start with a copy of the original COI data
            self.enriched_gdf = self.coi_gdf.copy()
            
            self.log_processing_step(f"Computing statistics for {len(self.raster_files)} raster files")
            
            # Process each raster file
            for raster_file in track_progress(
                self.raster_files,
                description="Computing zonal statistics",
                quiet=False
            ):
                success = self._compute_raster_statistics(raster_file)
                
                if success:
                    result.processed_count += 1
                else:
                    result.failed_count += 1
                    result.add_failed_file(raster_file)
                
                self.update_progress(1, f"Processed {raster_file.name}")
            
            # Add area column if requested
            if self.enrichment_config.add_area_column:
                self._add_area_column()
                self.log_processing_step("Added polygon area column")
            
            # Save enriched data
            self.log_processing_step(f"Saving enriched COI data to {self.enrichment_config.output_file}")
            self._save_enriched_data()
            
            # Update result
            new_columns = len(self.enriched_gdf.columns) - len(self.coi_gdf.columns)
            result.message = f"Successfully enriched COI data with {new_columns} new statistical columns"
            result.add_output_path(self.enrichment_config.output_file)
            
            # Add processing metadata
            result.metadata = {
                "coi_file": str(self.coi_file),
                "output_file": str(self.enrichment_config.output_file),
                "original_features": len(self.coi_gdf),
                "original_columns": len(self.coi_gdf.columns),
                "enriched_columns": len(self.enriched_gdf.columns),
                "new_columns_added": new_columns,
                "raster_files_processed": result.processed_count,
                "statistics_computed": list(self.enrichment_config.statistics)
            }
            
            self.add_metric("statistical_columns_added", new_columns)
            self.add_metric("enrichment_successful", True)
            
        except Exception as e:
            result.success = False
            result.message = f"Statistical enrichment failed: {str(e)}"
            self.logger.error(f"Statistical enrichment failed: {e}")
            raise GeospatialError(f"Statistical enrichment failed: {str(e)}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup enrichment-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Clear data from memory
        if hasattr(self, 'coi_gdf') and self.coi_gdf is not None:
            del self.coi_gdf
            cleanup_info["coi_gdf_cleared"] = True
        
        if hasattr(self, 'enriched_gdf') and self.enriched_gdf is not None:
            del self.enriched_gdf
            cleanup_info["enriched_gdf_cleared"] = True
        
        cleanup_info["enrichment_cleanup"] = "completed"
        
        return cleanup_info
    
    def _discover_coi_file(self) -> Path:
        """
        Discover the COI file using the specified pattern.
        
        Returns:
            Path to the COI file
            
        Raises:
            ProcessingError: If no file or multiple files found
        """
        coi_dir = self.enrichment_config.coi_directory
        pattern = self.enrichment_config.coi_pattern
        
        # Find all files matching the pattern
        matching_files = []
        
        for file_path in coi_dir.rglob("*"):
            if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                # Check if it's a vector file
                if file_path.suffix.lower() in {'.shp', '.geojson', '.gpkg', '.gml', '.kml'}:
                    matching_files.append(file_path)
        
        if len(matching_files) == 0:
            raise ProcessingError(
                f"No COI files found matching pattern '{pattern}' in directory {coi_dir}"
            )
        elif len(matching_files) > 1:
            raise ProcessingError(
                f"Multiple COI files found matching pattern '{pattern}' in directory {coi_dir}: "
                f"{[str(f) for f in matching_files]}. Please use a more specific pattern."
            )
        
        coi_file = matching_files[0]
        self.logger.info(f"Found COI file: {coi_file}")
        return coi_file
    
    def _discover_raster_files(self) -> List[Path]:
        """
        Discover all raster files to process.
        
        Returns:
            List of raster file paths
        """
        raster_files = []
        raster_dir = self.enrichment_config.raster_directory
        pattern = self.enrichment_config.raster_pattern
        
        # Define raster extensions
        raster_extensions = {'.tif', '.tiff', '.geotif', '.geotiff'}
        
        if self.enrichment_config.recursive:
            search_pattern = f"**/{pattern}"
        else:
            search_pattern = pattern
        
        # Find matching files
        for file_path in raster_dir.glob(search_pattern):
            if file_path.is_file() and file_path.suffix.lower() in raster_extensions:
                raster_files.append(file_path)
        
        return sorted(raster_files)
    
    def _compute_raster_statistics(self, raster_path: Path) -> bool:
        """
        Compute zonal statistics for a single raster.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.debug(f"Computing statistics for: {raster_path.name}")
            
            # Get dataset name from filename (remove extension)
            dataset_name = raster_path.stem
            
            # Clean dataset name for column naming
            dataset_name = self._clean_dataset_name(dataset_name)
            
            # Prepare geometries for zonal stats
            geometries = self.coi_gdf.geometry
            
            # Check CRS compatibility and reproject if needed
            coi_crs = self.coi_gdf.crs
            
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                
                if raster_crs and coi_crs and str(raster_crs) != str(coi_crs):
                    self.logger.debug(f"Reprojecting COI from {coi_crs} to {raster_crs}")
                    geometries = self.coi_gdf.to_crs(raster_crs).geometry
            
            # Compute zonal statistics
            stats_list = zonal_stats(
                geometries,
                str(raster_path),
                stats=self.enrichment_config.statistics,
                nodata=None,  # Let rasterstats handle nodata from raster
                all_touched=False
            )
            
            # Add statistics as new columns
            for stat_name in self.enrichment_config.statistics:
                column_name = f"{dataset_name}_{stat_name}"
                stat_values = [stats.get(stat_name) if stats else None for stats in stats_list]
                self.enriched_gdf[column_name] = stat_values
            
            self.statistics_computed[dataset_name] = {
                "raster_file": str(raster_path),
                "statistics": self.enrichment_config.statistics,
                "features_processed": len(stats_list)
            }
            
            self.logger.debug(f"Successfully computed statistics for {dataset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error computing statistics for {raster_path}: {e}")
            return False
    
    def _clean_dataset_name(self, name: str) -> str:
        """
        Clean dataset name for use in column names.
        
        Args:
            name: Raw dataset name
            
        Returns:
            Cleaned name suitable for column naming
        """
        # Remove common suffixes
        name = name.lower()
        
        # Remove file extensions that might be in the stem
        for ext in ['.tif', '.tiff', '.geotif', '.geotiff']:
            name = name.replace(ext, '')
        
        # Replace non-alphanumeric characters with underscores
        import re
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure name doesn't start with a number (invalid column name)
        if name and name[0].isdigit():
            name = f"data_{name}"
        
        return name
    
    def _add_area_column(self):
        """Add polygon area column to the enriched data."""
        if self.enriched_gdf.crs and self.enriched_gdf.crs.is_geographic:
            # For geographic CRS, reproject to equal-area projection
            # Use Africa Albers Equal Area Conic
            equal_area_crs = "ESRI:102022"
            try:
                area_gdf = self.enriched_gdf.to_crs(equal_area_crs)
                areas_m2 = area_gdf.geometry.area
            except Exception as e:
                self.logger.warning(f"Could not reproject for area calculation: {e}")
                # Fallback to simple area calculation (less accurate)
                areas_m2 = self.enriched_gdf.geometry.area * 111320 * 111320  # rough conversion
        else:
            # Assume projected CRS in meters
            areas_m2 = self.enriched_gdf.geometry.area
        
        # Convert to requested units
        if self.enrichment_config.area_units == "km2":
            areas = areas_m2 / 1_000_000
            column_name = "area_km2"
        else:  # m2
            areas = areas_m2
            column_name = "area_m2"
        
        self.enriched_gdf[column_name] = areas
        self.logger.debug(f"Added area column: {column_name}")
    
    def _save_enriched_data(self):
        """Save the enriched COI data to file."""
        output_file = self.enrichment_config.output_file
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine output format from file extension
        if output_file.suffix.lower() == '.geojson':
            driver = 'GeoJSON'
        elif output_file.suffix.lower() == '.shp':
            driver = 'ESRI Shapefile'
        elif output_file.suffix.lower() == '.gpkg':
            driver = 'GPKG'
        else:
            # Default to GeoJSON
            driver = 'GeoJSON'
            self.logger.warning(f"Unknown file extension, using GeoJSON format")
        
        # Save enriched data
        self.enriched_gdf.to_file(output_file, driver=driver)
        
        self.logger.info(f"Saved enriched COI data to: {output_file}")
        self.logger.info(f"Original features: {len(self.coi_gdf)}")
        self.logger.info(f"Original columns: {len(self.coi_gdf.columns)}")
        self.logger.info(f"Enriched columns: {len(self.enriched_gdf.columns)}")
        
        # Log summary of new columns
        original_columns = set(self.coi_gdf.columns)
        new_columns = [col for col in self.enriched_gdf.columns if col not in original_columns]
        
        if new_columns:
            self.logger.info(f"New columns added: {', '.join(new_columns)}")


# Convenience function for easy integration
def enrich_cities_with_statistics(
    coi_directory: Union[str, Path],
    raster_directory: Union[str, Path], 
    output_file: Union[str, Path],
    coi_pattern: str = "*AFRICAPOLIS*",
    raster_pattern: str = "*.tif",
    statistics: List[str] = None
) -> bool:
    """
    Convenience function to enrich cities with raster statistics.
    
    Args:
        coi_directory: Directory containing COI file
        raster_directory: Directory containing raster files
        output_file: Output file for enriched data
        coi_pattern: Pattern to identify COI file
        raster_pattern: Pattern for raster files
        statistics: Statistics to compute
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if statistics is None:
            statistics = ["mean", "max", "min"]
        
        # Create enrichment configuration
        config = StatisticalEnrichmentConfig(
            coi_directory=Path(coi_directory),
            raster_directory=Path(raster_directory),
            output_file=Path(output_file),
            coi_pattern=coi_pattern,
            raster_pattern=raster_pattern,
            statistics=statistics
        )
        
        # Create and run processor
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        return result.success
        
    except Exception as e:
        logging.getLogger('geoworkflow.enrichment').error(f"Statistical enrichment failed: {e}")
        return False