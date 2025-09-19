# File: src/geoworkflow/processors/aoi/processor.py
"""
AOI (Area of Interest) processor for creating and managing areas of interest.

This processor transforms the legacy define_aoi.py script into a modern,
enhanced processor class using the Phase 2.1 infrastructure.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin, EnhancedProcessingResult
from geoworkflow.core.exceptions import ProcessingError, ValidationError, GeospatialError
from geoworkflow.schemas.config_models import AOIConfig
from geoworkflow.core.base import ProcessingResult


class AOIProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced AOI processor for creating Areas of Interest from administrative boundaries.
    
    This processor can:
    - Extract specific countries with optional buffering
    - Extract all countries and dissolve boundaries into single polygon
    - Apply buffers in kilometers using proper projection
    - List available countries for exploration
    """
    
    def __init__(self, config: AOIConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize AOI processor.
        
        Args:
            config: AOI configuration object
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        config_dict = config.model_dump(mode='json') if hasattr(config, 'to_dict') else config.model_dump(mode='json')
        super().__init__(config_dict, logger)
        
        # Store typed config for easier access
        self.aoi_config = config if isinstance(config, AOIConfig) else AOIConfig.from_dict(config_dict)
        
        # Processing state
        self.boundaries_gdf: Optional[gpd.GeoDataFrame] = None
        self.processed_gdf: Optional[gpd.GeoDataFrame] = None
        self.available_countries: List[str] = []
        
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate AOI-specific inputs and configuration.
        
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
                "GeoPandas and Shapely are required for AOI processing. "
                "Please install with: conda install geopandas"
            )
            validation_result["valid"] = False
            return validation_result
        
        # Validate input file exists
        if not self.aoi_config.input_file.exists():
            validation_result["errors"].append(
                f"Input boundaries file does not exist: {self.aoi_config.input_file}"
            )
            validation_result["valid"] = False
        # Validate output directory can be created (but don't require output file to exist)
        try:
            output_dir = self.aoi_config.output_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            validation_result["info"]["output_dir_validated"] = str(output_dir)
        except Exception as e:
            validation_result["errors"].append(
                f"Cannot create output directory: {e}"
            )
            validation_result["valid"] = False


        # Validate country specification
        if not self.aoi_config.use_all_countries:
            if not self.aoi_config.countries or len(self.aoi_config.countries) == 0:
                validation_result["errors"].append(
                    "Either 'countries' must be specified or 'use_all_countries' must be True"
                )
                validation_result["valid"] = False
        
        # Validate buffer distance
        if self.aoi_config.buffer_km < 0:
            validation_result["errors"].append(
                f"Buffer distance must be non-negative, got {self.aoi_config.buffer_km}"
            )
            validation_result["valid"] = False
        elif self.aoi_config.buffer_km > 1000:
            validation_result["warnings"].append(
                f"Large buffer distance specified: {self.aoi_config.buffer_km} km"
            )
        
        # Try to load and validate boundaries file
        if validation_result["valid"]:
            try:
                test_gdf = gpd.read_file(self.aoi_config.input_file)
                
                # Check if country column exists
                if self.aoi_config.country_name_column not in test_gdf.columns:
                    validation_result["errors"].append(
                        f"Column '{self.aoi_config.country_name_column}' not found in input file. "
                        f"Available columns: {', '.join(test_gdf.columns)}"
                    )
                    validation_result["valid"] = False
                else:
                    # Store available countries for validation
                    self.available_countries = test_gdf[self.aoi_config.country_name_column].unique().tolist()
                    validation_result["info"]["available_countries"] = len(self.available_countries)
                    
                    # Validate requested countries exist
                    if self.aoi_config.countries:
                        missing_countries = [
                            c for c in self.aoi_config.countries 
                            if c not in self.available_countries
                        ]
                        if missing_countries:
                            validation_result["errors"].append(
                                f"Countries not found in dataset: {', '.join(missing_countries)}. "
                                f"Available countries: {', '.join(sorted(self.available_countries))}"
                            )
                            validation_result["valid"] = False
                
                # Check CRS
                if test_gdf.crs is None:
                    validation_result["warnings"].append(
                        "Input file has no CRS defined. Assuming WGS84 (EPSG:4326)"
                    )
                
                validation_result["info"]["input_crs"] = str(test_gdf.crs) if test_gdf.crs else "None"
                validation_result["info"]["feature_count"] = len(test_gdf)
                
            except Exception as e:
                validation_result["errors"].append(f"Error reading input file: {str(e)}")
                validation_result["valid"] = False
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths that must exist."""
        return ["input_file"]  # Only validate input paths, not outputs
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        if self.aoi_config.countries:
            return len(self.aoi_config.countries) + 2  # countries + buffer + save
        return 3  # load + process + save
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup AOI-specific processing resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Create output directory
        output_dir = self.aoi_config.output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_info["output_dir_created"] = str(output_dir)
        
        # Load boundaries
        try:
            self.log_processing_step("Loading administrative boundaries")
            self.boundaries_gdf = gpd.read_file(self.aoi_config.input_file)
            
            # Set CRS if not defined
            if self.boundaries_gdf.crs is None:
                self.logger.warning("Setting CRS to WGS84 (EPSG:4326)")
                self.boundaries_gdf.set_crs("EPSG:4326", inplace=True)
            
            setup_info["boundaries_loaded"] = True
            setup_info["feature_count"] = len(self.boundaries_gdf)
            setup_info["input_crs"] = str(self.boundaries_gdf.crs)
            
            self.add_metric("boundaries_loaded", len(self.boundaries_gdf))
            
        except Exception as e:
            raise ProcessingError(f"Failed to load boundaries file: {str(e)}")
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the main AOI processing logic.
        
        Returns:
            ProcessingResult with processing outcomes
        """
        result = ProcessingResult(success=True)
        
        try:
            # Step 1: Filter countries or use all
            if self.aoi_config.use_all_countries:
                self.log_processing_step("Processing all countries")
                filtered_gdf = self.boundaries_gdf.copy()
                self.add_metric("countries_processed", len(self.available_countries))
            else:
                self.log_processing_step(f"Filtering {len(self.aoi_config.countries)} countries")
                filtered_gdf = self.boundaries_gdf[
                    self.boundaries_gdf[self.aoi_config.country_name_column].isin(self.aoi_config.countries)
                ].copy()
                self.add_metric("countries_processed", len(self.aoi_config.countries))
            
            if filtered_gdf.empty:
                raise ProcessingError("No countries were extracted after filtering")
            
            self.update_progress(1, f"Filtered to {len(filtered_gdf)} features")
            
            # Step 2: Dissolve boundaries if requested
            if self.aoi_config.dissolve_boundaries:
                self.log_processing_step("Dissolving country boundaries into single polygon")
                # Use unary_union to combine geometries, then create new GeoDataFrame
                dissolved_geom = filtered_gdf.geometry.unary_union
                dissolved_gdf = gpd.GeoDataFrame(
                    geometry=[dissolved_geom], 
                    crs=filtered_gdf.crs
                )
                self.processed_gdf = dissolved_gdf
                self.add_metric("boundaries_dissolved", True)
            else:
                self.processed_gdf = filtered_gdf
                self.add_metric("boundaries_dissolved", False)
            
            self.update_progress(1, "Boundary processing complete")
            
            # Step 3: Apply buffer if requested
            if self.aoi_config.buffer_km > 0:
                self.log_processing_step(f"Applying {self.aoi_config.buffer_km} km buffer")
                
                # Warning for users about Africa-specific projection
                self.logger.warning(
                    "Using Africa Albers Equal Area Conic (ESRI:102022) for buffering operations. "
                    "This projection is optimized for African datasets and may not be accurate "
                    "for other continents. For non-African data, consider using a more appropriate "
                    "projected coordinate system (e.g., UTM zones or continental equal-area projections)."
                )
                
                # Africa Albers Equal Area Conic - optimized for African continent
                africa_albers = "ESRI:102022"
                original_crs = self.processed_gdf.crs
                
                try:
                    # Reproject to projected CRS for accurate buffering
                    self.logger.info(f"Reprojecting from {original_crs} to {africa_albers} for buffering")
                    gdf_projected = self.processed_gdf.to_crs(africa_albers)
                    
                    # Apply buffer in meters
                    buffer_distance_m = self.aoi_config.buffer_km * 1000
                    self.logger.info(f"Applying {self.aoi_config.buffer_km}km ({buffer_distance_m}m) buffer")
                    gdf_projected.geometry = gdf_projected.geometry.buffer(buffer_distance_m)
                    
                    # Reproject back to original CRS
                    self.logger.info(f"Reprojecting back to {original_crs}")
                    self.processed_gdf = gdf_projected.to_crs(original_crs)
                    
                except Exception as e:
                    self.logger.error(f"CRS transformation failed: {e}")
                    self.logger.warning("Falling back to degree-based buffering (less accurate)")
                    # Fallback to original degree-based buffering with conversion warning
                    buffer_distance_deg = self.aoi_config.buffer_km / 111.32  # More accurate conversion
                    self.logger.warning(f"Using approximate conversion: {self.aoi_config.buffer_km}km ≈ {buffer_distance_deg:.4f} degrees")
                    self.processed_gdf.geometry = self.processed_gdf.geometry.buffer(buffer_distance_deg)
                
                self.add_metric("buffer_applied_km", self.aoi_config.buffer_km)

            self.update_progress(1, "Buffer application complete")
            
            # Step 4: Save result
            self.log_processing_step(f"Saving AOI to {self.aoi_config.output_file}")
            self.processed_gdf.to_file(self.aoi_config.output_file, driver="GeoJSON")
            
            # Verify output file was created
            if not self.aoi_config.output_file.exists():
                raise ProcessingError(f"Output file was not created: {self.aoi_config.output_file}")
            
            self.update_progress(1, "AOI saved successfully")
            
            # Update result
            result.processed_count = 1
            result.message = f"Successfully created AOI with {len(self.processed_gdf)} features"
            result.add_output_path(self.aoi_config.output_file)
            
            # Add processing metadata
            result.metadata = {
                "output_file": str(self.aoi_config.output_file),
                "feature_count": len(self.processed_gdf),
                "countries_processed": len(self.aoi_config.countries) if self.aoi_config.countries else len(self.available_countries),
                "buffer_applied_km": self.aoi_config.buffer_km,
                "boundaries_dissolved": self.aoi_config.dissolve_boundaries,
                "output_crs": str(self.processed_gdf.crs)
            }
            
            self.add_metric("output_features", len(self.processed_gdf))


            # Store bounds and area for later access (before cleanup)
            self._aoi_bounds = self.processed_gdf.total_bounds.tolist()
            
            # Calculate area using proper CRS handling (similar to buffering logic)
            if self.processed_gdf.crs and str(self.processed_gdf.crs) == "EPSG:4326":
                # Use Africa Albers Equal Area Conic for accurate area calculation
                africa_albers = "ESRI:102022"
                original_crs = self.processed_gdf.crs
                
                try:
                    # Reproject to projected CRS for accurate area calculation
                    self.logger.debug(f"Calculating area using {africa_albers} projection")
                    gdf_projected = self.processed_gdf.to_crs(africa_albers)
                    area_m2 = gdf_projected.geometry.area.sum()
                    self._aoi_area_km2 = area_m2 / 1_000_000  # Convert to km²
                    
                    self.logger.debug(f"Area calculated: {self._aoi_area_km2:.2f} km²")
                    
                except Exception as e:
                    self.logger.warning(f"CRS transformation failed for area calculation: {e}")
                    self.logger.warning("Falling back to approximate degree-based area calculation")
                    
                    # Fallback: suppress the warning and use approximate calculation
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        area_deg2 = self.processed_gdf.geometry.area.sum()
                    
                    # More accurate approximation than the previous 12400 factor
                    # This varies by latitude, but 12100 km²/deg² is reasonable for Africa
                    self._aoi_area_km2 = area_deg2 * 12100
                    
            else:
                # Already in projected coordinates
                area_m2 = self.processed_gdf.geometry.area.sum()
                self._aoi_area_km2 = area_m2 / 1_000_000

        except Exception as e:
            result.success = False
            result.message = f"AOI processing failed: {str(e)}"
            self.logger.error(f"AOI processing failed: {e}")
            raise ProcessingError(f"AOI processing failed: {str(e)}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup AOI-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Clear large dataframes from memory
        if self.boundaries_gdf is not None:
            del self.boundaries_gdf
            cleanup_info["boundaries_gdf_cleared"] = True
        
        if self.processed_gdf is not None:
            del self.processed_gdf
            cleanup_info["processed_gdf_cleared"] = True
        
        cleanup_info["aoi_cleanup"] = "completed"
        
        return cleanup_info
    
    def list_available_countries(self, prefix: Optional[str] = None) -> List[str]:
        """
        List available countries in the boundaries file.
        
        Args:
            prefix: Optional prefix to filter countries
            
        Returns:
            List of available country names
        """
        if self.boundaries_gdf is None or self.boundaries_gdf.empty:  # FIXED
            # Load boundaries if not already loaded
            self.boundaries_gdf = gpd.read_file(self.aoi_config.input_file)
        
        countries = self.boundaries_gdf[self.aoi_config.country_name_column].unique().tolist()
        
        if prefix:
            countries = [c for c in countries if c.lower().startswith(prefix.lower())]
        
        return sorted(countries)
    
    def get_aoi_bounds(self) -> Optional[tuple]:
        return getattr(self, '_aoi_bounds', None)

    def get_aoi_area_km2(self) -> Optional[float]:
        return getattr(self, '_aoi_area_km2', None)


# Factory function for easy creation
def create_aoi_processor(config_path: Path) -> AOIProcessor:
    """
    Factory function to create AOI processor from configuration file.
    
    Args:
        config_path: Path to AOI configuration file
        
    Returns:
        Configured AOI processor instance
    """
    config = AOIConfig.from_file(config_path)
    return AOIProcessor(config)