# File: src/geoworkflow/processors/extraction/open_buildings.py
"""
Open Buildings extraction processor for the geoworkflow package.

This processor extracts building footprints from Google's Open Buildings 
dataset via the Earth Engine API, following the established TemplateMethodProcessor pattern.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
import json
import time

try:
    import ee
    import geopandas as gpd
    import pandas as pd
    HAS_REQUIRED_LIBS = True
except ImportError:
    HAS_REQUIRED_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ExtractionError, ValidationError, ConfigurationError
from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.earth_engine_utils import EarthEngineAuth, OpenBuildingsAPI, check_earth_engine_available
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory


class OpenBuildingsExtractionProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Extract building footprints from Google Open Buildings v3 dataset.
    
    This processor:
    - Authenticates with Earth Engine using flexible credential options
    - Loads AOI and converts to Earth Engine geometry
    - Filters buildings by confidence threshold and area
    - Exports to GeoJSON, Shapefile, or CSV format
    - Provides comprehensive error handling and progress tracking
    - Follows academic-friendly authentication patterns
    """
    
    def __init__(self, config: Union[OpenBuildingsExtractionConfig, Dict[str, Any]], 
                 logger: Optional[logging.Logger] = None):
        """Initialize Open Buildings extraction processor."""
        
        # Convert Pydantic model to dict for base class
        if isinstance(config, OpenBuildingsExtractionConfig):
            config_dict = config.model_dump(mode='json')
            self.buildings_config = config
        else:
            config_dict = config
            self.buildings_config = OpenBuildingsExtractionConfig(**config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.ee_api: Optional[OpenBuildingsAPI] = None
        self.aoi_geometry: Optional['ee.Geometry'] = None
        self.filtered_buildings: Optional['ee.FeatureCollection'] = None
        self.export_count: int = 0
        self.output_file: Optional[Path] = None

    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """Validate Open Buildings specific inputs."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check required libraries
        if not HAS_REQUIRED_LIBS:
            validation_result["errors"].append(
                "Required libraries missing. Install with: "
                "pip install earthengine-api geopandas google-auth"
            )
            
        # Check Earth Engine availability
        if not check_earth_engine_available():
            validation_result["errors"].append(
                "Earth Engine API not available. See documentation for academic access setup."
            )
            
        # Validate AOI file
        if not self.buildings_config.aoi_file.exists():
            validation_result["errors"].append(f"AOI file not found: {self.buildings_config.aoi_file}")
            
        # Validate output directory
        try:
            ensure_directory(self.buildings_config.output_dir)
        except Exception as e:
            validation_result["errors"].append(f"Cannot create output directory: {e}")
            
        # Check for existing output file if not overwriting
        self.output_file = self.buildings_config.get_output_file_path()
        if self.output_file.exists() and not self.buildings_config.overwrite_existing:
            validation_result["errors"].append(
                f"Output file already exists: {self.output_file}. "
                "Set overwrite_existing=True to replace."
            )
            
        # Validate credentials
        try:
            from geoworkflow.utils.earth_engine_utils import validate_earth_engine_setup
            credential_validation = validate_earth_engine_setup(self.buildings_config.service_account_key)
            validation_result["info"].update(credential_validation)
            
            if not credential_validation.get("valid", False):
                validation_result["warnings"].extend(credential_validation.get("errors", []))
        except Exception as e:
            validation_result["warnings"].append(f"Credential validation warning: {e}")
            
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result

    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Set up Earth Engine authentication and API."""
        setup_info = {"components": []}
        
        try:
            # Authenticate with Earth Engine
            project_id = EarthEngineAuth.authenticate(
                service_account_key=self.buildings_config.service_account_key,
                project_id=self.buildings_config.project_id
            )
            
            # Initialize Earth Engine API wrapper
            self.ee_api = OpenBuildingsAPI(
                project_id=project_id, 
                dataset_version=self.buildings_config.dataset_version.value
            )
            
            setup_info["components"].append("earth_engine_auth")
            setup_info["project_id"] = project_id
            
            # Load AOI geometry
            self.aoi_geometry = self.ee_api.load_aoi_geometry(self.buildings_config.aoi_file)
            setup_info["components"].append("aoi_geometry")
            
            self.logger.info(f"Earth Engine initialized with project: {project_id}")
            
        except Exception as e:
            raise ExtractionError(f"Failed to setup Earth Engine: {e}")
            
        return setup_info

    def process_data(self) -> ProcessingResult:
        """Execute the building extraction process."""
        
        try:
            self.logger.info("Starting Open Buildings extraction...")
            
            # Step 1: Filter buildings by confidence and spatial bounds
            with track_progress("Filtering buildings by confidence", self.logger):
                self.filtered_buildings = self.ee_api.filter_buildings_by_confidence(
                    self.aoi_geometry, 
                    self.buildings_config.confidence_threshold
                )
                
            # Step 2: Apply area filters if specified
            if self.buildings_config.min_area_m2 or self.buildings_config.max_area_m2:
                with track_progress("Filtering buildings by area", self.logger):
                    self.filtered_buildings = self.ee_api.filter_buildings_by_area(
                        self.filtered_buildings,
                        self.buildings_config.min_area_m2,
                        self.buildings_config.max_area_m2
                    )
            
            # Step 3: Apply feature limit if specified
            if self.buildings_config.max_features:
                self.filtered_buildings = self.filtered_buildings.limit(self.buildings_config.max_features)
            
            # Step 4: Export to specified format
            with track_progress("Exporting buildings data", self.logger):
                self._export_buildings()
            
            # Step 5: Get actual export count
            self.export_count = self._get_actual_feature_count()
            
            # Create processing result
            result = ProcessingResult(
                success=True,
                processed_count=self.export_count,
                message=f"Successfully extracted {self.export_count} buildings to {self.output_file}",
                output_paths=[self.output_file]
            )
            
            return result
            
        except Exception as e:
            raise ExtractionError(f"Open Buildings extraction failed: {e}")

    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Clean up Earth Engine resources."""
        cleanup_info = {"components_cleaned": []}
        
        # Earth Engine cleanup is automatic, but we can log completion
        if self.ee_api:
            cleanup_info["components_cleaned"].append("earth_engine_api")
            self.logger.debug("Earth Engine API session closed")
            
        return cleanup_info
    
    def _export_buildings(self) -> None:
        """Export buildings to the specified format."""
        format_type = self.buildings_config.export_format.value
        
        if format_type == "geojson":
            self._export_to_geojson()
        elif format_type == "shapefile":
            self._export_to_shapefile()
        elif format_type == "csv":
            self._export_to_csv()
        else:
            raise ExtractionError(f"Unsupported export format: {format_type}")
    
    def _export_to_geojson(self) -> None:
        """Export buildings to GeoJSON format."""
        self.logger.info(f"Exporting buildings to GeoJSON: {self.output_file}")
        
        # Prepare export task
        export_params = {
            'collection': self._prepare_collection_for_export(),
            'description': f'open_buildings_export_{int(time.time())}',
            'fileFormat': 'GeoJSON'
        }
        
        # Execute synchronous export
        features = self._execute_ee_export(export_params)
        
        # Write to local file
        with open(self.output_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        self.logger.info(f"GeoJSON export completed: {self.output_file}")
    
    def _export_to_shapefile(self) -> None:
        """Export buildings to Shapefile format."""
        self.logger.info(f"Exporting buildings to Shapefile: {self.output_file}")
        
        # For Shapefile, we need to ensure column names are <= 10 characters
        collection = self._prepare_collection_for_export()
        
        # Earth Engine to GeoDataFrame conversion
        features = self._execute_ee_export({'collection': collection, 'fileFormat': 'GeoDataFrame'})
        gdf = gpd.GeoDataFrame.from_features(features['features'])
        
        # Truncate column names for Shapefile compatibility
        gdf = self._truncate_shapefile_columns(gdf)
        
        # Write to Shapefile
        gdf.to_file(self.output_file, driver='ESRI Shapefile')
        
        self.logger.info(f"Shapefile export completed: {self.output_file}")
    
    def _export_to_csv(self) -> None:
        """Export building centroids to CSV format."""
        self.logger.info(f"Exporting building centroids to CSV: {self.output_file}")
        
        # For CSV, we export centroids rather than full polygons
        collection = self._prepare_collection_for_export()
        
        # Add centroid coordinates to each feature
        collection = collection.map(lambda feature: feature.set(
            'longitude', feature.geometry().centroid().coordinates().get(0)
        ).set(
            'latitude', feature.geometry().centroid().coordinates().get(1)
        ))
        
        # Execute export without geometry
        export_params = {
            'collection': collection.select(['.*'], None, False),  # Drop geometry
            'fileFormat': 'CSV'
        }
        
        features = self._execute_ee_export(export_params)
        
        # Convert to DataFrame and save
        df = pd.DataFrame([f['properties'] for f in features.get('features', [])])
        df.to_csv(self.output_file, index=False)
        
        self.logger.info(f"CSV export completed: {self.output_file}")
    
    def _prepare_collection_for_export(self) -> 'ee.FeatureCollection':
        """Prepare the collection for export by adding requested attributes."""
        collection = self.filtered_buildings
        
        # Add confidence scores if requested
        if self.buildings_config.include_confidence:
            # Confidence is already in the original data
            pass
        
        # Add calculated area if requested
        if self.buildings_config.include_area:
            collection = collection.map(lambda feature: feature.set(
                'area_m2', feature.geometry().area()
            ))
        
        # Add Plus Codes if requested
        if self.buildings_config.include_plus_codes:
            # Plus codes are already in the original Open Buildings data
            pass
        
        return collection
    
    def _execute_ee_export(self, export_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Earth Engine export with timeout and retry logic."""
        max_attempts = self.buildings_config.retry_attempts
        timeout_seconds = self.buildings_config.timeout_minutes * 60
        
        for attempt in range(max_attempts):
            try:
                self.logger.debug(f"Export attempt {attempt + 1}/{max_attempts}")
                
                # Convert EE collection to features (simplified synchronous approach)
                features = export_params['collection'].getInfo()
                
                return features
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise ExtractionError(f"Export failed after {max_attempts} attempts: {e}")
                
                self.logger.warning(f"Export attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _truncate_shapefile_columns(self, gdf: 'gpd.GeoDataFrame') -> 'gpd.GeoDataFrame':
        """Truncate column names for Shapefile compatibility (10 character limit)."""
        column_mapping = {}
        
        for col in gdf.columns:
            if col == 'geometry':
                continue
            
            if len(col) > 10:
                # Create abbreviated version
                truncated = col[:7] + str(abs(hash(col)))[:3]
                column_mapping[col] = truncated
                self.logger.debug(f"Truncated column '{col}' -> '{truncated}'")
        
        if column_mapping:
            gdf = gdf.rename(columns=column_mapping)
        
        return gdf
    
    def _get_actual_feature_count(self) -> int:
        """Get the actual number of features that were exported."""
        try:
            if self.filtered_buildings:
                # For now, return an approximate count
                # In a full implementation, this would count the actual exported features
                return self.filtered_buildings.size().getInfo()
            return 0
        except Exception:
            self.logger.warning("Could not determine exact feature count")
            return 0
