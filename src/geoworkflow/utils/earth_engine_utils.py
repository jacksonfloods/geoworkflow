#/src/geoworkflow/utils/earth_engine_utils.py
"""
Earth Engine utilities for the geoworkflow package.

This module provides Earth Engine integration with academic-friendly
authentication and error handling patterns.
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from shapely.geometry import box
import geopandas as gpd

# Additional imports for CSV export
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Shapely for geometry operations
try:
    from shapely.geometry import mapping
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    mapping = None


# Proper conditional imports for Earth Engine
HAS_EARTH_ENGINE = False
EARTH_ENGINE_ERROR = None

try:
    import ee
    HAS_EARTH_ENGINE = True
except ImportError as e:
    EARTH_ENGINE_ERROR = str(e)
    # Only import for type checking, not runtime
    if TYPE_CHECKING:
        import ee
    else:
        # Runtime stubs that will never be used when EE is unavailable
        ee = None


# Import our exceptions and constants
from ..core.exceptions import (
    EarthEngineError, 
    EarthEngineAuthenticationError,
    EarthEngineQuotaError,
    EarthEngineTimeoutError,
    EarthEngineGeometryError
)
from ..core.constants import (
    EARTH_ENGINE_DATASETS, DEFAULT_EARTH_ENGINE_TIMEOUT, DEFAULT_BUILDING_CONFIDENCE,
    DEFAULT_MIN_BUILDING_AREA,DEFAULT_MAX_BUILDING_AREA, DEFAULT_EE_RETRY_ATTEMPTS,
    DEFAULT_EE_RETRY_DELAY, EE_ERROR_PATTERNS
)
# Add after existing imports
from geoworkflow.core.constants import (
    EARTH_ENGINE_DATASETS, MAX_EE_FEATURES_PER_REQUEST,
    GRID_EXPORT_THRESHOLD, BASE_GRID_SIZE_M, SUBDIVISION_THRESHOLD,
    MAX_FEATURES_PER_CELL, MIN_CELL_SIZE_M, MAX_SUBDIVISION_DEPTH,
    DEFAULT_GRID_WORKERS, GRID_CRS_METRIC, GRID_PROGRESS_UPDATE_INTERVAL
)

logger = logging.getLogger(__name__)

# Type aliases for better code clarity and robustness
if TYPE_CHECKING:
    EEGeometry = ee.Geometry
    EEFeatureCollection = ee.FeatureCollection
    EETask = ee.batch.Task
else:
    EEGeometry = Any
    EEFeatureCollection = Any
    EETask = Any


def check_earth_engine_available() -> bool:
    """
    Check if Earth Engine is available and provide helpful error messages.
    
    Returns:
        True if Earth Engine is available, False otherwise
    """
    if not HAS_EARTH_ENGINE:
        logger.error(
            f"Earth Engine API not available: {EARTH_ENGINE_ERROR}\n"
            "Install with: pip install geoworkflow[earth-engine]\n"
            "Or manually: pip install earthengine-api google-auth google-cloud-storage"
        )
        return False
    return True


def validate_earth_engine_setup(service_account_key: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validate Earth Engine setup with academic-friendly error messages.
    
    Args:
        service_account_key: Optional path to service account key
        
    Returns:
        Dictionary with validation results and helpful guidance
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
        "setup_guidance": []
    }
    
    # Check if Earth Engine is available
    if not check_earth_engine_available():
        validation_result["valid"] = False
        validation_result["errors"].append("Earth Engine API not available")
        validation_result["setup_guidance"].append(
            "Install Earth Engine: pip install geoworkflow[earth-engine]"
        )
        return validation_result
    
    # Check authentication options
    auth_methods = []
    
    # Check service account key
    if service_account_key:
        if service_account_key.exists():
            auth_methods.append("service_account_key")
            validation_result["info"]["service_account_key"] = str(service_account_key)
        else:
            validation_result["errors"].append(f"Service account key not found: {service_account_key}")
            validation_result["valid"] = False
    
    # Check environment variable
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        auth_methods.append("environment_variable")
        validation_result["info"]["environment_credentials"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Check default location
    default_cred_path = Path.home() / ".config" / "earthengine" / "credentials"
    if default_cred_path.exists():
        auth_methods.append("default_credentials")
        validation_result["info"]["default_credentials"] = str(default_cred_path)
    
    # Validate authentication setup
    if not auth_methods:
        validation_result["valid"] = False
        validation_result["errors"].append("No Earth Engine authentication found")
        validation_result["setup_guidance"].extend([
            "Set up Earth Engine authentication:",
            "1. For service account: provide service_account_key path",
            "2. For user auth: run 'earthengine authenticate'", 
            "3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
        ])
    else:
        validation_result["info"]["available_auth_methods"] = auth_methods
    
    return validation_result


class EarthEngineAuth:
    """Handle Earth Engine authentication with multiple credential sources."""
    
    @staticmethod
    def authenticate(service_account_key: Optional[Path] = None, 
                    service_account_email: Optional[str] = None,
                    project_id: Optional[str] = None) -> str:
        """
        Authenticate with Earth Engine using academic-friendly credential discovery.
        
        Args:
            service_account_key: Optional path to service account key
            service_account_email: Service account email (required with service_account_key)
            project_id: Optional Google Cloud project ID
            
        Returns:
            Project ID for Earth Engine initialization
            
        Raises:
            EarthEngineAuthenticationError: If authentication fails with helpful guidance
        """
        if not check_earth_engine_available():
            raise EarthEngineAuthenticationError("Earth Engine API not available")
        
        try:
            # Method 1: Service account key file
            if service_account_key and service_account_key.exists():
                if not service_account_email:
                    raise EarthEngineAuthenticationError(
                        "service_account_email is required when using service account key file"
                    )
                
                logger.info(f"Using service account: {service_account_email}")
                
                # Extract project ID from service account if not provided
                if not project_id:
                    project_id = EarthEngineAuth.get_project_id_from_service_account(service_account_key)
                
                # Initialize with service account
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_email,  # Now explicitly required
                    key_file=str(service_account_key)
                )
                ee.Initialize(credentials, project=project_id)
                
                logger.info(f"Earth Engine authenticated with service account, project: {project_id}")
                return project_id
            
            
            # Method 2: Environment variable
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                cred_path = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
                logger.info(f"Using credentials from environment variable: {cred_path}")
                
                if not project_id:
                    project_id = EarthEngineAuth.get_project_id_from_service_account(cred_path)
                
                ee.Initialize(project=project_id)
                logger.info(f"Earth Engine authenticated with environment credentials, project: {project_id}")
                return project_id
            
            # Method 3: Default credentials (user authentication)
            else:
                logger.info("Attempting to use default Earth Engine credentials")
                
                # For user credentials, project_id is often required
                if not project_id:
                    # Try to initialize without project first
                    try:
                        ee.Initialize()
                        # If successful, try to get default project
                        project_id = ee.data.getAssetRoots()[0]['id'] if ee.data.getAssetRoots() else "earthengine-legacy"
                    except Exception:
                        raise EarthEngineAuthenticationError(
                            "Default authentication failed and no project_id provided. "
                            "Please provide project_id or use service account authentication."
                        )
                else:
                    ee.Initialize(project=project_id)
                
                logger.info(f"Earth Engine authenticated with default credentials, project: {project_id}")
                return project_id
                
        except Exception as e:
            error_message = str(e).lower()
            
            # Provide specific guidance based on error type
            if any(pattern in error_message for pattern in EE_ERROR_PATTERNS['authentication']):
                raise EarthEngineAuthenticationError(
                    "Earth Engine authentication failed. For academic access:\n"
                    "1. Sign up at: https://earthengine.google.com/signup/\n"
                    "2. Create service account: https://developers.google.com/earth-engine/guides/service_account\n"
                    "3. Set service_account_key in config or GOOGLE_APPLICATION_CREDENTIALS env var\n"
                    f"Original error: {e}"
                )
            elif any(pattern in error_message for pattern in EE_ERROR_PATTERNS['quota']):
                raise EarthEngineQuotaError(
                    "Earth Engine quota exceeded. Try:\n"
                    "1. Reduce AOI size or increase confidence_threshold\n"
                    "2. Set max_features to limit results\n"
                    "3. Wait and retry (quotas reset daily)\n"
                    f"Original error: {e}"
                )
            else:
                raise EarthEngineAuthenticationError(f"Earth Engine authentication failed: {e}")
    
    @staticmethod
    def get_project_id_from_service_account(key_path: Path) -> str:
        """
        Extract project ID from service account key file.
        
        Args:
            key_path: Path to service account JSON key file
            
        Returns:
            Project ID from the service account
            
        Raises:
            EarthEngineAuthenticationError: If project ID cannot be extracted
        """
        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
            
            project_id = key_data.get('project_id')
            if not project_id:
                raise EarthEngineAuthenticationError(
                    f"No project_id found in service account key: {key_path}"
                )
            
            logger.debug(f"Extracted project_id '{project_id}' from service account key")
            return project_id
            
        except json.JSONDecodeError as e:
            raise EarthEngineAuthenticationError(
                f"Invalid JSON in service account key file {key_path}: {e}"
            )
        except Exception as e:
            raise EarthEngineAuthenticationError(
                f"Failed to read service account key {key_path}: {e}"
            )


class OpenBuildingsAPI:
    """Wrapper for Earth Engine Open Buildings operations."""
    
    def __init__(self, project_id: str, dataset_version: str = "v3"):
        """
        Initialize the Open Buildings API wrapper.
        
        Args:
            project_id: Google Cloud project ID
            dataset_version: Open Buildings dataset version
            
        Raises:
            EarthEngineError: If Earth Engine is not available or dataset is unsupported
        """
        if not check_earth_engine_available():
            raise EarthEngineError("Earth Engine API not available")
        
        self.project_id = project_id
        self.dataset_version = dataset_version
        
        # Get dataset collection path
        dataset_key = f'open_buildings_{dataset_version}'
        if dataset_key not in EARTH_ENGINE_DATASETS:
            raise EarthEngineError(f"Unsupported dataset version: {dataset_version}")
        
        self.collection_path = EARTH_ENGINE_DATASETS[dataset_key]
        self.collection = ee.FeatureCollection(self.collection_path)
        
        logger.info(f"Initialized Open Buildings API for dataset: {self.collection_path}")

    def _truncate_shapefile_columns(self, gdf: 'gpd.GeoDataFrame') -> 'gpd.GeoDataFrame':
        """Truncate column names for Shapefile compatibility (10 char limit)."""
        column_mapping = {}
        for col in gdf.columns:
            if col != 'geometry' and len(col) > 10:
                # Truncate to 10 characters
                new_col = col[:10]
                # Handle duplicates by adding numbers
                counter = 1
                while new_col in column_mapping.values():
                    new_col = col[:8] + f"{counter:02d}"
                    counter += 1
                column_mapping[col] = new_col
        
        if column_mapping:
            gdf = gdf.rename(columns=column_mapping)
            logger.info(f"Truncated {len(column_mapping)} column names for Shapefile compatibility")
        
        return gdf

    def load_aoi_geometry(self, aoi_file: Path) -> EEGeometry:
        """
        Load AOI from file and convert to Earth Engine geometry.
        
        Args:
            aoi_file: Path to AOI file (GeoJSON, Shapefile, etc.)
            
        Returns:
            Earth Engine geometry object
            
        Raises:
            EarthEngineGeometryError: If geometry loading or conversion fails
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineError("Earth Engine not available")
            
        try:
            # Load AOI using geopandas
            aoi_gdf = gpd.read_file(aoi_file)
            
            if aoi_gdf.empty:
                raise EarthEngineGeometryError(f"AOI file is empty: {aoi_file}")
            
            # Reproject to WGS84 if needed
            if aoi_gdf.crs and aoi_gdf.crs.to_string() != 'EPSG:4326':
                logger.info(f"Reprojecting AOI from {aoi_gdf.crs} to EPSG:4326")
                aoi_gdf = aoi_gdf.to_crs('EPSG:4326')
            
            # Dissolve all features into single geometry if multiple
            if len(aoi_gdf) > 1:
                logger.info(f"Dissolving {len(aoi_gdf)} AOI features into single geometry")
                dissolved = aoi_gdf.dissolve()
                geometry = dissolved.geometry.iloc[0]
            else:
                geometry = aoi_gdf.geometry.iloc[0]
            
            # Convert to GeoJSON-like format for Earth Engine
            geom_dict = geometry.__geo_interface__
            
            # Create Earth Engine geometry
            ee_geometry = ee.Geometry(geom_dict)
            
            # Validate geometry
            area = ee_geometry.area().getInfo()
            logger.info(f"AOI geometry loaded successfully, area: {area:,.0f} square meters")
            
            # Check if geometry is too complex
            coordinates = geom_dict.get('coordinates', [])
            if isinstance(coordinates, list) and len(coordinates) > 0:
                if isinstance(coordinates[0], list) and len(coordinates[0]) > 1000:
                    logger.warning(
                        f"AOI has {len(coordinates[0])} vertices, this may cause Earth Engine timeouts. "
                        "Consider simplifying the geometry."
                    )
            
            return ee_geometry
            
        except Exception as e:
            if isinstance(e, EarthEngineGeometryError):
                raise
            else:
                raise EarthEngineGeometryError(f"Failed to load AOI geometry from {aoi_file}: {e}")
    
    def filter_buildings_by_confidence(self, geometry: EEGeometry, 
                                     confidence_threshold: float) -> EEFeatureCollection:
        """
        Filter buildings by confidence within AOI.
        
        Args:
            geometry: AOI geometry
            confidence_threshold: Minimum confidence threshold (0.5-1.0)
            
        Returns:
            Filtered Earth Engine FeatureCollection
            
        Raises:
            EarthEngineError: If filtering operation fails
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineError("Earth Engine not available")
            
        try:
            # Apply spatial filter
            spatial_filtered = self.collection.filterBounds(geometry)
            
            # Apply confidence filter
            confidence_filtered = spatial_filtered.filter(
                ee.Filter.gte('confidence', confidence_threshold)
            )
            
            logger.info(f"Applied confidence filter >= {confidence_threshold}")
            
            return confidence_filtered
            
        except Exception as e:
            raise EarthEngineError(f"Failed to filter buildings by confidence: {e}")
    
    def filter_buildings_by_area(self, collection: EEFeatureCollection,
                               min_area: Optional[float] = None,
                               max_area: Optional[float] = None) -> EEFeatureCollection:
        """
        Filter buildings by area.
        
        Args:
            collection: Input FeatureCollection
            min_area: Minimum area in square meters
            max_area: Maximum area in square meters
            
        Returns:
            Area-filtered FeatureCollection
            
        Raises:
            EarthEngineError: If filtering operation fails
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineError("Earth Engine not available")
            
        try:
            filtered_collection = collection
            
            if min_area is not None:
                filtered_collection = filtered_collection.filter(
                    ee.Filter.gte('area_in_meters', min_area)
                )
                logger.info(f"Applied minimum area filter >= {min_area} m²")
            
            if max_area is not None:
                filtered_collection = filtered_collection.filter(
                    ee.Filter.lte('area_in_meters', max_area)
                )
                logger.info(f"Applied maximum area filter <= {max_area} m²")
            
            return filtered_collection
            
        except Exception as e:
            raise EarthEngineError(f"Failed to filter buildings by area: {e}")
    
    def get_feature_count(self, collection: EEFeatureCollection) -> int:
        """
        Get the number of features in a collection.
        
        Args:
            collection: Earth Engine FeatureCollection
            
        Returns:
            Number of features (0 if count fails)
        """
        if not HAS_EARTH_ENGINE:
            return 0
            
        try:
            count = collection.size().getInfo()
            return count
        except Exception as e:
            logger.warning(f"Failed to get feature count: {e}")
            return 0
    
    def export_to_drive(self, collection: EEFeatureCollection, 
                       description: str, folder: Optional[str] = None) -> EETask:
        """
        Export collection to Google Drive.
        
        Args:
            collection: FeatureCollection to export
            description: Export description/filename
            folder: Optional Google Drive folder
            
        Returns:
            Earth Engine export task
            
        Raises:
            EarthEngineError: If export task creation fails
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineError("Earth Engine not available")
            
        try:
            task = ee.batch.Export.table.toDrive(
                collection=collection,
                description=description,
                folder=folder,
                fileFormat='GeoJSON'
            )
            
            task.start()
            logger.info(f"Started export task: {description}")
            
            return task
            
        except Exception as e:
            raise EarthEngineError(f"Failed to start export task: {e}")
    
    def wait_for_task(self, task: EETask, timeout_minutes: int = 30) -> bool:
        """
        Wait for an Earth Engine task to complete.
        
        Args:
            task: Earth Engine task
            timeout_minutes: Maximum time to wait
            
        Returns:
            True if task completed successfully, False otherwise
        """
        if not HAS_EARTH_ENGINE:
            return False
            
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                status = task.status()
                state = status.get('state')
                
                if state in ['COMPLETED']:
                    logger.info(f"Task completed successfully: {task.config.get('description', 'Unknown')}")
                    return True
                elif state in ['FAILED', 'CANCELLED']:
                    error_message = status.get('error_message', 'Unknown error')
                    logger.error(f"Task failed: {error_message}")
                    return False
                elif state in ['RUNNING']:
                    logger.debug(f"Task running... ({time.time() - start_time:.0f}s elapsed)")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error checking task status: {e}")
                return False
        
        logger.error(f"Task timed out after {timeout_minutes} minutes")
        return False
 

    def export_to_format(self, collection: EEFeatureCollection, 
                        output_path: Path, format_type: str,
                        include_properties: Optional[List[str]] = None,
                        max_features: Optional[int] = None,
                        # Grid processing parameters
                        enable_grid_processing: bool = True,
                        grid_size_m: int = BASE_GRID_SIZE_M,
                        grid_workers: int = DEFAULT_GRID_WORKERS,
                        grid_threshold: int = GRID_EXPORT_THRESHOLD) -> None:

        """
        Export Earth Engine collection to specified format.
        Automatically uses grid-based processing for large datasets.
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineError("Earth Engine not available")
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Apply feature limit if specified
            if max_features is not None:
                collection = collection.limit(max_features)
            
            # Select properties if specified
            if include_properties is not None:
                properties_with_geometry = ['geometry'] + include_properties
                collection = collection.select(properties_with_geometry)
            
            # Count features to determine export strategy
            logger.info("Counting features in collection...")
            feature_count = collection.size().getInfo()
            logger.info(f"Collection contains {feature_count:,} features")
            
            
            # Choose export strategy based on feature count and config
            if (enable_grid_processing and feature_count > grid_threshold):
                logger.info(f"Large dataset detected ({feature_count:,} features). Using grid-based export...")
                self._export_using_grid_processing(collection, output_path, format_type, 
                                                grid_size_m, grid_workers, grid_threshold)
            else:
                logger.info(f"Using direct export for {feature_count:,} features...")
                self._export_using_direct_method(collection, output_path, format_type)
                
            logger.info(f"Successfully exported {feature_count:,} buildings to {output_path}")
            
        except Exception as e:
            if isinstance(e, EarthEngineError):
                raise
            else:
                raise EarthEngineError(f"Export to {format_type} failed: {e}")

    def _export_to_geojson(self, collection: EEFeatureCollection, output_path: Path) -> None:
        """Export collection to GeoJSON format."""
        try:
            # Convert Earth Engine collection to GeoJSON
            geojson_data = collection.getInfo()
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
                
        except Exception as e:
            raise EarthEngineError(f"Failed to export GeoJSON: {e}")

    def _export_to_shapefile(self, collection: EEFeatureCollection, output_path: Path) -> None:
        """Export collection to Shapefile format."""
        try:
            # First export to GeoJSON in memory
            geojson_data = collection.getInfo()
            
            # Convert to GeoDataFrame using geopandas
            if not HAS_EARTH_ENGINE:
                raise EarthEngineError("GeoPandas not available for shapefile export")
                
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            
            # Set CRS to WGS84 (Earth Engine default)
            gdf.crs = 'EPSG:4326'
            
            # Export to shapefile
            gdf.to_file(output_path, driver='ESRI Shapefile')
            
        except Exception as e:
            raise EarthEngineError(f"Failed to export Shapefile: {e}")

    def _export_to_csv(self, collection: EEFeatureCollection, output_path: Path) -> None:
        """Export collection to CSV format (building centroids and attributes)."""
        try:
            # Add centroid coordinates to features
            def add_centroid_coords(feature):
                centroid = feature.geometry().centroid()
                coords = centroid.coordinates()
                return feature.set({
                    'centroid_longitude': coords.get(0),
                    'centroid_latitude': coords.get(1)
                })
            
            # Apply centroid calculation
            collection_with_centroids = collection.map(add_centroid_coords)
            
            # Convert to client-side data
            data = collection_with_centroids.getInfo()
            
            # Extract features and create CSV data
            csv_data = []
            for feature in data['features']:
                properties = feature.get('properties', {})
                csv_data.append(properties)
            
            # Write to CSV using pandas
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            raise EarthEngineError(f"Failed to export CSV: {e}")

    def _save_combined_features(self, all_features: List[Dict], 
                            output_path: Path, format_type: str) -> None:
        """Save combined features to the specified format."""
        logger.info(f"Saving {len(all_features):,} features to {format_type} format...")
        
        if format_type.lower() == 'geojson':
            geojson_data = {
                "type": "FeatureCollection",
                "features": all_features
            }
            with open(output_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
        
        elif format_type.lower() == 'shapefile':
            gdf = gpd.GeoDataFrame.from_features(all_features)
            gdf.crs = 'EPSG:4326'
            gdf = self._truncate_shapefile_columns(gdf)
            gdf.to_file(output_path, driver='ESRI Shapefile')
        
        elif format_type.lower() == 'csv':
            if not HAS_PANDAS:
                raise EarthEngineError("Pandas is required for CSV export. Install with: pip install pandas")
            
            # Extract centroids for CSV
            rows = []
            for feature in all_features:
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                if geom.get('type') == 'Polygon' and geom.get('coordinates'):
                    # Calculate centroid (simplified)
                    coords = geom['coordinates'][0]
                    if coords:
                        centroid_lon = sum(c[0] for c in coords) / len(coords)
                        centroid_lat = sum(c[1] for c in coords) / len(coords)
                        props.update({'longitude': centroid_lon, 'latitude': centroid_lat})
                rows.append(props)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        else:
            raise EarthEngineError(f"Unsupported format: {format_type}")

    def wait_for_export_task(self, task: EETask, timeout_minutes: int = 30) -> bool:
        """
        Wait for an Earth Engine export task to complete.
        
        Args:
            task: Earth Engine Task object
            timeout_minutes: Maximum time to wait
            
        Returns:
            True if task completed successfully, False otherwise
        """
        if not HAS_EARTH_ENGINE:
            return False
            
        import time
        
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            task_status = task.status()
            state = task_status['state']
            
            if state == 'COMPLETED':
                logger.info("Export task completed successfully")
                return True
            elif state == 'FAILED':
                error_message = task_status.get('error_message', 'Unknown error')
                logger.error(f"Export task failed: {error_message}")
                return False
            elif state in ['CANCEL_REQUESTED', 'CANCELLED']:
                logger.warning("Export task was cancelled")
                return False
            else:
                # Still running, wait and check again
                logger.info(f"Export task status: {state}")
                time.sleep(10)  # Wait 10 seconds before checking again
        
        # Timeout reached
        logger.error(f"Export task timeout after {timeout_minutes} minutes")
        return False

    def _export_using_direct_method(self, collection: EEFeatureCollection, 
                                output_path: Path, format_type: str) -> None:
        """Use existing direct export for smaller datasets."""
        if format_type.lower() == 'geojson':
            self._export_to_geojson(collection, output_path)
        elif format_type.lower() == 'shapefile':
            self._export_to_shapefile(collection, output_path)
        elif format_type.lower() == 'csv':
            self._export_to_csv(collection, output_path)
        else:
            raise EarthEngineError(f"Unsupported export format: {format_type}")

    def _export_using_grid_processing(self, collection: EEFeatureCollection,
                                    output_path: Path, format_type: str,
                                    grid_size_m: int = BASE_GRID_SIZE_M,
                                    grid_workers: int = DEFAULT_GRID_WORKERS,
                                    grid_threshold: int = SUBDIVISION_THRESHOLD) -> None:
        """Use grid-based processing for large datasets."""
        logger.info("Starting grid-based export processing...")
        
        # Get collection bounds
        bounds = collection.geometry().bounds().getInfo()
        bounds_list = [bounds['coordinates'][0][0][0], bounds['coordinates'][0][0][1],
                    bounds['coordinates'][0][2][0], bounds['coordinates'][0][2][1]]
        
        # Create grid tasks using provided parameters
        grid_tasks = self._create_grid_tasks_from_bounds(bounds_list, grid_size_m)
        logger.info(f"Created {len(grid_tasks)} grid cells for processing")
        
        # Process grid cells in parallel using provided workers
        all_features = self._process_grid_cells_parallel(collection, grid_tasks, grid_workers, grid_threshold)
        
        # Combine and export results
        self._save_combined_features(all_features, output_path, format_type)

    def _create_grid_tasks_from_bounds(self, bounds_wgs84: List[float], grid_size_m: int) -> List[Dict]:
        """Create grid tasks covering the bounding box."""
        west, south, east, north = bounds_wgs84
        
        # Convert to Africa Albers for metric grid
        bbox_wgs84 = box(west, south, east, north)
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_wgs84], crs="EPSG:4326")
        bbox_albers = bbox_gdf.to_crs(GRID_CRS_METRIC)
        
        x_min, y_min, x_max, y_max = bbox_albers.total_bounds
        x_coords = np.arange(x_min, x_max, grid_size_m)  # Use parameter, not constant
        y_coords = np.arange(y_min, y_max, grid_size_m)  # Use parameter, not constant
        
        tasks = []
        task_id = 0
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                cell_bounds_albers = (x, y, x_coords[i+1], y_coords[j+1])
                cell_geom_albers = box(*cell_bounds_albers)
                
                # Transform back to WGS84
                cell_gdf_albers = gpd.GeoDataFrame([1], geometry=[cell_geom_albers], crs=GRID_CRS_METRIC)
                cell_gdf_wgs84 = cell_gdf_albers.to_crs("EPSG:4326")
                bounds_wgs84_cell = cell_gdf_wgs84.total_bounds.tolist()
                
                tasks.append({
                    'task_id': task_id,
                    'grid_id': f"{i}_{j}",
                    'bounds_wgs84': bounds_wgs84_cell,
                    'grid_size_m': grid_size_m  # Use parameter, not constant
                })
                task_id += 1
        
        return tasks

    def _process_grid_cells_parallel(self, collection: EEFeatureCollection, 
                                    grid_tasks: List[Dict], grid_workers: int, 
                                    subdivision_threshold: int) -> List[Dict]:
        """Process grid cells in parallel and return all features."""
        all_features = []
        results_lock = threading.Lock()
        completed_tasks = 0
        
        def process_single_cell(task: Dict) -> None:
            nonlocal completed_tasks
            
            try:
                # Create geometry for this cell
                bounds = task['bounds_wgs84']
                west, south, east, north = bounds
                cell_geometry = ee.Geometry.Rectangle([west, south, east, north])
                
                # Filter collection to this cell
                cell_buildings = collection.filterBounds(cell_geometry)
                building_count = cell_buildings.size().getInfo()
                
                if building_count == 0:
                    return
                
                # Check if subdivision needed - use parameter, not constant
                if building_count > subdivision_threshold:
                    logger.info(f"Subdividing dense cell {task['grid_id']} ({building_count} buildings)")
                    cell_features = self._process_cell_with_subdivision(
                        cell_buildings, bounds, subdivision_threshold=subdivision_threshold
                    )
                else:
                    # Direct download
                    cell_features = cell_buildings.getInfo().get('features', [])
                
                # Thread-safe feature collection
                with results_lock:
                    all_features.extend(cell_features)
                    completed_tasks += 1
                    
                    if completed_tasks % 10 == 0:
                        progress = (completed_tasks / len(grid_tasks)) * 100
                        logger.info(f"Progress: {completed_tasks}/{len(grid_tasks)} cells ({progress:.1f}%)")
            
            except Exception as e:
                logger.warning(f"Error processing cell {task['grid_id']}: {e}")
                with results_lock:
                    completed_tasks += 1
        
        # Process in parallel - use parameter, not constant
        with ThreadPoolExecutor(max_workers=grid_workers) as executor:
            futures = [executor.submit(process_single_cell, task) for task in grid_tasks]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Cell processing error: {e}")
        
        logger.info(f"Collected {len(all_features):,} features from {len(grid_tasks)} grid cells")
        return all_features

    def _process_cell_with_subdivision(self, cell_buildings: EEFeatureCollection, 
                                    bounds_wgs84: List[float], 
                                    subdivision_threshold: int = SUBDIVISION_THRESHOLD,
                                    max_features_per_cell: int = MAX_FEATURES_PER_CELL,
                                    min_cell_size_m: float = MIN_CELL_SIZE_M,
                                    max_depth: int = MAX_SUBDIVISION_DEPTH,
                                    depth: int = 0) -> List[Dict]:
        """
        Recursively subdivide a dense cell and collect features.
        
        This method handles cells that exceed the Earth Engine feature limit by:
        1. Checking if subdivision is needed based on feature count and depth limits
        2. Converting to metric CRS for precise geometric subdivision
        3. Splitting the cell into 4 quadrants
        4. Recursively processing each quadrant until feature counts are manageable
        
        Args:
            cell_buildings: Earth Engine FeatureCollection for this cell
            bounds_wgs84: Cell bounds in WGS84 [west, south, east, north]
            subdivision_threshold: Feature count threshold to trigger subdivision
            max_features_per_cell: Maximum features to download per cell (EE limit)
            min_cell_size_m: Minimum cell size in meters to prevent infinite subdivision
            max_depth: Maximum recursion depth to prevent infinite loops
            depth: Current recursion depth
            
        Returns:
            List of feature dictionaries ready for export
            
        Raises:
            EarthEngineError: If subdivision process fails
        """
        try:
            # Count buildings in current cell
            building_count = cell_buildings.size().getInfo()
            
            # Base cases - stop subdivision
            if building_count <= max_features_per_cell:
                # Cell is small enough, download directly
                return cell_buildings.getInfo().get('features', [])
            
            if depth >= max_depth:
                logger.warning(
                    f"Maximum subdivision depth ({max_depth}) reached at depth {depth}. "
                    f"Truncating to {max_features_per_cell} features to avoid infinite recursion."
                )
                return cell_buildings.limit(max_features_per_cell).getInfo().get('features', [])
            
            # Convert bounds to metric CRS for precise geometric operations
            west, south, east, north = bounds_wgs84
            bbox_wgs84 = box(west, south, east, north)
            gdf_wgs84 = gpd.GeoDataFrame([1], geometry=[bbox_wgs84], crs="EPSG:4326")
            gdf_albers = gdf_wgs84.to_crs(GRID_CRS_METRIC)
            x_min, y_min, x_max, y_max = gdf_albers.total_bounds
            
            # Check minimum cell size to prevent infinite subdivision
            current_width_m = x_max - x_min
            current_height_m = y_max - y_min
            current_size_m = min(current_width_m, current_height_m)
            
            if current_size_m < min_cell_size_m:
                logger.warning(
                    f"Cell size ({current_size_m:.1f}m) below minimum ({min_cell_size_m}m). "
                    f"Truncating to {max_features_per_cell} features to prevent over-subdivision."
                )
                return cell_buildings.limit(max_features_per_cell).getInfo().get('features', [])
            
            # Log subdivision attempt
            logger.debug(
                f"Subdividing cell at depth {depth}: {building_count} buildings, "
                f"size {current_size_m:.1f}m"
            )
            
            # Split into 4 quadrants in metric space
            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            
            quadrants_albers = [
                (x_min, y_min, mid_x, mid_y),    # Southwest
                (mid_x, y_min, x_max, mid_y),    # Southeast  
                (x_min, mid_y, mid_x, y_max),    # Northwest
                (mid_x, mid_y, x_max, y_max)     # Northeast
            ]
            
            # Process each quadrant
            all_features = []
            
            for i, quad_albers in enumerate(quadrants_albers):
                try:
                    # Convert quadrant bounds back to WGS84
                    quad_geom_albers = box(*quad_albers)
                    quad_gdf_albers = gpd.GeoDataFrame([1], geometry=[quad_geom_albers], crs=GRID_CRS_METRIC)
                    quad_gdf_wgs84 = quad_gdf_albers.to_crs("EPSG:4326")
                    quad_bounds_wgs84 = quad_gdf_wgs84.total_bounds.tolist()
                    
                    # Create Earth Engine geometry for this quadrant
                    quad_west, quad_south, quad_east, quad_north = quad_bounds_wgs84
                    quad_geometry = ee.Geometry.Rectangle([quad_west, quad_south, quad_east, quad_north])
                    
                    # Filter buildings to this quadrant
                    quad_buildings = cell_buildings.filterBounds(quad_geometry)
                    quad_count = quad_buildings.size().getInfo()
                    
                    if quad_count == 0:
                        # No buildings in this quadrant, skip
                        continue
                        
                    elif quad_count <= max_features_per_cell:
                        # Quadrant is small enough, download directly
                        logger.debug(f"Downloading {quad_count} buildings from quadrant {i}")
                        quad_features = quad_buildings.getInfo().get('features', [])
                        all_features.extend(quad_features)
                        
                    else:
                        # Quadrant still too large, recurse
                        logger.debug(
                            f"Quadrant {i} has {quad_count} buildings (>{max_features_per_cell}), "
                            f"subdividing further (depth {depth+1})"
                        )
                        subdivided_features = self._process_cell_with_subdivision(
                            quad_buildings, 
                            quad_bounds_wgs84,
                            subdivision_threshold=subdivision_threshold,
                            max_features_per_cell=max_features_per_cell,
                            min_cell_size_m=min_cell_size_m,
                            max_depth=max_depth,
                            depth=depth + 1
                        )
                        all_features.extend(subdivided_features)
                        
                except Exception as e:
                    logger.error(f"Error processing quadrant {i} at depth {depth}: {e}")
                    # Continue with other quadrants rather than failing completely
                    continue
            
            logger.debug(
                f"Subdivision at depth {depth} complete: collected {len(all_features)} features "
                f"from {len(quadrants_albers)} quadrants"
            )
            
            return all_features
            
        except Exception as e:
            logger.error(f"Subdivision failed at depth {depth}: {e}")
            # Fallback: try to download what we can with the limit
            try:
                logger.warning(f"Falling back to truncated download of {max_features_per_cell} features")
                return cell_buildings.limit(max_features_per_cell).getInfo().get('features', [])
            except Exception as fallback_error:
                logger.error(f"Fallback download also failed: {fallback_error}")
                return []  # Return empty list rather than crashing
            
def retry_ee_operation(func, max_attempts: int = DEFAULT_EE_RETRY_ATTEMPTS, 
                      delay: float = DEFAULT_EE_RETRY_DELAY):
    """
    Retry Earth Engine operations with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            error_message = str(e).lower()
            
            # Don't retry authentication errors
            if any(pattern in error_message for pattern in EE_ERROR_PATTERNS['authentication']):
                raise
            
            # Retry quota and timeout errors
            if attempt < max_attempts - 1:
                if any(pattern in error_message for pattern in EE_ERROR_PATTERNS['quota']):
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Earth Engine quota/timeout error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
                    continue
            
            # Re-raise if last attempt or non-retryable error
            if attempt == max_attempts - 1:
                raise


def get_available_datasets() -> Dict[str, str]:
    """
    Get available Earth Engine datasets for Open Buildings.
    
    Returns:
        Dictionary mapping dataset keys to collection paths
    """
    return EARTH_ENGINE_DATASETS.copy()


def format_academic_guidance() -> str:
    """
    Get comprehensive academic setup guidance.
    
    Returns:
        Formatted guidance string for academic users
    """
    guidance = """
Earth Engine Setup for Academic Users:

1. Get Earth Engine Access:
   - Visit: https://earthengine.google.com/signup/
   - Select 'Register for a noncommercial use account'
   - Provide academic institution details
   - Wait for approval (usually 1-2 days)

2. Setup Authentication:
   - Option A: Service Account (Recommended for teams)
     * Go to Google Cloud Console
     * Create/select project
     * Enable Earth Engine API
     * Create service account with Earth Engine permissions
     * Download JSON key file
     * Set service_account_key in config
   
   - Option B: User Credentials
     * Run: earthengine authenticate
     * Follow browser authentication
     * Credentials stored automatically

3. Configuration Tips:
   - Start with small AOIs for testing
   - Use confidence_threshold >= 0.75 for reliable results
   - Set max_features for large areas to avoid timeouts
   - Higher confidence = fewer but more reliable buildings

4. Common Issues:
   - Quota exceeded: Use smaller AOIs or higher confidence thresholds
   - Authentication failed: Check credentials and project setup
   - Geometry too complex: Simplify AOI polygons
   - Timeout errors: Break large requests into smaller chunks
    """
    return guidance.strip()


# Also add this utility function at the module level (outside classes):

def create_buildings_feature_properties(include_confidence: bool = True, 
                                      include_area: bool = True,
                                      include_plus_codes: bool = True) -> List[str]:
    """
    Create list of feature properties to include in building exports.
    
    Args:
        include_confidence: Include confidence scores
        include_area: Include calculated area
        include_plus_codes: Include Plus Codes
        
    Returns:
        List of property names to select
    """
    properties = []
    
    if include_confidence:
        properties.append('confidence')
    
    if include_area:
        properties.append('area_in_meters')
    
    if include_plus_codes:
        properties.append('plus_code')
    
    # Always include basic identifiers if available
    properties.extend(['longitude', 'latitude'])
    
    return properties


