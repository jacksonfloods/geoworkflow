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

import json  # For GeoJSON export
import time  # For task waiting
from typing import Optional, Union, Dict, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime

# Additional imports for CSV export
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Shapely for geometry operations (if not already imported)
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
    import geopandas as gpd
    HAS_EARTH_ENGINE = True
except ImportError as e:
    EARTH_ENGINE_ERROR = str(e)
    # Only import for type checking, not runtime
    if TYPE_CHECKING:
        import ee
        import geopandas as gpd
    else:
        # Runtime stubs that will never be used when EE is unavailable
        ee = None
        gpd = None

# Import our exceptions and constants
from ..core.exceptions import (
    EarthEngineError, 
    EarthEngineAuthenticationError,
    EarthEngineQuotaError,
    EarthEngineTimeoutError,
    EarthEngineGeometryError
)
from ..core.constants import (
    EARTH_ENGINE_DATASETS,
    DEFAULT_EARTH_ENGINE_TIMEOUT,
    DEFAULT_BUILDING_CONFIDENCE,
    DEFAULT_MIN_BUILDING_AREA,
    DEFAULT_MAX_BUILDING_AREA,
    DEFAULT_EE_RETRY_ATTEMPTS,
    DEFAULT_EE_RETRY_DELAY,
    EE_ERROR_PATTERNS
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
                    project_id: Optional[str] = None) -> str:
        """
        Authenticate with Earth Engine using academic-friendly credential discovery.
        
        Order of precedence:
        1. Explicit service account key file
        2. GOOGLE_APPLICATION_CREDENTIALS environment variable  
        3. Default EE credentials location
        
        Args:
            service_account_key: Optional path to service account key
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
                logger.info(f"Using service account key: {service_account_key}")
                
                # Extract project ID from service account if not provided
                if not project_id:
                    project_id = EarthEngineAuth.get_project_id_from_service_account(service_account_key)
                
                # Initialize with service account
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Will be extracted from key file
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
    
    def load_aoi_geometry(self, aoi_file: Path) -> EEGeometry:
        """
        Load AOI from file and convert to Earth Engine geometry.
        
        Args:
            aoi_file: Path to AOI file (GeoJSON, Shapefile, etc.)
            
        Returns:
            Earth Engine Geometry object
            
        Raises:
            EarthEngineGeometryError: If geometry loading or conversion fails
        """
        if not HAS_EARTH_ENGINE:
            raise EarthEngineGeometryError("Earth Engine not available")
            
        if not aoi_file.exists():
            raise EarthEngineGeometryError(f"AOI file not found: {aoi_file}")
            
        try:
            # Load AOI using geopandas
            gdf = gpd.read_file(aoi_file)
            
            if gdf.empty:
                raise EarthEngineGeometryError(f"AOI file is empty: {aoi_file}")
            
            # Ensure CRS is WGS84 for Earth Engine
            if gdf.crs is None:
                logger.warning("AOI file has no CRS, assuming EPSG:4326")
                gdf.crs = 'EPSG:4326'
            elif gdf.crs.to_string() != 'EPSG:4326':
                logger.info(f"Reprojecting AOI from {gdf.crs} to EPSG:4326")
                gdf = gdf.to_crs('EPSG:4326')
            
            # Combine all geometries into one (union)
            if len(gdf) > 1:
                logger.info(f"Combining {len(gdf)} AOI features into single geometry")
                combined_geom = gdf.geometry.unary_union
            else:
                combined_geom = gdf.geometry.iloc[0]
            
            # Convert to GeoJSON format for Earth Engine
            if hasattr(combined_geom, '__geo_interface__'):
                geojson_geom = combined_geom.__geo_interface__
            else:
                # Fallback: use shapely's mapping
                from shapely.geometry import mapping
                geojson_geom = mapping(combined_geom)
            
            # Create Earth Engine geometry
            ee_geometry = ee.Geometry(geojson_geom)
            
            # Validate geometry
            try:
                # Simple validation: check if geometry is valid
                area = ee_geometry.area().getInfo()  
                if area <= 0:
                    raise EarthEngineGeometryError("AOI geometry has zero or negative area")
            except Exception as e:
                logger.warning(f"Geometry validation warning: {e}")
                if "too complex" in str(e).lower():
                    raise EarthEngineGeometryError(
                        f"AOI geometry is too complex for Earth Engine. "
                        "Consider simplifying the geometry."
                    )
            
            logger.info(f"Successfully loaded AOI geometry from {aoi_file}")
            return ee_geometry
            
        except Exception as e:
            if isinstance(e, EarthEngineGeometryError):
                raise
            else:
                raise EarthEngineGeometryError(f"Failed to load AOI geometry from {aoi_file}: {e}")

    def export_to_format(self, collection: EEFeatureCollection, 
                        output_path: Path, format_type: str,
                        include_properties: Optional[List[str]] = None,
                        max_features: Optional[int] = None) -> None:
        """
        Export Earth Engine collection to specified format.
        
        Args:
            collection: Earth Engine FeatureCollection to export
            output_path: Output file path
            format_type: Export format ('geojson', 'shapefile', 'csv')
            include_properties: List of properties to include (None for all)
            max_features: Maximum number of features to export
            
        Raises:
            EarthEngineError: If export operation fails
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
                # Always include geometry
                properties_with_geometry = ['geometry'] + include_properties
                collection = collection.select(properties_with_geometry)
            
            # Convert to client-side for local export
            logger.info(f"Converting Earth Engine collection to {format_type} format...")
            
            if format_type.lower() == 'geojson':
                self._export_to_geojson(collection, output_path)
            elif format_type.lower() == 'shapefile':
                self._export_to_shapefile(collection, output_path)
            elif format_type.lower() == 'csv':
                self._export_to_csv(collection, output_path)
            else:
                raise EarthEngineError(f"Unsupported export format: {format_type}")
                
            logger.info(f"Successfully exported buildings to {output_path}")
            
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