"""
Global constants for the geoworkflow package.

This module contains constants used throughout the package including
file extensions, coordinate reference systems, and default values.
"""

from enum import Enum
from pathlib import Path
from typing import Set, Dict, Any


# ============================================================================
# File Extensions and Patterns
# ============================================================================

# Supported raster file extensions
RASTER_EXTENSIONS: Set[str] = {
    '.tif', '.tiff', '.geotif', '.geotiff', 
    '.TIF', '.TIFF', '.GEOTIF', '.GEOTIFF',
    '.nc', '.netcdf', '.hdf', '.h5'
}

# Supported vector file extensions
VECTOR_EXTENSIONS: Set[str] = {
    '.shp', '.geojson', '.gpkg', '.gml', '.kml', '.kmz',
    '.SHP', '.GEOJSON', '.GPKG', '.GML', '.KML', '.KMZ'
}

# Supported archive extensions
ARCHIVE_EXTENSIONS: Set[str] = {
    '.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2',
    '.ZIP', '.TAR', '.TAR.GZ', '.TGZ', '.TAR.BZ2', '.TBZ2'
}

# Default file patterns
DEFAULT_RASTER_PATTERN: str = "*.tif"
DEFAULT_VECTOR_PATTERN: str = "*.{shp,geojson,gpkg,kml}"
DEFAULT_ARCHIVE_PATTERN: str = "*.zip"


# ============================================================================
# Coordinate Reference Systems
# ============================================================================

# Common CRS codes used in African geospatial work
class CommonCRS:
    """Common coordinate reference systems for African geospatial data."""
    
    # Geographic coordinate systems
    WGS84 = "EPSG:4326"
    WGS84_UTM_33S = "EPSG:32733"  # Common for Southern Africa
    WGS84_UTM_34S = "EPSG:32734"  # Common for Southern/Eastern Africa
    
    # Projected coordinate systems
    AFRICA_ALBERS = "ESRI:102022"  # Africa Albers Equal Area Conic
    AFRICA_LAMBERT = "ESRI:102023"  # Africa Lambert Conformal Conic
    
    # Web Mercator (for web mapping)
    WEB_MERCATOR = "EPSG:3857"


# Default CRS for processing
DEFAULT_CRS: str = CommonCRS.WGS84
DEFAULT_PROCESSING_CRS: str = CommonCRS.AFRICA_ALBERS


# ============================================================================
# Processing Stages
# ============================================================================

class ProcessingStage(Enum):
    """Processing stages in the geoworkflow pipeline."""
    
    SOURCE = "00_source"
    EXTRACTED = "01_extracted"
    CLIPPED = "02_clipped"
    PROCESSED = "03_processed"
    ANALYSIS_READY = "04_analysis_ready"


# Stage directories
STAGE_DIRECTORIES: Dict[ProcessingStage, str] = {
    ProcessingStage.SOURCE: "00_source/archives",
    ProcessingStage.EXTRACTED: "01_extracted",
    ProcessingStage.CLIPPED: "02_clipped",
    ProcessingStage.PROCESSED: "03_processed",
    ProcessingStage.ANALYSIS_READY: "04_analysis_ready"
}


# ============================================================================
# Data Type Categories
# ============================================================================

class DataType(Enum):
    """Data type categories for processing decisions."""
    
    RASTER = "raster"
    VECTOR = "vector"
    ARCHIVE = "archive"
    NETCDF = "netcdf"
    OTHER = "other"


# Data source categories for specialized processing
class DataSource(Enum):
    """Known data sources with specialized processing requirements."""
    
    COPERNICUS = "copernicus"
    ODIAC = "odiac"
    PM25 = "pm25"
    AFRICA_POLIS = "africa_polis"
    LANDSAT = "landsat"
    MODIS = "modis"
    SENTINEL = "sentinel"
    OTHER = "other"


# ============================================================================
# Visualization Constants
# ============================================================================

# Default visualization parameters
DEFAULT_VISUALIZATION_CONFIG: Dict[str, Any] = {
    "colormap": "viridis",
    "classification_method": "auto",
    "n_classes": 8,
    "dpi": 150,
    "figure_width": 14,
    "figure_height": 10,
    "show_colorbar": True,
    "add_basemap": True,
    "basemap_source": "CartoDB.Positron",
    "raster_alpha": 0.85,
    "basemap_alpha": 0.7
}

# Land cover color schemes (UN LCCS)
LAND_COVER_COLORS: Dict[int, str] = {
    0: "#000000",     # No Data
    10: "#ffff64",    # Cropland
    20: "#aaf0f0",    # Forest
    30: "#dcf064",    # Grassland
    40: "#c8c8c8",    # Shrubland
    50: "#006400",    # Wetlands
    60: "#ffb432",    # Settlement
    70: "#ffc85a",    # Bare/sparse vegetation
    80: "#0064c8",    # Water bodies
    90: "#ffffff"     # Snow/ice
}


# ============================================================================
# Processing Defaults
# ============================================================================

# Default buffer distances (in kilometers)
DEFAULT_BUFFER_KM: float = 100.0

# Default resampling methods
DEFAULT_RESAMPLING_CATEGORICAL: str = "nearest"
DEFAULT_RESAMPLING_CONTINUOUS: str = "cubic"

# Memory and performance settings
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_MAX_WORKERS: int = 4
DEFAULT_TIMEOUT_SECONDS: int = 3600  # 1 hour


# ============================================================================
# Logging Configuration
# ============================================================================

# Log levels
LOG_LEVELS: Dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# File Size Limits
# ============================================================================

# Maximum file sizes for different operations (in bytes)
MAX_MEMORY_RASTER_SIZE: int = 1024 * 1024 * 1024  # 1 GB
MAX_VISUALIZATION_SIZE: int = 100 * 1024 * 1024    # 100 MB

# Downsampling thresholds
DOWNSAMPLE_THRESHOLD_PIXELS: int = 10000 * 10000   # 100 million pixels


# ============================================================================
# Application Metadata
# ============================================================================

# Package information
PACKAGE_NAME: str = "geoworkflow"
PACKAGE_VERSION: str = "1.0.0"
PACKAGE_AUTHOR: str = "Geoworkflow Team"

# Default project structure
DEFAULT_PROJECT_STRUCTURE: Dict[str, str] = {
    "data": "data",
    "config": "config", 
    "outputs": "outputs",
    "logs": "logs",
    "cache": "data/cache"
}

# ============================================================================
# Google Earth Engine Constants
# ============================================================================
EARTH_ENGINE_DATASETS = {
    'open_buildings_v3': 'GOOGLE/Research/open-buildings/v3/polygons',
    'open_buildings_temporal_v1': 'GOOGLE/Research/open-buildings-temporal/v1/polygons'
}

# Default Earth Engine settings
DEFAULT_EARTH_ENGINE_TIMEOUT = 1800  # 30 minutes in seconds
DEFAULT_BUILDING_CONFIDENCE = 0.75
DEFAULT_MIN_BUILDING_AREA = 10.0  # square meters
DEFAULT_MAX_BUILDING_AREA = 100000.0  # square meters
DEFAULT_CHUNK_SIZE = 1000


# Earth Engine quota and retry settings
DEFAULT_EE_RETRY_ATTEMPTS = 3
DEFAULT_EE_RETRY_DELAY = 5  # seconds
MAX_EE_FEATURES_PER_REQUEST = 5000

# Common Earth Engine error patterns for academic-friendly messages
EE_ERROR_PATTERNS = {
    'authentication': ['authentication', 'unauthorized', 'credentials'],
    'quota': ['quota', 'limit', 'rate limit', 'too many requests'],
    'timeout': ['timeout', 'deadline', 'cancelled'],
    'geometry': ['invalid geometry', 'self-intersection', 'too complex']
}