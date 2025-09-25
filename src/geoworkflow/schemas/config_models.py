# File: src/geoworkflow/schemas/config_models.py
"""
Configuration models for the geoworkflow package using Pydantic v2.

This module defines all configuration schemas used throughout the workflow,
providing type safety, validation, and documentation for configuration options.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        use_enum_values=True
    )


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ResamplingMethod(str, Enum):
    """Resampling methods for raster operations."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBIC_SPLINE = "cubic_spline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"


class ClassificationMethod(str, Enum):
    """Classification methods for visualization."""
    AUTO = "auto"
    EQUAL_INTERVAL = "equal_interval"
    QUANTILE = "quantile"
    JENKS = "jenks"
    LOG = "log"


class NoDataDetectionMethod(str, Enum):
    """NoData detection methods."""
    AUTO = "auto"
    METADATA_ONLY = "metadata_only"
    COMMON_VALUES = "common_values"
    STATISTICAL = "statistical"

class OpenBuildingsDataset(str, Enum):
    """Open Buildings dataset versions."""
    V3 = "v3"
    TEMPORAL_V1 = "temporal_v1"  # For future enhancement


class BuildingExportFormat(str, Enum):
    """Export formats for building data."""
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    CSV = "csv"


class EarthEngineAuthMethod(str, Enum):
    """Earth Engine authentication methods."""
    SERVICE_ACCOUNT = "service_account"
    USER_CREDENTIALS = "user_credentials"
    DEFAULT = "default"


class AOIConfig(BaseConfig):
    """Configuration for Area of Interest operations."""
    
    # Source specification
    input_file: Path = Field(..., description="Path to administrative boundaries file")
    country_name_column: str = Field("NAME_0", description="Column name containing country names")
    countries: Optional[List[str]] = Field(None, description="List of country names")
    use_all_countries: bool = Field(False, description="Use all countries in the boundaries file")
    buffer_km: Optional[float] = Field(None, ge=0, le=1000, description="Buffer distance in kilometers")
    dissolve_boundaries: bool = Field(False, description="Dissolve country boundaries into single polygon")
    
    # Output configuration  
    output_file: Optional[Path] = Field(None, description="Output file path for AOI")
    output_crs: str = Field("EPSG:4326", description="Output coordinate reference system")
    
    # Processing options
    simplify_tolerance: Optional[float] = Field(None, ge=0, description="Geometry simplification tolerance")
    validate_geometry: bool = Field(True, description="Validate geometry after processing")


# Data Source Configuration
class DataSourceConfig(BaseConfig):
    """Configuration for data sources."""
    
    name: str = Field(..., description="Data source name")
    source_type: str = Field(..., description="Type of data source")
    url: Optional[str] = Field(None, description="Download URL")
    local_path: Optional[Path] = Field(None, description="Local file path")
    credentials: Optional[Dict[str, str]] = Field(None, description="Authentication credentials")
    
    # Download options
    chunk_size: int = Field(8192, ge=1024, description="Download chunk size in bytes")
    timeout: int = Field(300, ge=30, description="Download timeout in seconds")
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts")


# Extraction Configuration
class ExtractionConfig(BaseConfig):
    """Configuration for archive extraction operations."""
    
    # Input specification
    zip_file: Optional[Path] = Field(None, description="Single ZIP file to extract")
    zip_folder: Optional[Path] = Field(None, description="Folder containing ZIP files")
    
    # Output configuration
    output_dir: Path = Field(..., description="Output directory for extracted files")
    preserve_structure: bool = Field(True, description="Preserve directory structure")
    
    # Processing options
    overwrite_existing: bool = Field(False, description="Overwrite existing extracted files")
    cleanup_archives: bool = Field(False, description="Remove archives after extraction")
    
    # File filtering
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to extract")
    exclude_patterns: Optional[List[str]] = Field(None, description="Patterns to exclude")


class ClippingConfig(BaseConfig):
    """Configuration for spatial clipping operations."""
    
    # Input/Output (using notebook-compatible field names)
    input_directory: Path = Field(..., description="Input directory containing files to clip")
    output_dir: Path = Field(..., description="Output directory for clipped files")
    aoi_file: Path = Field(..., description="Area of Interest file for clipping")
    
    # Processing options
    raster_pattern: str = Field("*.tif", description="Pattern to match raster files")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    overwrite_existing: bool = Field(False, description="Overwrite existing clipped files")
    all_touched: bool = Field(False, description="Include pixels touched by geometry")
    
    # CRS handling
    target_crs: Optional[str] = Field(None, description="Target CRS for output")
    reproject_aoi: bool = Field(True, description="Reproject AOI to match data CRS")
    
    # File filtering
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to process")


# Enhanced Alignment Configuration with NoData Detection
class AlignmentConfig(BaseConfig):
    """Configuration for raster alignment operations with enhanced nodata handling."""
    
    # Core alignment settings
    reference_raster: Path = Field(..., description="Reference raster for alignment")
    input_directory: Optional[Path] = Field(None, description="Input directory containing rasters")
    output_dir: Path = Field(..., description="Output directory for aligned rasters")
    
    # Processing options
    resampling_method: ResamplingMethod = Field(ResamplingMethod.CUBIC, description="Resampling method")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    skip_existing: bool = Field(True, description="Skip files that already exist")
    preserve_directory_structure: bool = Field(True, description="Preserve input directory structure")
    
    # Enhanced NoData Detection Options
    auto_detect_nodata: bool = Field(
        True, 
        description="Automatically detect nodata values when metadata is missing"
    )
    nodata_detection_method: NoDataDetectionMethod = Field(
        NoDataDetectionMethod.AUTO,
        description="Method for nodata detection"
    )
    nodata_validation: bool = Field(
        True,
        description="Validate detected nodata values for reliability"
    )
    force_nodata_redetection: bool = Field(
        False,
        description="Force re-detection even when cached metadata exists"
    )
    nodata_cache_metadata: bool = Field(
        True,
        description="Cache detected nodata values in raster metadata"
    )
    
    # CRS handling
    target_crs: Optional[str] = Field(None, description="Target CRS (uses reference if None)")
    
    # File filtering
    file_extensions: List[str] = Field(
        ['.tif', '.tiff'], 
        description="File extensions to process"
    )
    
    @field_validator('reference_raster')
    @classmethod
    def validate_reference_raster_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Reference raster does not exist: {v}")
        return v


# Statistical Enrichment Configuration
class StatisticalEnrichmentConfig(BaseConfig):
    """Configuration for statistical enrichment operations."""
    
    # Input/Output
    input_dir: Path = Field(..., description="Directory containing aligned rasters")
    output_dir: Path = Field(..., description="Output directory for enriched data")
    aoi_file: Path = Field(..., description="AOI file for statistical analysis")
    
    # Statistics to compute
    compute_mean: bool = Field(True, description="Compute mean values")
    compute_median: bool = Field(True, description="Compute median values")
    compute_std: bool = Field(True, description="Compute standard deviation")
    compute_percentiles: bool = Field(True, description="Compute percentiles")
    percentiles: List[float] = Field([25, 75], description="Percentiles to compute")
    
    # Processing options
    use_masks: bool = Field(True, description="Use nodata masks in calculations")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    
    # Output format
    output_format: str = Field("csv", description="Output format (csv, json, xlsx)")
    include_geometry: bool = Field(False, description="Include geometry in output")


# Visualization Configuration
class VisualizationConfig(BaseConfig):
    """Configuration for data visualization."""
    
    # Input/Output
    input_dir: Path = Field(..., description="Input directory containing data")
    output_dir: Path = Field(..., description="Output directory for visualizations")
    
    # Visualization options
    create_images: bool = Field(True, description="Create raster images")
    create_plots: bool = Field(True, description="Create statistical plots")
    create_maps: bool = Field(True, description="Create interactive maps")
    
    # Image settings
    dpi: int = Field(300, ge=72, description="Image resolution in DPI")
    figsize: Tuple[float, float] = Field((12, 8), description="Figure size in inches")
    colormap: str = Field("viridis", description="Default colormap")
    
    # Classification
    classification_method: ClassificationMethod = Field(
        ClassificationMethod.AUTO, 
        description="Data classification method"
    )
    n_classes: int = Field(5, ge=2, le=20, description="Number of classes")
    
    # Processing options
    downsample: bool = Field(True, description="Downsample large rasters for visualization")
    max_pixels: int = Field(1000000, ge=10000, description="Maximum pixels for visualization")
    overwrite: bool = Field(False, description="Overwrite existing visualizations")
    
    # Map options
    include_basemap: bool = Field(True, description="Include basemap in maps")
    basemap_alpha: float = Field(0.7, ge=0, le=1, description="Basemap transparency")


# Processing Configuration
class ProcessingConfig(BaseConfig):
    """Main processing configuration."""
    
    # Processing stages
    stages: List[str] = Field(..., description="Processing stages to run")
    
    # Resource management
    max_workers: int = Field(4, ge=1, le=16, description="Maximum number of worker processes")
    memory_limit_gb: Optional[float] = Field(None, ge=1.0, description="Memory limit in GB")
    
    # Error handling
    stop_on_error: bool = Field(True, description="Stop pipeline on error")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    
    @field_validator('stages')
    @classmethod
    def validate_stages(cls, v):
        valid_stages = {"extract", "clip", "align", "integrate", "enrich", "visualize"}
        invalid_stages = set(v) - valid_stages
        if invalid_stages:
            raise ValueError(f"Invalid stages: {invalid_stages}. Valid stages: {valid_stages}")
        return v


# Workflow Configuration
class WorkflowConfig(BaseConfig):
    """Complete workflow configuration."""
    
    # Metadata
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field("1.0", description="Workflow version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    # Core components
    aoi: AOIConfig = Field(..., description="AOI configuration")
    processing: ProcessingConfig = Field(..., description="Processing configuration")
    
    # Data sources
    data_sources: Dict[str, DataSourceConfig] = Field(..., description="Data source configurations")
    
    # Stage configurations (optional - only needed if stage is enabled)
    extraction: Optional[ExtractionConfig] = Field(None, description="Extraction configuration")
    clipping: Optional[ClippingConfig] = Field(None, description="Clipping configuration")
    alignment: Optional[AlignmentConfig] = Field(None, description="Alignment configuration")
    enrichment: Optional[StatisticalEnrichmentConfig] = Field(None, description="Enrichment configuration")
    visualization: Optional[VisualizationConfig] = Field(None, description="Visualization configuration")
    
    # Directory paths
    base_dir: Path = Field(Path.cwd(), description="Base working directory")
    source_dir: Path = Field(..., description="Source data directory")
    output_dir: Path = Field(..., description="Main output directory")
    
    # Derived directories
    @property
    def extracted_dir(self) -> Path:
        return self.output_dir / "extracted"
    
    @property
    def clipped_dir(self) -> Path:
        return self.output_dir / "clipped"
    
    @property
    def aligned_dir(self) -> Path:
        return self.output_dir / "aligned"
    
    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"
    
    @property
    def analysis_ready_dir(self) -> Path:
        return self.output_dir / "analysis_ready"
    
    @property
    def aoi_dir(self) -> Path:
        return self.output_dir / "aoi"
    
    @model_validator(mode='after')
    def validate_stage_dependencies(self):
        """Validate that required stage configurations are provided."""
        stage_configs = {
            "extract": self.extraction,
            "clip": self.clipping,
            "align": self.alignment,
            "enrich": self.enrichment,
            "visualize": self.visualization
        }
        
        for stage in self.processing.stages:
            if stage in stage_configs and stage_configs[stage] is None:
                raise ValueError(f"Stage '{stage}' is enabled but no configuration provided")
        
        return self
    
    def get_stage_config(self, stage_name: str) -> Optional[BaseConfig]:
        """Get configuration for a specific stage."""
        stage_configs = {
            "extract": self.extraction,
            "clip": self.clipping,
            "align": self.alignment,
            "enrich": self.enrichment,
            "visualize": self.visualization
        }
        return stage_configs.get(stage_name)
    
    def has_stage(self, stage_name: str) -> bool:
        """Check if a stage is included in the workflow."""
        return stage_name in self.processing.stages
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save workflow configuration to YAML file."""
        import yaml
        
        output_path = Path(output_path)
        config_dict = self.model_dump(mode='json', exclude={'created_at'})
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'WorkflowConfig':
        """Load workflow configuration from YAML file."""
        import yaml
        
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
class OpenBuildingsExtractionConfig(BaseConfig):
    """Configuration for Open Buildings dataset extraction via Earth Engine."""
    
    # Required inputs (user must provide)
    aoi_file: Path = Field(..., description="Area of Interest boundary file")
    output_dir: Path = Field(..., description="Output directory for extracted buildings")
    
    # Authentication (flexible options for academic teams)
    service_account_key: Optional[Path] = Field(
        None, 
        description="Path to service account key JSON file"
    )
    project_id: Optional[str] = Field(
        None,
        description="Google Cloud Project ID (can be inferred from service account)"
    )
    auth_method: EarthEngineAuthMethod = Field(
        EarthEngineAuthMethod.SERVICE_ACCOUNT,
        description="Authentication method to use"
    )
    
    # Core extraction settings with sensible defaults
    dataset_version: OpenBuildingsDataset = Field(
        OpenBuildingsDataset.V3, 
        description="Open Buildings dataset version"
    )
    confidence_threshold: float = Field(
        0.75, 
        ge=0.5, 
        le=1.0,
        description="Minimum building confidence threshold (0.5-1.0)"
    )
    
    # Building filtering
    min_area_m2: Optional[float] = Field(
        10.0, 
        ge=0, 
        description="Minimum building area in m²"
    )
    max_area_m2: Optional[float] = Field(
        100000.0, 
        description="Maximum building area in m² (None for no limit)"
    )
    
    # Export options
    export_format: BuildingExportFormat = Field(
        BuildingExportFormat.GEOJSON, 
        description="Export format for building data"
    )
    include_confidence: bool = Field(
        True, 
        description="Include confidence scores in output"
    )
    include_area: bool = Field(
        True, 
        description="Include calculated area in output"
    )
    include_plus_codes: bool = Field(
        True, 
        description="Include Google Plus Codes in output"
    )
    
    # Processing limits
    max_features: Optional[int] = Field(
        None, 
        description="Maximum number of features to export (None for no limit)"
    )
    chunk_size: int = Field(
        1000, 
        ge=100, 
        le=10000,
        description="Processing chunk size for large extractions"
    )
    
    # Processing options
    overwrite_existing: bool = Field(
        False, 
        description="Overwrite existing output files"
    )
    create_index: bool = Field(
        True, 
        description="Create spatial index for output data"
    )
    
    # Timeout and retry settings
    timeout_minutes: int = Field(
        30, 
        ge=5, 
        le=120,
        description="Timeout for Earth Engine operations in minutes"
    )
    retry_attempts: int = Field(
        3, 
        ge=1, 
        le=5,
        description="Number of retry attempts for failed operations"
    )

    @field_validator('aoi_file')
    @classmethod
    def validate_aoi_file_exists(cls, v):
        """Validate that AOI file exists."""
        if not Path(v).exists():
            raise ValueError(f"AOI file does not exist: {v}")
        return v

    @field_validator('service_account_key')
    @classmethod
    def validate_service_account_key(cls, v):
        """Validate service account key file if provided."""
        if v is not None and not Path(v).exists():
            raise ValueError(f"Service account key file does not exist: {v}")
        return v

    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold and provide guidance."""
        if v < 0.7:
            import warnings
            warnings.warn(
                f"Confidence threshold {v} is below 0.7. "
                "Lower thresholds may include low-quality building detections."
            )
        return v

    @model_validator(mode='after')
    def validate_authentication_setup(self):
        """Validate authentication configuration."""
        # If service account is specified, ensure it's the right auth method
        if self.service_account_key and self.auth_method != EarthEngineAuthMethod.SERVICE_ACCOUNT:
            import warnings
            warnings.warn(
                "Service account key provided but auth_method is not 'service_account'. "
                "Setting auth_method to 'service_account'."
            )
            self.auth_method = EarthEngineAuthMethod.SERVICE_ACCOUNT
        
        return self
    
    def get_output_file_path(self) -> Path:
        """Get the full output file path based on configuration."""
        extension_map = {
            BuildingExportFormat.GEOJSON: "geojson",
            BuildingExportFormat.SHAPEFILE: "shp",
            BuildingExportFormat.CSV: "csv"
        }
        extension = extension_map[self.export_format]
        return self.output_dir / f"open_buildings.{extension}"
    
    def get_academic_setup_guidance(self) -> str:
        """Get setup guidance for academic users."""
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
        """
        return guidance.strip()