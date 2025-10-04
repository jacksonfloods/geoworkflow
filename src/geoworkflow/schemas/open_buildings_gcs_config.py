"""
Configuration model for GCS-based Open Buildings extraction.

This configuration supports direct Google Cloud Storage access for
building footprint extraction from the Open Buildings v3 dataset.
"""
from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class OpenBuildingsGCSConfig(BaseModel):
    """
    Configuration for GCS-based Open Buildings extraction.
    
    This is the primary extraction method for Open Buildings data,
    significantly faster than the Earth Engine approach.
    
    Example:
        >>> config = OpenBuildingsGCSConfig(
        ...     aoi_file=Path("study_area.geojson"),
        ...     output_dir=Path("./buildings/"),
        ...     confidence_threshold=0.75,
        ...     num_workers=4
        ... )
    """
    
    # ==================== Required Inputs ====================
    aoi_file: Path = Field(
        ...,
        description="Area of Interest boundary file (GeoJSON, Shapefile, etc.)"
    )
    
    output_dir: Path = Field(
        ...,
        description="Output directory for extracted buildings"
    )
    
    # ==================== Data Source Settings ====================
    data_type: Literal["polygons", "points"] = Field(
        default="polygons",
        description="Type of building data to extract (polygons include geometry)"
    )
    
    s2_level: int = Field(
        default=6,
        ge=4,
        le=8,
        description="S2 cell level - must match GCS bucket structure (default: 6)"
    )
    
    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/polygons_s2_level_6_gzip_no_header",
        description="GCS path to Open Buildings data"
    )
    
    # ==================== Filtering Options ====================
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum building confidence score (0.5-1.0)"
    )
    
    min_area_m2: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum building area in square meters"
    )
    
    max_area_m2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum building area in square meters (None for no limit)"
    )
    
    # ==================== Export Settings ====================
    export_format: Literal["geojson", "shapefile", "csv", "geoparquet"] = Field(
        default="geojson",
        description="Output file format"
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores in output"
    )
    
    include_area: bool = Field(
        default=True,
        description="Include area calculations in output"
    )
    
    include_plus_codes: bool = Field(
        default=True,
        description="Include Plus Codes in output"
    )
    
    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )
    
    # ==================== Performance Settings ====================
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for downloading S2 cells"
    )
    
    chunk_size: int = Field(
        default=2_000_000,
        ge=100_000,
        description="Number of records to process per chunk"
    )
    
    # ==================== Authentication (optional for public data) ====================
    service_account_key: Optional[Path] = Field(
        default=None,
        description="Path to GCS service account key (not required for Open Buildings)"
    )
    
    use_anonymous_access: bool = Field(
        default=True,
        description="Use anonymous access for public data (recommended for Open Buildings)"
    )
    
    # ==================== Validators ====================
    @field_validator('aoi_file')
    @classmethod
    def validate_aoi_exists(cls, v: Path) -> Path:
        """Validate that AOI file exists."""
        if not v.exists():
            raise ValueError(f"AOI file not found: {v}")
        return v
    
    @field_validator('service_account_key')
    @classmethod
    def validate_service_account(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate service account key if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"Service account key file not found: {v}")
        return v
    
    @field_validator('max_area_m2')
    @classmethod
    def validate_area_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that max_area > min_area if both are set."""
        if v is not None:
            min_area = info.data.get('min_area_m2', 0.0)
            if v <= min_area:
                raise ValueError(
                    f"max_area_m2 ({v}) must be greater than min_area_m2 ({min_area})"
                )
        return v
    
    @field_validator('gcs_bucket_path')
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Validate GCS bucket path format."""
        if not v.startswith('gs://'):
            raise ValueError(f"GCS path must start with 'gs://': {v}")
        return v
    
    # ==================== Helper Methods ====================
    def get_output_file_path(self) -> Path:
        """
        Generate output file path based on format.
        
        Returns:
            Path object for the output file
        """
        extensions = {
            "geojson": ".geojson",
            "shapefile": ".shp",
            "csv": ".csv",
            "geoparquet": ".parquet"
        }
        extension = extensions.get(self.export_format, ".geojson")
        return self.output_dir / f"open_buildings{extension}"
    
    def get_gcs_file_pattern(self) -> str:
        """
        Get the GCS file pattern for S2 cells.
        
        Returns:
            String pattern for GCS files
        """
        return f"{self.gcs_bucket_path}/*.csv.gz"
    
    def estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB based on settings.
        
        Returns:
            Estimated memory usage in megabytes
        """
        # Rough estimate: chunk_size * 200 bytes per record
        memory_per_worker = (self.chunk_size * 200) / (1024 * 1024)
        total_memory = memory_per_worker * self.num_workers
        return total_memory
    
    def summary(self) -> dict:
        """
        Get a summary of key configuration settings.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "aoi_file": str(self.aoi_file),
            "output_dir": str(self.output_dir),
            "output_format": self.export_format,
            "confidence_threshold": self.confidence_threshold,
            "area_filter": f"{self.min_area_m2}-{self.max_area_m2 or 'unlimited'} m²",
            "parallel_workers": self.num_workers,
            "s2_level": self.s2_level,
            "data_source": self.gcs_bucket_path,
            "estimated_memory_mb": round(self.estimate_memory_usage(), 2)
        }
    
    class Config:
        """Pydantic model configuration."""
        # Allow Path objects
        arbitrary_types_allowed = True
        # Enable validation on assignment
        validate_assignment = True
        # JSON schema extras
        json_schema_extra = {
            "example": {
                "aoi_file": "study_area.geojson",
                "output_dir": "./buildings/",
                "confidence_threshold": 0.75,
                "min_area_m2": 10.0,
                "export_format": "geojson",
                "num_workers": 4
            }
        }


# Alternative configuration for points (lighter-weight)
class OpenBuildingsGCSPointsConfig(OpenBuildingsGCSConfig):
    """
    Specialized configuration for extracting building points only.
    
    This is faster and requires less storage than polygon extraction.
    """
    
    data_type: Literal["points"] = Field(
        default="points",
        description="Extract building centroids only"
    )
    
    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/points_s2_level_6_gzip",
        description="GCS path to Open Buildings points data"
    )
    
    # Points don't have geometry column, so this is always CSV
    export_format: Literal["csv", "geojson"] = Field(
        default="csv",
        description="Output format (CSV for tabular, GeoJSON to create points)"
    )
