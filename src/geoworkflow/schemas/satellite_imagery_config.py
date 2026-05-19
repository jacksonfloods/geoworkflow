"""
Configuration model for satellite imagery extraction from Google Earth Engine.

Extracts optical RGB satellite imagery from Sentinel-2 MSI Level-2A
via Google Earth Engine for sub-Saharan Africa research.
"""

from pathlib import Path
from typing import List, Optional, Union, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator


class SatelliteImageryConfig(BaseModel):
    """
    Configuration for satellite imagery extraction via Google Earth Engine.

    Example:
        # Single AOI mode
        config = SatelliteImageryConfig(
            aoi_file=Path("nairobi.geojson"),
            output_dir=Path("./outputs"),
            start_date="2024-01-01",
            end_date="2024-06-30"
        )

        # Batch mode: AfricaPolis agglomerations
        config = SatelliteImageryConfig(
            aoi_file="africapolis",
            country=["KEN", "TZA"],
            city=["Nairobi"],  # Optional city filter
            output_dir=Path("./outputs"),
            start_date="2024-01-01",
            end_date="2024-06-30"
        )

        # Batch mode: Country boundaries
        config = SatelliteImageryConfig(
            aoi_file="countries",
            country=["KEN", "TZA", "UGA"],
            output_dir=Path("./outputs"),
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
    """

    # ==================== REQUIRED INPUTS ====================
    aoi_file: Union[Path, str] = Field(
        ...,
        description="Path to AOI file OR 'africapolis' for agglomerations OR 'countries' for country boundaries"
    )

    output_dir: Path = Field(
        ...,
        description="Directory for output GeoTIFF files"
    )

    start_date: str = Field(
        ...,
        description="Start date for imagery search (YYYY-MM-DD format)"
    )

    end_date: str = Field(
        ...,
        description="End date for imagery search (YYYY-MM-DD format)"
    )

    # ==================== BATCH PROCESSING PARAMETERS ====================
    country: Optional[Union[List[str], str]] = Field(
        default=None,
        description="ISO3 country codes list or 'all' (required for batch modes: 'africapolis' or 'countries')"
    )

    city: Optional[List[str]] = Field(
        default=None,
        description="City/agglomeration names to filter (optional, only for 'africapolis' mode)"
    )

    # ==================== IMAGERY SETTINGS ====================
    resolution_m: int = Field(
        default=10,
        ge=10,
        le=500,
        description="Output resolution in meters (10m native for Sentinel-2 RGB)"
    )

    max_cloud_probability: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Maximum cloud probability percentage for scene-level filtering"
    )

    apply_cloud_mask: bool = Field(
        default=True,
        description="Apply per-pixel cloud masking using SCL band"
    )

    # ==================== EXPORT OPTIONS ====================
    scale_to_uint8: bool = Field(
        default=True,
        description="Scale reflectance to 0-255 for RGB visualization"
    )

    clip_to_aoi: bool = Field(
        default=True,
        description="Clip output raster to AOI boundary"
    )

    output_crs: str = Field(
        default="EPSG:4326",
        description="Output CRS (EPSG:4326 for geographic, ESRI:102022 for Africa Albers)"
    )

    compression: Literal["lzw", "deflate", "none"] = Field(
        default="lzw",
        description="GeoTIFF compression method"
    )

    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )

    buffer_aoi_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Buffer AOI by N meters before extraction"
    )

    # ==================== EARTH ENGINE AUTH ====================
    service_account_key: Optional[Path] = Field(
        default=None,
        description="Path to GCP service account key JSON file"
    )

    service_account_email: Optional[str] = Field(
        default=None,
        description="Service account email (required when using service_account_key)"
    )

    project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud Project ID for Earth Engine"
    )

    # ==================== PROCESSING OPTIONS ====================
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for batch processing"
    )

    tile_size_m: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Tile size in meters for large area exports (areas larger than this will be tiled)"
    )

    # ==================== VALIDATORS ====================
    @field_validator('aoi_file')
    @classmethod
    def validate_aoi(cls, v):
        """Convert string to Path if not batch mode keyword."""
        if isinstance(v, str) and v.lower() in ["africapolis", "countries"]:
            return v.lower()

        # Otherwise treat as path
        path = Path(v)
        if not path.exists():
            raise ValueError(f"AOI file not found: {path}")
        if path.suffix.lower() not in ['.geojson', '.json', '.shp', '.gpkg']:
            raise ValueError(f"AOI file must be GeoJSON, Shapefile, or GeoPackage. Got: {path.suffix}")
        return path

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD."""
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD format.")
        return v

    @field_validator('service_account_key')
    @classmethod
    def validate_service_account_key(cls, v):
        """Validate service account key file exists."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Service account key file not found: {path}")
            return path
        return v

    @model_validator(mode='after')
    def validate_date_range(self):
        """Ensure end_date is after start_date."""
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        if end < start:
            raise ValueError("end_date must be after start_date")
        return self

    @model_validator(mode='after')
    def validate_batch_mode_requirements(self):
        """Ensure country is provided for batch modes (africapolis or countries)."""
        if isinstance(self.aoi_file, str) and self.aoi_file in ["africapolis", "countries"]:
            if self.country is None:
                mode_name = "AfricaPolis" if self.aoi_file == "africapolis" else "Countries"
                raise ValueError(
                    f"{mode_name} mode requires 'country' parameter. "
                    "Provide ISO3 codes list or 'all'."
                )
            if isinstance(self.country, list) and len(self.country) == 0:
                raise ValueError("Country list cannot be empty for batch mode.")
        return self

    @model_validator(mode='after')
    def validate_auth_configuration(self):
        """Validate authentication configuration."""
        if self.service_account_key and not self.service_account_email:
            raise ValueError(
                "service_account_email is required when using service_account_key"
            )
        return self

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = 'forbid'  # Raise error on unknown fields
