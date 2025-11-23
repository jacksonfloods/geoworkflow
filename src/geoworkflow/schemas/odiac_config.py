"""
Configuration model for ODIAC CO2 emissions extraction.

Extracts fossil fuel CO2 emissions data from the ODIAC dataset via
NASA's GHG Center STAC/Raster APIs.
"""

from pathlib import Path
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ODIACConfig(BaseModel):
    """
    Configuration for ODIAC fossil fuel CO2 emissions extraction.

    Example:
        # Single AOI mode
        config = ODIACConfig(
            aoi_file=Path("nairobi.geojson"),
            output_dir=Path("./outputs"),
            year=2022
        )

        # Batch mode: AfricaPolis agglomerations
        config = ODIACConfig(
            aoi_file="africapolis",
            country=["KEN", "TZA"],
            city=["Nairobi"],  # Optional city filter
            output_dir=Path("./outputs"),
            year=2022
        )

        # Batch mode: Country boundaries
        config = ODIACConfig(
            aoi_file="countries",
            country=["KEN", "TZA", "UGA"],
            output_dir=Path("./outputs"),
            year=2022
        )
    """

    # ==================== REQUIRED INPUTS ====================
    aoi_file: Union[Path, str] = Field(
        ...,
        description="Path to AOI file OR 'africapolis' for agglomerations OR 'countries' for country boundaries"
    )

    output_dir: Path = Field(
        ...,
        description="Directory for output files"
    )

    year: int = Field(
        ...,
        ge=2000,
        le=2023,
        description="Year of ODIAC data to extract (2000-2023)"
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

    # ==================== DATA SOURCE ====================
    stac_api_url: str = Field(
        default="https://earth.gov/ghgcenter/api/stac",
        description="NASA GHG Center STAC API endpoint"
    )

    raster_api_url: str = Field(
        default="https://earth.gov/ghgcenter/api/raster",
        description="NASA GHG Center Raster API endpoint"
    )

    collection_name: str = Field(
        default="odiac-ffco2-monthgrid-v2023",
        description="STAC collection name"
    )

    asset_name: str = Field(
        default="co2-emissions",
        description="Asset name within STAC items"
    )

    api_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API request timeout in seconds"
    )

    # ==================== SPATIAL PROCESSING ====================
    buffer_aoi_meters: float = Field(
        default=0.0,
        ge=0.0,
        description="Buffer AOI by N meters before extraction (useful for edge cases)"
    )

    # ==================== EXPORT OPTIONS ====================
    export_format: Literal["geotiff", "cog"] = Field(
        default="geotiff",
        description="Output raster format: 'geotiff' or 'cog' (cloud-optimized)"
    )

    output_crs: str = Field(
        default="ESRI:102022",
        description="CRS for output files (default: Africa Albers Equal Area)"
    )

    export_monthly: bool = Field(
        default=True,
        description="Export individual monthly TIFFs (12 files per city)"
    )

    export_annual: bool = Field(
        default=True,
        description="Export annual average TIFF"
    )

    export_statistics: bool = Field(
        default=True,
        description="Export zonal statistics CSV"
    )

    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )

    # ==================== PROCESSING OPTIONS ====================
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for batch processing"
    )

    compression: Literal["lzw", "deflate", "none"] = Field(
        default="lzw",
        description="TIFF compression method"
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
        if not path.suffix.lower() in ['.geojson', '.json', '.shp', '.gpkg']:
            raise ValueError(f"AOI file must be GeoJSON, Shapefile, or GeoPackage. Got: {path.suffix}")
        return path

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator('year')
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Validate year is within ODIAC data availability."""
        # ODIAC v2023 covers 2000-2022
        # Could be extended to 2023 depending on data availability
        if v < 2000 or v > 2023:
            raise ValueError(f"Year must be between 2000-2023. Got: {v}")
        return v

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
    def validate_export_options(self):
        """Ensure at least one export option is enabled."""
        if not (self.export_monthly or self.export_annual or self.export_statistics):
            raise ValueError(
                "At least one export option must be enabled: "
                "export_monthly, export_annual, or export_statistics"
            )
        return self

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = 'forbid'  # Raise error on unknown fields
