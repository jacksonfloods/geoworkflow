"""
Configuration model for OSM highway extraction.


Follows the pattern established in open_buildings_gcs_config.py
"""


from pathlib import Path
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import warnings




class OSMHighwaysConfig(BaseModel):
    """
    Configuration for OSM highway extraction via Geofabrik PBF files.
    
    Example:
        config = OSMHighwaysConfig(
            aoi_file=Path("nairobi.geojson"),
            output_dir=Path("./outputs"),
            geofabrik_regions=["kenya"],
            include_attributes=["highway", "surface", "lanes", "name"]
        )
    """
    
    # ==================== REQUIRED INPUTS ====================
    aoi_file: Union[Path, str] = Field(
        ...,
        description="Path to AOI file OR 'africapolis' for batch mode"
    )
    
    output_dir: Path = Field(
        ...,
        description="Directory for output files"
    )
    
    # ==================== BATCH PROCESSING PARAMETERS ====================
    country: Optional[Union[List[str], str]] = Field(
        default=None,
        description="ISO3 country codes list or 'all' (required for AfricaPolis mode)"
    )
    
    city: Optional[List[str]] = Field(
        default=None,
        description="Agglomeration names to filter (optional, AND logic with country)"
    )
    
    # ==================== DATA SOURCE ====================
    geofabrik_regions: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of Geofabrik region names (e.g., ['kenya', 'tanzania']). "
            "If None, auto-detect from AOI bounds."
        )
    )

    pbf_cache_dir: Path = Field(
        default= Path(__file__).resolve().parents[4] /"data"/ ".cache" / "osm",
        description="Directory to cache downloaded PBF files"
    )
    
    force_redownload: bool = Field(
        default=False,
        description="Force re-download of PBF files even if cached"
    )
    
    max_cache_age_days: Optional[int] = Field(
        default=None,
        description=(
            "Warn if cached PBF is older than this many days. "
            "None = no age warning. User must manually decide if too old."
        )
    )
    
    # ==================== HIGHWAY FILTERING ====================
    highway_types: Union[List[str], Literal["all"]] = Field(
        default="all",
        description=(
            "Highway types to extract. Options:\n"
            "- 'all': Extract all highway types\n"
            "- List of types: ['motorway', 'trunk', 'primary', 'secondary', "
            "'tertiary', 'residential', 'service', 'footway', 'cycleway', ...]\n"
            "See: https://wiki.openstreetmap.org/wiki/Key:highway"
        )
    )
    
    include_attributes: Union[List[str], Literal["all"]] = Field(
        default=["highway", "surface", "lanes", "name"],
        description=(
            "OSM attributes to include in output:\n"
            "- Default: ['highway', 'surface', 'lanes', 'name']\n"
            "- 'all': Keep all available OSM attributes (~50+ columns)\n"
            "- Custom list: ['highway', 'maxspeed', 'oneway', 'bridge', ...]\n\n"
            "Common attributes: highway, name, ref, surface, lanes, maxspeed, "
            "oneway, width, bridge, tunnel, access, lit, tracktype"
        )
    )
    
    # ==================== SPATIAL PROCESSING ====================
    clip_to_aoi: bool = Field(
        default=True,
        description="Clip geometries to AOI boundary (vs. just filter intersecting)"
    )
    
    buffer_aoi_meters: float = Field(
        default=0.0,
        ge=0.0,
        description="Buffer AOI by N meters before extraction (useful for edge cases)"
    )
    
    # ==================== EXPORT OPTIONS ====================
    export_format: Literal["geojson", "shapefile", "geoparquet", "csv"] = Field(
        default="geojson",
        description="Output file format"
    )
    
    output_crs: str = Field(
        default="EPSG:4326",
        description="CRS for output file (e.g., 'EPSG:4326', 'EPSG:32736')"
    )
    
    simplify_tolerance_meters: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Simplify geometries with tolerance in meters. "
            "None = no simplification. Useful for reducing file size."
        )
    )
    
    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )
    
    create_spatial_index: bool = Field(
        default=True,
        description="Create spatial index for output (shapefile .shx, geoparquet index)"
    )
    
    # ==================== PROCESSING OPTIONS ====================
    num_workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of parallel workers (currently unused, reserved for future)"
    )
    
    chunk_size: int = Field(
        default=50000,
        ge=1000,
        description="Process highways in chunks of N features (memory management)"
    )
    
    # ==================== VALIDATORS ====================
    @field_validator('aoi_file')
    @classmethod
    def validate_aoi(cls, v):
        """Convert string to Path if not AfricaPolis keyword."""
        if isinstance(v, str) and v.lower() == "africapolis":
            return v.lower()  # Normalize to lowercase
        
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
    
    @field_validator('pbf_cache_dir')
    @classmethod
    def validate_cache_dir(cls, v: Path) -> Path:
        """Create cache directory if it doesn't exist."""
        v = v.expanduser()  # Expand ~ to home directory
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('highway_types')
    @classmethod
    def validate_highway_types(cls, v: Union[List[str], str]) -> Union[List[str], str]:
        """Validate highway types."""
        if v == "all":
            return v
        
        # Valid OSM highway types (common ones)
        valid_types = {
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'unclassified', 'residential', 'service',
            'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
            'living_street', 'pedestrian', 'track', 'road',
            'footway', 'bridleway', 'steps', 'path', 'cycleway'
        }
        
        if not isinstance(v, list):
            raise ValueError("highway_types must be 'all' or a list of highway types")
        
        # Warn about invalid types but don't fail
        invalid = [t for t in v if t not in valid_types]
        if invalid:
            warnings.warn(
                f"Unrecognized highway types (may be valid but uncommon): {invalid}. "
                f"See: https://wiki.openstreetmap.org/wiki/Key:highway"
            )
        
        return v
    
    @field_validator('include_attributes')
    @classmethod
    def validate_include_attributes(cls, v: Union[List[str], str]) -> Union[List[str], str]:
        """Validate attribute list."""
        if v == "all":
            return v
        
        if not isinstance(v, list):
            raise ValueError("include_attributes must be 'all' or a list of attribute names")
        
        if len(v) == 0:
            raise ValueError("include_attributes cannot be empty. Use 'all' or specify attributes.")
        
        # Always include 'highway' attribute
        if 'highway' not in v:
            warnings.warn("Adding 'highway' to include_attributes (required for highway type)")
            v = ['highway'] + v
        
        return v
    
    @model_validator(mode='after')
    def validate_geofabrik_regions(self) -> 'OSMHighwaysConfig':
        """Validate region names if provided."""
        if self.geofabrik_regions is not None:
            # Import here to avoid circular dependency
            from geoworkflow.utils.geofabrik_utils import GEOFABRIK_REGIONS
            
            invalid = [r for r in self.geofabrik_regions if r not in GEOFABRIK_REGIONS]
            if invalid:
                available = ', '.join(sorted(GEOFABRIK_REGIONS.keys())[:20])
                raise ValueError(
                    f"Invalid Geofabrik regions: {invalid}. "
                    f"Available regions include: {available}... "
                    f"See: https://download.geofabrik.de/"
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_africapolis_requirements(self):
        """Ensure country is provided for AfricaPolis mode."""
        if isinstance(self.aoi_file, str) and self.aoi_file == "africapolis":
            if self.country is None:
                raise ValueError(
                    "AfricaPolis mode requires 'country' parameter. "
                    "Provide ISO3 codes list or 'all'."
                )
            if isinstance(self.country, list) and len(self.country) == 0:
                raise ValueError("Country list cannot be empty for AfricaPolis mode.")
        return self
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = 'forbid'  # Raise error on unknown fields