"""
OSM Highway extraction processor using Geofabrik PBF files.

This processor extracts highway networks from OpenStreetMap data via direct
download of regional PBF files from Geofabrik. It follows the established
TemplateMethodProcessor pattern used in the Open Buildings GCS processor.

Key features:
- Downloads and caches regional PBF files
- Supports multi-region AOIs
- Filters by highway type and attributes
- Exports to multiple formats
- Tracks download dates for data freshness

Performance:
- Small urban area (10 km²): ~10-30 seconds
- Medium city (100 km²): ~30-60 seconds
- Large region (1000 km²): ~2-5 minutes
- Download time (first run): ~1-5 minutes per region

Example:
    config = OSMHighwaysConfig(
        aoi_file=Path("lagos.geojson"),
        output_dir=Path("./outputs"),
        highway_types=["motorway", "trunk", "primary"],
        include_attributes=["highway", "name", "surface", "lanes"]
    )
    
    processor = OSMHighwaysProcessor(config)
    result = processor.process()
    print(f"Extracted {result.processed_count} highway segments")
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
import json

try:
    import pyrosm
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape
    HAS_REQUIRED_LIBS = True
except ImportError:
    HAS_REQUIRED_LIBS = False

from geoworkflow.schemas.processing_result import BatchProcessResult
from geoworkflow.utils.config_loader import ConfigLoader
from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import (
    ExtractionError,
    ValidationError,
    ConfigurationError
)
from geoworkflow.schemas.osm_highways_config import OSMHighwaysConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.geofabrik_utils import (
    detect_regions_from_aoi,
    get_cached_pbf,
    list_cached_pbfs
)
from geoworkflow.utils.osm_utils import (
    filter_highways_by_type,
    select_highway_attributes,
    clean_highway_attributes,
    validate_highway_geometries,
    deduplicate_highways,
    clip_highways_to_aoi,
    calculate_highway_length,
    summarize_highway_network
)
from geoworkflow.utils.resource_utils import ensure_directory
from geoworkflow.schemas.processing_result import BatchProcessResult
from geoworkflow.utils.config_loader import ConfigLoader

# ISO3 country codes to Geofabrik region name mapping
ISO3_TO_GEOFABRIK = {
    "DZA": "algeria", "AGO": "angola", "BEN": "benin", "BWA": "botswana",
    "BFA": "burkina-faso", "BDI": "burundi", "CMR": "cameroon", "CPV": "cape-verde",
    "CAF": "central-african-republic", "TCD": "chad", "COM": "comoros",
    "COG": "congo-brazzaville", "COD": "congo-democratic-republic", "DJI": "djibouti",
    "EGY": "egypt", "GNQ": "equatorial-guinea", "ERI": "eritrea", "ETH": "ethiopia",
    "GAB": "gabon", "GHA": "ghana", "GIN": "guinea", "GNB": "guinea-bissau",
    "CIV": "ivory-coast", "KEN": "kenya", "LSO": "lesotho", "LBR": "liberia",
    "LBY": "libya", "MDG": "madagascar", "MWI": "malawi", "MLI": "mali",
    "MRT": "mauritania", "MUS": "mauritius", "MAR": "morocco", "MOZ": "mozambique",
    "NAM": "namibia", "NER": "niger", "NGA": "nigeria", "RWA": "rwanda",
    "SHN": "saint-helena-ascension-and-tristan-da-cunha", "STP": "sao-tome-and-principe",
    "SEN": "senegal", "SYC": "seychelles", "SLE": "sierra-leone", "SOM": "somalia",
    "ZAF": "south-africa", "SSD": "south-sudan", "SDN": "sudan", "TZA": "tanzania",
    "TGO": "togo", "TUN": "tunisia", "UGA": "uganda", "ZMB": "zambia", "ZWE": "zimbabwe"
}


class OSMHighwaysProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Extract highway networks from OpenStreetMap via Geofabrik PBF files.
    
    This processor provides fast, reliable extraction of highway data for
    arbitrary AOIs by downloading regional PBF files and filtering locally.
    
    Workflow:
    1. Load AOI and detect/validate regions
    2. Download/cache regional PBF files
    3. Extract all highways from PBF(s)
    4. Filter by spatial intersection with AOI
    5. Filter by highway type if specified
    6. Select requested attributes
    7. Validate and clean geometries
    8. Export to requested format
    
    The processor handles multi-region AOIs automatically by downloading
    all necessary PBF files and merging results.
    """
    
    def __init__(
        self,
        config: Union[OSMHighwaysConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize OSM Highways processor.
        
        Args:
            config: Configuration (OSMHighwaysConfig or dict)
            logger: Optional logger instance
        """
        # Check dependencies
        if not HAS_REQUIRED_LIBS:
            raise ImportError(
                "Required libraries not available. Install with: "
                "pip install pyrosm geopandas shapely"
            )
        
        # Convert Pydantic model to dict for base class
        if isinstance(config, OSMHighwaysConfig):
            config_dict = config.model_dump(mode='json')
            self.highways_config = config
        else:
            config_dict = config
            self.highways_config = OSMHighwaysConfig(**config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.aoi_gdf: Optional[gpd.GeoDataFrame] = None
        self.aoi_crs: None
        self.regions: List[str] = []
        self.pbf_files: List[Path] = []
        self.pbf_metadata: List[Any] = []
        self.highways_raw: Optional[gpd.GeoDataFrame] = None
        self.highways_filtered: Optional[gpd.GeoDataFrame] = None
        self.output_file: Optional[Path] = None

    def process(self):
        if self._is_africapolis_mode():
            geometries = self._load_batch_geometries()  # Returns list of (geom, name, iso3)
        else:
            geometries = self._load_single_geometry()   # Returns list with 1 item        
        return self._process_geometries(geometries)

    def process_data(self) -> Dict[str, Any]:
        raise NotImplementedError("Use process() method directly")

    def _is_africapolis_mode(self) -> bool:
        """Check if running in AfricaPolis batch mode."""
        return isinstance(self.highways_config.aoi_file, str) and self.highways_config.aoi_file == "africapolis"
    

    def _process_geometries(self, geometries:list):
        """Process all geometries grouped by country."""
        succeeded = []
        failed = {}

        by_country = self._group_by_country(geometries)

        for iso3, geom_list in by_country.items():
            try:
                pbf_data, _ = self._load_country_pbf(iso3)
            except Exception as e:
                for geom, name in geom_list:
                    failed[name] = f"PBF load failed: {str(e)}"
                continue
            
            for geom, name in geom_list:
                try:
                    highways = self._extract_highways(pbf_data,geom)
                    self._export(highways, name, iso3)
                    succeeded.append(name)
                except Exception as e:
                    failed[name] = str(e)
                    self.logger.error(f"✗ {name}: {e}")
        
        # Return appropriate result type
        if self._is_africapolis_mode():
            return BatchProcessResult(
                success=len(succeeded) > 0,
                total_count=len(geometries),
                succeeded_count=len(succeeded),
                failed_count=len(failed),
                succeeded=succeeded,
                failed=failed,
                output_files=[]
            )
        else:
            return ProcessingResult(
                success=len(succeeded) > 0,
                processed_count=len(succeeded),
                message=f"Processed {len(succeeded)} geometries"
            )
        
    def _filter_agglomerations(self, gdf: gpd.GeoDataFrame, columns: Dict[str, str]) -> gpd.GeoDataFrame:
        """Filter by country and city using AND logic."""
        filtered = gdf.copy()
        
        if isinstance(self.highways_config.country, str) and self.highways_config.country.lower() == "all":
            pass
        elif isinstance(self.highways_config.country, list):
            filtered = filtered[filtered[columns["iso3"]].isin(self.highways_config.country)]
        
        if self.highways_config.city:
            filtered = filtered[filtered[columns["name"]].isin(self.highways_config.city)]
        
        if len(filtered) == 0:
            raise ValueError("No agglomerations match the specified filters")
        
        self.logger.info(f"Filters: country={self.highways_config.country}, city={self.highways_config.city}")
        return filtered
    
    def _load_country_pbf(self, iso3: str):
        """Load and parse PBF for a country once."""
        from geoworkflow.utils.geofabrik_utils import get_cached_pbf
        
        region = ISO3_TO_GEOFABRIK.get(iso3, iso3.lower())
        if region not in ISO3_TO_GEOFABRIK.values():
            self.logger.warning(f"ISO3 code {iso3} not found in mapping, using: {region}")
        
        self.logger.info(f"  Loading PBF for {iso3}...")
        pbf_path, metadata = get_cached_pbf(
            region=region,
            cache_dir=self.highways_config.pbf_cache_dir,
            force_redownload=self.highways_config.force_redownload,
            max_age_days=self.highways_config.max_cache_age_days
        )
        
        osm = pyrosm.OSM(str(pbf_path))
        highways_data = osm.get_network(network_type="all")
        
        if highways_data is None or len(highways_data) == 0:
            raise ExtractionError(f"No highways found in {region} PBF")
        
        self.logger.info(f"  Loaded {len(highways_data):,} highway segments")
        return highways_data, osm

    
    def _clip_highways(self, pbf_data: gpd.GeoDataFrame, geometry) -> gpd.GeoDataFrame:
        """Clip highways to geometry from pre-parsed PBF."""

        temp_aoi = gpd.GeoDataFrame({'geometry': [geometry]}, crs=self.aoi_crs)
        
        # Reproject to match PBF data CRS
        if temp_aoi.crs != pbf_data.crs:
            temp_aoi = temp_aoi.to_crs(pbf_data.crs)
        
        buffer_meters = self.highways_config.buffer_aoi_meters or 0
        clipped = clip_highways_to_aoi(pbf_data, temp_aoi, buffer_meters=buffer_meters)
        return clipped
    
    def _get_driver(self) -> str:
        """Get GDAL driver name from export format."""
        driver_map = {"geojson": "GeoJSON", "gpkg": "GPKG", "shapefile": "ESRI Shapefile"}
        return driver_map.get(self.highways_config.export_format, "GPKG")
    
    def _get_extension(self) -> str:
        """Get file extension from export format."""
        ext_map = {"geojson": ".geojson", "gpkg": ".gpkg", "shapefile": ".shp"}
        return ext_map.get(self.highways_config.export_format, ".gpkg")
         
    def _cleanup_custom_processing(self) -> None:
        """
        Cleanup after processing.
        
        Currently minimal - Pyrosm handles cleanup internally.
        Could add: temp file cleanup, cache statistics, etc.
        """
        self.logger.info("Processing cleanup complete")

    def _load_batch_geometries(self):
        """Load agglomerations from AfricaPolis."""
        agglo_path = ConfigLoader.get_africapolis_path()
        columns = ConfigLoader.get_africapolis_columns()
        
        if not agglo_path.exists():
            raise FileNotFoundError(f"AfricaPolis file not found: {agglo_path}")
        
        agglomerations = gpd.read_file(agglo_path)
        self.aoi_crs = agglomerations.crs

        filtered = self._filter_agglomerations(agglomerations, columns)
        
        # Return list of (geometry, name, iso3)
        return [(row.geometry, row[columns["name"]], row[columns["iso3"]]) for _, row in filtered.iterrows()]

    def _load_single_geometry(self):
        """Load single AOI geometry."""
        aoi_gdf = gpd.read_file(self.highways_config.aoi_file)
        self.aoi_crs = aoi_gdf.crs

        # Detect country from geometry
        region_name = detect_regions_from_aoi(aoi_gdf)[0]  
        
        # Reverse lookup: region name -> ISO3
        GEOFABRIK_TO_ISO3 = {v: k for k, v in ISO3_TO_GEOFABRIK.items()}
        iso3 = GEOFABRIK_TO_ISO3.get(region_name, region_name.upper())
        
        # Return list with single item: (geometry, name, iso3, aoi_crs)
        return [(aoi_gdf.union_all(), self.highways_config.aoi_file.stem, iso3)]

    def _group_by_country(self, geometries):
        """Group geometries by ISO3 country code."""
        by_country = {}
        for geom, name, iso3 in geometries:
            if iso3 not in by_country:
                by_country[iso3] = []
            by_country[iso3].append((geom, name))
        return by_country

    def _extract_highways(self, pbf_data, geometry):
        """Extract and filter highways for a single geometry."""
        highways = self._clip_highways(pbf_data, geometry)
        
        if len(highways) == 0:
            raise ExtractionError("No highways found")
        
        if self.highways_config.highway_types != "all":
            highways = filter_highways_by_type(highways, self.highways_config.highway_types)
        
        if self.highways_config.include_attributes != "all":
            highways = select_highway_attributes(highways, self.highways_config.include_attributes)
        
        highways = clean_highway_attributes(highways)
        highways = validate_highway_geometries(highways)
        highways = deduplicate_highways(highways)
        highways = calculate_highway_length(highways)
        
        return highways

    def _export(self, highways, name, iso3):
        """Export highways to file."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        ext = self._get_extension()
        output_file = self.highways_config.output_dir / f"{iso3}_{safe_name}_hwys{ext}"
        
        driver = self._get_driver()
        highways.to_file(output_file, driver=driver)
        
        self.logger.info(f"✓ {name} -> {output_file.name}")