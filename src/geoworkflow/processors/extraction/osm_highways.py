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
        self.regions: List[str] = []
        self.pbf_files: List[Path] = []
        self.pbf_metadata: List[Any] = []
        self.highways_raw: Optional[gpd.GeoDataFrame] = None
        self.highways_filtered: Optional[gpd.GeoDataFrame] = None
        self.output_file: Optional[Path] = None
    
    def process_data(self) -> Dict[str, Any]:
        """
        Main processing method (required abstract method from base class).
        
        The actual implementation follows the template method pattern where
        the base class process() method orchestrates the workflow by calling:
        - validate_inputs() -> _validate_custom_inputs()
        - setup_processing() -> _setup_custom_processing()  
        - process_data() -> this method
        - cleanup_resources() -> _cleanup_custom_processing()
        
        This method executes the core highway extraction and filtering logic.
        
        Returns:
            Dictionary with processing statistics and results
        """
        return self._execute_core_processing()
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate OSM-specific inputs.
        
        Checks:
        - AOI file is readable and valid
        - Output directory is writable
        - Cache directory exists
        - Highway types are valid
        - Export format is supported
        
        Returns:
            Dictionary with validation results and warnings
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate AOI file
        try:
            aoi_test = gpd.read_file(self.highways_config.aoi_file)
            if len(aoi_test) == 0:
                validation_result["errors"].append("AOI file contains no features")
            if aoi_test.crs is None:
                validation_result["warnings"].append("AOI has no CRS, assuming EPSG:4326")
        except Exception as e:
            validation_result["errors"].append(f"Failed to read AOI file: {e}")
        
        # Validate output directory
        if not self.highways_config.output_dir.exists():
            try:
                self.highways_config.output_dir.mkdir(parents=True)
            except Exception as e:
                validation_result["errors"].append(f"Cannot create output directory: {e}")
        
        # Validate cache directory
        if not self.highways_config.pbf_cache_dir.exists():
            try:
                self.highways_config.pbf_cache_dir.mkdir(parents=True)
            except Exception as e:
                validation_result["errors"].append(f"Cannot create cache directory: {e}")
        
        # Validate export format
        supported_formats = ['geojson', 'shapefile', 'geoparquet', 'csv']
        if self.highways_config.export_format not in supported_formats:
            validation_result["errors"].append(
                f"Unsupported export format: {self.highways_config.export_format}. "
                f"Supported: {supported_formats}"
            )
        
        # Check if PBF files are cached
        cached = list_cached_pbfs(self.highways_config.pbf_cache_dir)
        if cached:
            validation_result["warnings"].append(
                f"Found {len(cached)} cached PBF file(s). "
                f"Set force_redownload=True to refresh."
            )
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """
        Setup processing: load AOI, detect regions, download PBFs.
        
        Returns:
            Dictionary with setup information
        """
        setup_info = {"components": []}
        
        try:
            # Load AOI
            self.logger.info(f"Loading AOI from {self.highways_config.aoi_file}")
            self.aoi_gdf = gpd.read_file(self.highways_config.aoi_file)
            
            # Ensure WGS84 for region detection and PBF downloads
            # CRITICAL FIX: Properly check if CRS is WGS84 by comparing normalized CRS
            if self.aoi_gdf.crs is None:
                self.logger.warning("AOI has no CRS, assuming EPSG:4326")
                self.aoi_gdf.set_crs("EPSG:4326", inplace=True)
            else:
                # Check if already in WGS84 by comparing EPSG codes
                from pyproj import CRS
                target_crs = CRS.from_epsg(4326)
                current_crs = CRS(self.aoi_gdf.crs)
                
                if not current_crs.equals(target_crs):
                    self.logger.info(f"Reprojecting AOI from {self.aoi_gdf.crs} to EPSG:4326")
                    self.aoi_gdf = self.aoi_gdf.to_crs("EPSG:4326")
                    self.logger.info(f"AOI reprojected. New bounds: {self.aoi_gdf.total_bounds}")
            
            setup_info["aoi_features"] = len(self.aoi_gdf)
            setup_info["aoi_bounds"] = self.aoi_gdf.total_bounds.tolist()
            setup_info["aoi_crs"] = str(self.aoi_gdf.crs)
            setup_info["components"].append("aoi_loaded")
            
            # Detect or validate regions
            if self.highways_config.geofabrik_regions is None:
                self.logger.info("Auto-detecting Geofabrik regions from AOI...")
                self.regions = detect_regions_from_aoi(self.aoi_gdf)
            else:
                self.regions = self.highways_config.geofabrik_regions
                self.logger.info(f"Using specified regions: {self.regions}")
            
            setup_info["regions"] = self.regions
            setup_info["multi_region"] = len(self.regions) > 1
            
            if len(self.regions) > 1:
                self.logger.warning(
                    f"AOI spans {len(self.regions)} regions: {self.regions}. "
                    "Will download and merge multiple PBF files."
                )
            
            # Download/cache PBF files
            self.logger.info("Checking cache and downloading PBF files if needed...")
            for region in self.regions:
                self.logger.info(f"Processing region: {region}")
                pbf_path, metadata = get_cached_pbf(
                    region=region,
                    cache_dir=self.highways_config.pbf_cache_dir,
                    force_redownload=self.highways_config.force_redownload,
                    max_age_days=self.highways_config.max_cache_age_days
                )
                self.pbf_files.append(pbf_path)
                self.pbf_metadata.append(metadata)
                
                if metadata:
                    age_days = metadata.age_days()
                    self.logger.info(
                        f"  {region}: {metadata.file_size_mb:.1f} MB, "
                        f"{age_days} days old"
                    )
            
            setup_info["pbf_files"] = [str(p) for p in self.pbf_files]
            setup_info["components"].append("pbf_files_ready")
            
            return setup_info
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise ConfigurationError(f"Setup failed: {e}")
    
    def _execute_core_processing(self) -> Dict[str, Any]:
        """
        Execute highway extraction and filtering.
        
        Steps:
        1. Extract highways from PBF file(s)
        2. Spatial filter by AOI
        3. Filter by highway type if specified
        4. Select requested attributes
        5. Clean and validate geometries
        6. Calculate derived attributes
        
        Returns:
            Dictionary with processing statistics
        """
        processing_stats = {}
        all_highways = []
        
        try:
            # Extract from each PBF file
            self.logger.info(f"Extracting highways from {len(self.pbf_files)} PBF file(s)...")
            
            for pbf_path, region in zip(self.pbf_files, self.regions):
                self.logger.info(f"Reading OSM data from {region}...")
                
                # Initialize Pyrosm
                osm = pyrosm.OSM(str(pbf_path))
                
                # Extract all highways
                # network_type="all" gets all highway types
                highways_region = osm.get_network(network_type="all")
                
                if highways_region is None or len(highways_region) == 0:
                    self.logger.warning(f"No highways found in {region}")
                    continue
                
                self.logger.info(
                    f"  Extracted {len(highways_region):,} highway segments from {region}"
                )
                
                # Spatial filter by AOI
                # Ensure both geometries are in the same CRS for spatial operations
                # self.aoi_gdf is already in EPSG:4326 from _setup_custom_processing()
                if highways_region.crs != self.aoi_gdf.crs:
                    self.logger.debug(
                        f"CRS mismatch: highways={highways_region.crs}, AOI={self.aoi_gdf.crs}. "
                        f"Reprojecting AOI to match highways CRS for spatial operations."
                    )
                    aoi_gdf_for_filter = self.aoi_gdf.to_crs(highways_region.crs)
                else:
                    aoi_gdf_for_filter = self.aoi_gdf
                
                # Create unified geometry for intersection test
                aoi_geom = aoi_gdf_for_filter.union_all()
                
                # Perform spatial intersection using GeoSeries method
                highways_in_aoi = highways_region[
                    highways_region.geometry.intersects(aoi_geom)
                ].copy()
                
                self.logger.info(
                    f"  Filtered to AOI: {len(highways_in_aoi):,} segments"
                )
                
                all_highways.append(highways_in_aoi)
            
            if not all_highways:
                raise ExtractionError("No highways found in any region")
            
            # Merge all regions
            self.logger.info("Merging highways from all regions...")
            self.highways_raw = gpd.GeoDataFrame(
                pd.concat(all_highways, ignore_index=True),
                crs=all_highways[0].crs
            )
            processing_stats["highways_extracted"] = len(self.highways_raw)
            
            # Deduplicate (for multi-region overlap)
            if len(self.regions) > 1:
                self.logger.info("Deduplicating highways across region boundaries...")
                self.highways_raw = deduplicate_highways(self.highways_raw)
                processing_stats["highways_after_dedup"] = len(self.highways_raw)
            
            # Filter by highway type
            if self.highways_config.highway_types != "all":
                self.logger.info(
                    f"Filtering by highway types: {self.highways_config.highway_types}"
                )
                self.highways_filtered = filter_highways_by_type(
                    self.highways_raw,
                    self.highways_config.highway_types
                )
            else:
                self.highways_filtered = self.highways_raw.copy()
            
            processing_stats["highways_after_type_filter"] = len(self.highways_filtered)
            
            # Clip to AOI if requested
            if self.highways_config.clip_to_aoi:
                self.logger.info("Clipping highways to AOI boundary...")
                self.highways_filtered = clip_highways_to_aoi(
                    self.highways_filtered,
                    self.aoi_gdf,
                    buffer_meters=self.highways_config.buffer_aoi_meters
                )
                processing_stats["highways_after_clip"] = len(self.highways_filtered)
            
            # Select attributes
            if self.highways_config.include_attributes != "all":
                self.logger.info(
                    f"Selecting attributes: {self.highways_config.include_attributes}"
                )
                self.highways_filtered = select_highway_attributes(
                    self.highways_filtered,
                    self.highways_config.include_attributes
                )
            
            # Clean attributes
            self.logger.info("Cleaning and standardizing attributes...")
            self.highways_filtered = clean_highway_attributes(self.highways_filtered)
            
            # Validate geometries
            self.logger.info("Validating geometries...")
            self.highways_filtered = validate_highway_geometries(self.highways_filtered)
            processing_stats["highways_final"] = len(self.highways_filtered)
            
            # Calculate length
            self.logger.info("Calculating highway lengths...")
            self.highways_filtered = calculate_highway_length(self.highways_filtered)
            
            # Generate summary
            summary = summarize_highway_network(self.highways_filtered)
            processing_stats["summary"] = summary
            
            self.logger.info(
                f"Extraction complete: {processing_stats['highways_final']:,} highways, "
                f"{summary['total_length_km']:.1f} km total"
            )
            
            return processing_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise ExtractionError(f"Highway extraction failed: {e}")
    
    def _cleanup_custom_processing(self) -> None:
        """
        Cleanup after processing.
        
        Currently minimal - Pyrosm handles cleanup internally.
        Could add: temp file cleanup, cache statistics, etc.
        """
        self.logger.info("Processing cleanup complete")
    
    def _export_results(self) -> Path:
        """
        Export highways to requested format.
        
        Supports:
        - GeoJSON (.geojson)
        - Shapefile (.shp)
        - GeoParquet (.parquet)
        - CSV (.csv) - geometry as WKT
        
        Returns:
            Path to output file
        """
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            aoi_name = self.highways_config.aoi_file.stem
            region_str = "_".join(self.regions) if len(self.regions) <= 3 else "multi_region"
            
            base_name = f"highways_{region_str}_{aoi_name}_{timestamp}"
            
            # Reproject if needed
            if self.highways_config.output_crs != "EPSG:4326":
                self.logger.info(f"Reprojecting to {self.highways_config.output_crs}")
                self.highways_filtered = self.highways_filtered.to_crs(
                    self.highways_config.output_crs
                )
            
            # Simplify if requested
            if self.highways_config.simplify_tolerance_meters:
                self.logger.info(
                    f"Simplifying geometries (tolerance: "
                    f"{self.highways_config.simplify_tolerance_meters}m)"
                )
                self.highways_filtered['geometry'] = self.highways_filtered.geometry.simplify(
                    self.highways_config.simplify_tolerance_meters,
                    preserve_topology=True
                )
            
            # Export based on format
            format_map = {
                'geojson': ('.geojson', 'GeoJSON'),
                'shapefile': ('.shp', 'ESRI Shapefile'),
                'geoparquet': ('.parquet', 'Parquet'),
                'csv': ('.csv', 'CSV')
            }
            
            ext, driver = format_map[self.highways_config.export_format]
            self.output_file = self.highways_config.output_dir / f"{base_name}{ext}"
            
            # Check overwrite
            if self.output_file.exists() and not self.highways_config.overwrite_existing:
                raise ExtractionError(
                    f"Output file exists: {self.output_file}. "
                    "Set overwrite_existing=True to overwrite."
                )
            
            self.logger.info(f"Exporting to {self.output_file}...")
            
            if self.highways_config.export_format == 'csv':
                # CSV export with WKT geometry
                df = self.highways_filtered.copy()
                df['geometry'] = df.geometry.to_wkt()
                df.to_csv(self.output_file, index=False)
            else:
                # Geospatial formats
                self.highways_filtered.to_file(
                    self.output_file,
                    driver=driver,
                    index=self.highways_config.create_spatial_index
                )
            
            file_size_mb = self.output_file.stat().st_size / (1024 ** 2)
            self.logger.info(
                f"Export complete: {self.output_file} ({file_size_mb:.2f} MB)"
            )
            
            # Create metadata file
            self._export_metadata()
            
            return self.output_file
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise ExtractionError(f"Failed to export results: {e}")
    
    def _export_metadata(self) -> None:
        """Export processing metadata JSON file."""
        metadata = {
            "processor": "OSMHighwaysProcessor",
            "version": "1.0.0",
            "processing_date": datetime.now().isoformat(),
            "config": {
                "aoi_file": str(self.highways_config.aoi_file),
                "regions": self.regions,
                "highway_types": self.highways_config.highway_types,
                "include_attributes": self.highways_config.include_attributes,
                "export_format": self.highways_config.export_format,
            },
            "data_sources": [
                {
                    "region": region,
                    "download_date": meta.download_date.isoformat() if meta else "unknown",
                    "file_size_mb": meta.file_size_mb if meta else None,
                    "geofabrik_url": meta.geofabrik_url if meta else None
                }
                for region, meta in zip(self.regions, self.pbf_metadata)
            ],
            "results": {
                "output_file": str(self.output_file),
                "feature_count": len(self.highways_filtered),
                "total_length_km": self.highways_filtered['length_m'].sum() / 1000 
                    if 'length_m' in self.highways_filtered.columns else None,
                "highway_type_counts": self.highways_filtered['highway'].value_counts().to_dict()
                    if 'highway' in self.highways_filtered.columns else {}
            }
        }
        
        meta_file = self.output_file.with_suffix('.meta.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved: {meta_file}")