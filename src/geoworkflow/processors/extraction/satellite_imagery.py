"""
Satellite imagery extraction processor for Google Earth Engine.

This processor extracts optical RGB satellite imagery from Sentinel-2 MSI
Level-2A via Google Earth Engine. It follows the established
TemplateMethodProcessor pattern used in other extraction processors.

Key features:
- Sentinel-2 MSI Level-2A (10m resolution RGB)
- Cloud masking using Scene Classification Layer (SCL)
- Median temporal compositing for cloud-free imagery
- Batch processing with AfricaPolis agglomerations or country boundaries
- GeoTIFF export with tiling for large AOIs

Example (Single AOI):
    config = SatelliteImageryConfig(
        aoi_file=Path("nairobi.geojson"),
        output_dir=Path("./outputs"),
        start_date="2024-01-01",
        end_date="2024-06-30"
    )

    processor = SatelliteImageryProcessor(config)
    result = processor.process()
    print(f"Output: {result.output_paths}")

Example (AfricaPolis mode):
    config = SatelliteImageryConfig(
        aoi_file="africapolis",
        country=["KEN"],
        city=["Nairobi"],
        output_dir=Path("./outputs"),
        start_date="2024-01-01",
        end_date="2024-06-30"
    )

    processor = SatelliteImageryProcessor(config)
    result = processor.process()
    print(f"Processed {result.succeeded_count} cities")
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import tempfile

try:
    import ee
    import geopandas as gpd
    import numpy as np
    import requests
    import rasterio
    from rasterio.merge import merge
    from rasterio.io import MemoryFile
    from shapely.geometry import mapping, box
    HAS_REQUIRED_LIBS = True
except ImportError:
    HAS_REQUIRED_LIBS = False

from geoworkflow.schemas.processing_result import BatchProcessResult
from geoworkflow.utils.config_loader import ConfigLoader
from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.exceptions import ExtractionError
from geoworkflow.schemas.satellite_imagery_config import SatelliteImageryConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.earth_engine_utils import EarthEngineAuth, check_earth_engine_available


# Sentinel-2 collection and band mappings
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = {
    "red": "B4",
    "green": "B3",
    "blue": "B2",
    "nir": "B8",
    "scl": "SCL"  # Scene Classification Layer for cloud masking
}

# SCL values to mask (clouds, shadows, etc.)
# 3=Cloud shadow, 8=Cloud medium prob, 9=Cloud high prob, 10=Thin cirrus
SCL_MASK_VALUES = [3, 8, 9, 10]


class SatelliteImageryProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Extract satellite imagery from Google Earth Engine.

    This processor provides efficient extraction of Sentinel-2 RGB imagery
    for arbitrary AOIs using cloud-based processing via Earth Engine.

    Workflow:
    1. Authenticate with Earth Engine
    2. Load AOI (single or batch AfricaPolis/Countries mode)
    3. For each geometry:
       a. Query Sentinel-2 collection for date range
       b. Apply scene-level cloud filtering
       c. Apply per-pixel cloud masking using SCL
       d. Create median composite
       e. Export as GeoTIFF (with tiling for large areas)

    The processor handles batch processing automatically with parallel
    execution using thread-based workers.
    """

    def __init__(
        self,
        config: Union[SatelliteImageryConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize satellite imagery processor.

        Args:
            config: Configuration (SatelliteImageryConfig or dict)
            logger: Optional logger instance
        """
        # Check dependencies
        if not HAS_REQUIRED_LIBS:
            raise ImportError(
                "Required libraries not available. Install with: "
                "pip install earthengine-api geopandas rasterio requests shapely"
            )

        # Convert Pydantic model to dict for base class
        if isinstance(config, SatelliteImageryConfig):
            config_dict = config.model_dump(mode='json')
            self.imagery_config = config
        else:
            config_dict = config
            self.imagery_config = SatelliteImageryConfig(**config_dict)

        super().__init__(config_dict, logger)

        # Processing state
        self.project_id: Optional[str] = None
        self.aoi_crs = None
        self.collection: Optional[ee.ImageCollection] = None

    def _get_max_workers(self) -> int:
        """Return configured number of parallel workers."""
        return self.imagery_config.num_workers

    def process(self) -> Union[BatchProcessResult, ProcessingResult]:
        """Main entry point for processing."""
        # Setup Earth Engine
        self._setup_earth_engine()

        # Load geometries (batch or single)
        if self._is_africapolis_mode():
            geometries = self._load_batch_geometries()
        elif self._is_countries_mode():
            geometries = self._load_countries_geometries()
        else:
            geometries = self._load_single_geometry()

        # Process all geometries
        return self._process_geometries(geometries)

    def process_data(self) -> Dict[str, Any]:
        """Not used - call process() directly."""
        raise NotImplementedError("Use process() method directly")

    def _setup_earth_engine(self):
        """Initialize Earth Engine authentication and collection."""
        if not check_earth_engine_available():
            raise ExtractionError(
                "Earth Engine API not available. Install with: pip install earthengine-api"
            )

        try:
            self.project_id = EarthEngineAuth.authenticate(
                service_account_key=self.imagery_config.service_account_key,
                service_account_email=self.imagery_config.service_account_email,
                project_id=self.imagery_config.project_id
            )

            # Initialize Sentinel-2 collection
            self.collection = ee.ImageCollection(S2_COLLECTION)

            self.logger.info(
                f"Earth Engine initialized with project: {self.project_id}"
            )

        except Exception as e:
            raise ExtractionError(f"Failed to setup Earth Engine: {e}")

    def _is_africapolis_mode(self) -> bool:
        """Check if running in AfricaPolis batch mode."""
        return (isinstance(self.imagery_config.aoi_file, str) and
                self.imagery_config.aoi_file == "africapolis")

    def _is_countries_mode(self) -> bool:
        """Check if running in Countries batch mode."""
        return (isinstance(self.imagery_config.aoi_file, str) and
                self.imagery_config.aoi_file == "countries")

    def _mask_clouds(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Sentinel-2 image using SCL band.

        The Scene Classification Layer (SCL) provides per-pixel classification:
        - 3: Cloud shadow
        - 8: Cloud medium probability
        - 9: Cloud high probability
        - 10: Thin cirrus

        Args:
            image: Sentinel-2 ee.Image with SCL band

        Returns:
            Cloud-masked ee.Image
        """
        scl = image.select('SCL')

        # Create mask for valid pixels (not in mask values)
        mask = scl.neq(3)  # Not cloud shadow
        for val in SCL_MASK_VALUES[1:]:
            mask = mask.And(scl.neq(val))

        return image.updateMask(mask)

    def _create_composite(self, geometry: ee.Geometry) -> ee.Image:
        """
        Create a cloud-free median composite for the given geometry.

        Args:
            geometry: Earth Engine geometry for the AOI

        Returns:
            Median composite ee.Image with RGB bands

        Raises:
            ExtractionError: If no valid imagery found for the date range
        """
        # Filter collection by bounds and date
        filtered = (self.collection
            .filterBounds(geometry)
            .filterDate(
                self.imagery_config.start_date,
                self.imagery_config.end_date
            ))

        # Apply scene-level cloud filtering
        max_cloud = self.imagery_config.max_cloud_probability
        filtered = filtered.filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud)
        )

        # Check if we have any images after filtering
        image_count = filtered.size().getInfo()
        if image_count == 0:
            raise ExtractionError(
                f"No cloud-free imagery found for date range "
                f"{self.imagery_config.start_date} to {self.imagery_config.end_date} "
                f"with max_cloud_probability={max_cloud}%. "
                f"Try: extending date range, or increasing max_cloud_probability."
            )

        self.logger.info(f"Found {image_count} images for composite")

        # Apply per-pixel cloud masking if enabled
        if self.imagery_config.apply_cloud_mask:
            filtered = filtered.map(self._mask_clouds)

        # Select RGB bands
        rgb_bands = [S2_BANDS["red"], S2_BANDS["green"], S2_BANDS["blue"]]
        filtered = filtered.select(rgb_bands)

        # Create median composite
        composite = filtered.median()

        # Clip to geometry if requested
        if self.imagery_config.clip_to_aoi:
            composite = composite.clip(geometry)

        return composite

    def _scale_to_uint8(self, image: ee.Image) -> ee.Image:
        """
        Scale Sentinel-2 reflectance values to 0-255 for RGB visualization.

        Sentinel-2 L2A surface reflectance values are typically 0-10000,
        with most land surfaces in the 0-3000 range for RGB.

        Args:
            image: Sentinel-2 ee.Image

        Returns:
            Scaled ee.Image with uint8 values (0-255)
        """
        # Scale factor: 0.0001 to convert to reflectance, then stretch 0-0.3 to 0-255
        return (image
            .multiply(0.0001)
            .clamp(0, 0.3)
            .multiply(255 / 0.3)
            .uint8())

    def _export_image_to_geotiff(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        name: str,
        iso3: str
    ) -> Optional[Path]:
        """
        Export Earth Engine image to local GeoTIFF.

        Uses getDownloadURL for smaller areas, tiling for larger areas.

        Args:
            image: Earth Engine image to export
            geometry: Geometry for export bounds
            name: Location name for filename
            iso3: ISO3 country code

        Returns:
            Path to output file, or None if skipped
        """
        # Build output filename
        safe_name = name.replace(" ", "_").replace("/", "_")
        start_date = self.imagery_config.start_date.replace("-", "")
        end_date = self.imagery_config.end_date.replace("-", "")

        filename = f"{iso3}_{safe_name}_S2_{start_date}_{end_date}.tif"
        output_path = self.imagery_config.output_dir / filename

        # Check if exists
        if output_path.exists() and not self.imagery_config.overwrite_existing:
            self.logger.debug(f"Skipping existing file: {output_path.name}")
            return output_path

        # Scale if needed
        if self.imagery_config.scale_to_uint8:
            image = self._scale_to_uint8(image)

        # Get geometry bounds for export
        bounds_info = geometry.bounds().getInfo()
        coords = bounds_info['coordinates'][0]
        west = coords[0][0]
        south = coords[0][1]
        east = coords[2][0]
        north = coords[2][1]

        # Calculate approximate area in meters (rough estimate)
        # Using ~111km per degree at equator
        width_m = abs(east - west) * 111000
        height_m = abs(north - south) * 111000
        area_m2 = width_m * height_m

        tile_threshold = self.imagery_config.tile_size_m ** 2

        if area_m2 < tile_threshold:
            # Direct download for smaller areas
            self._download_direct(image, geometry, output_path)
        else:
            # Tiled download for larger areas
            self._download_tiled(image, geometry, output_path, west, south, east, north)

        return output_path

    def _download_direct(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        output_path: Path
    ):
        """
        Download image directly using getDownloadURL.

        Args:
            image: Earth Engine image
            geometry: Export geometry
            output_path: Output file path
        """
        url = image.getDownloadURL({
            'scale': self.imagery_config.resolution_m,
            'crs': self.imagery_config.output_crs,
            'region': geometry,
            'format': 'GEO_TIFF',
            'filePerBand': False
        })

        response = requests.get(url, timeout=300)
        response.raise_for_status()

        # Write to file
        with open(output_path, 'wb') as f:
            f.write(response.content)

        self.logger.info(f"Downloaded: {output_path.name}")

    def _download_tiled(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        output_path: Path,
        west: float,
        south: float,
        east: float,
        north: float
    ):
        """
        Download large image using tiled approach with merging.

        Args:
            image: Earth Engine image
            geometry: Export geometry
            output_path: Final output file path
            west, south, east, north: Bounding box coordinates
        """
        tile_size_deg = self.imagery_config.tile_size_m / 111000  # Rough conversion

        # Create tile grid
        tiles = []
        x = west
        while x < east:
            y = south
            while y < north:
                tile_east = min(x + tile_size_deg, east)
                tile_north = min(y + tile_size_deg, north)
                tiles.append((x, y, tile_east, tile_north))
                y += tile_size_deg
            x += tile_size_deg

        self.logger.info(f"Downloading {len(tiles)} tiles for large AOI...")

        # Download tiles to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_paths = []

            for i, (tx, ty, tx2, ty2) in enumerate(tiles):
                tile_region = ee.Geometry.Rectangle([tx, ty, tx2, ty2])
                tile_path = temp_path / f"tile_{i:04d}.tif"

                try:
                    self._download_direct(image, tile_region, tile_path)
                    tile_paths.append(tile_path)
                except Exception as e:
                    self.logger.warning(f"Tile {i} failed: {e}")

            # Merge tiles
            if tile_paths:
                self._merge_tiles(tile_paths, output_path)
            else:
                raise ExtractionError("No tiles downloaded successfully")

    def _merge_tiles(self, tile_paths: List[Path], output_path: Path):
        """
        Merge downloaded tiles into single GeoTIFF.

        Args:
            tile_paths: List of tile file paths
            output_path: Output merged file path
        """
        datasets = [rasterio.open(p) for p in tile_paths]

        try:
            merged, transform = merge(datasets)

            # Determine compression
            compress = self.imagery_config.compression
            if compress == "none":
                compress = None

            # Write merged output
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=merged.shape[1],
                width=merged.shape[2],
                count=merged.shape[0],
                dtype=merged.dtype,
                crs=self.imagery_config.output_crs,
                transform=transform,
                compress=compress
            ) as dst:
                dst.write(merged)

            self.logger.info(f"Merged {len(tile_paths)} tiles: {output_path.name}")

        finally:
            for ds in datasets:
                ds.close()

    def _process_single_geometry(
        self,
        geometry,
        name: str,
        iso3: str
    ) -> Tuple[str, bool, Optional[Union[Path, str]]]:
        """
        Process a single geometry - thread-safe.

        This method is designed to be called by multiple threads concurrently.

        Args:
            geometry: Shapely geometry
            name: Location name
            iso3: Country ISO3 code

        Returns:
            Tuple of (name, success, result_or_error)
        """
        try:
            # Convert to WGS84 for Earth Engine
            temp_gdf = gpd.GeoDataFrame(
                {'geometry': [geometry]},
                crs=self.aoi_crs
            )
            if temp_gdf.crs != "EPSG:4326":
                temp_gdf = temp_gdf.to_crs("EPSG:4326")

            geom_wgs84 = temp_gdf.geometry.iloc[0]

            # Apply buffer if specified
            if self.imagery_config.buffer_aoi_m > 0:
                temp_albers = temp_gdf.to_crs("ESRI:102022")
                buffered = temp_albers.buffer(self.imagery_config.buffer_aoi_m)
                temp_buffered = gpd.GeoDataFrame(
                    {'geometry': buffered},
                    crs="ESRI:102022"
                )
                temp_wgs84 = temp_buffered.to_crs("EPSG:4326")
                geom_wgs84 = temp_wgs84.geometry.iloc[0]

            # Convert to Earth Engine geometry
            ee_geometry = ee.Geometry(geom_wgs84.__geo_interface__)

            # Create composite
            composite = self._create_composite(ee_geometry)

            # Export to GeoTIFF
            output_path = self._export_image_to_geotiff(
                composite, ee_geometry, name, iso3
            )

            return (name, True, output_path)

        except Exception as e:
            return (name, False, str(e))

    def _process_geometries(
        self,
        geometries: List[Tuple]
    ) -> Union[BatchProcessResult, ProcessingResult]:
        """Process all geometries with thread-based parallelization."""
        succeeded = []
        failed = {}
        output_files = []

        max_workers = self._get_max_workers()

        self.logger.info(
            f"Processing {len(geometries)} geometries with {max_workers} parallel threads"
        )

        # Process all geometries in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(
                    self._process_single_geometry,
                    geom, name, iso3
                ): name
                for geom, name, iso3 in geometries
            }

            # Collect results as they complete
            for future in as_completed(future_to_name):
                name, success, result = future.result()

                if success:
                    succeeded.append(name)
                    if result:
                        output_files.append(result)
                    self.logger.info(f"  + {name}")
                else:
                    failed[name] = result
                    self.logger.error(f"  x {name}: {result}")

        # Return appropriate result type
        if self._is_africapolis_mode() or self._is_countries_mode():
            return BatchProcessResult(
                success=len(succeeded) > 0,
                total_count=len(geometries),
                succeeded_count=len(succeeded),
                failed_count=len(failed),
                succeeded=succeeded,
                failed=failed,
                output_files=output_files
            )
        else:
            return ProcessingResult(
                success=len(succeeded) > 0,
                processed_count=len(succeeded),
                output_paths=output_files,
                message=f"Processed {len(succeeded)} geometries"
            )

    def _filter_agglomerations(
        self,
        gdf: gpd.GeoDataFrame,
        columns: Dict[str, str]
    ) -> gpd.GeoDataFrame:
        """Filter by country and city using AND logic."""
        filtered = gdf.copy()

        if isinstance(self.imagery_config.country, str) and \
           self.imagery_config.country.lower() == "all":
            pass
        elif isinstance(self.imagery_config.country, list):
            filtered = filtered[filtered[columns["iso3"]].isin(self.imagery_config.country)]

        if self.imagery_config.city:
            filtered = filtered[filtered[columns["name"]].isin(self.imagery_config.city)]

        if len(filtered) == 0:
            raise ValueError("No agglomerations match the specified filters")

        self.logger.info(
            f"Filters: country={self.imagery_config.country}, "
            f"city={self.imagery_config.city}"
        )
        return filtered

    def _load_batch_geometries(self) -> List[Tuple]:
        """Load agglomerations from AfricaPolis."""
        agglo_path = ConfigLoader.get_africapolis_path()
        columns = ConfigLoader.get_africapolis_columns()

        if not agglo_path.exists():
            raise FileNotFoundError(f"AfricaPolis file not found: {agglo_path}")

        self.logger.info(f"Loading AfricaPolis agglomerations from: {agglo_path}")
        agglomerations = gpd.read_file(agglo_path)
        self.aoi_crs = agglomerations.crs

        filtered = self._filter_agglomerations(agglomerations, columns)

        self.logger.info(f"Selected {len(filtered)} agglomerations")

        # Return list of (geometry, name, iso3)
        return [
            (row.geometry, row[columns["name"]], row[columns["iso3"]])
            for _, row in filtered.iterrows()
        ]

    def _load_countries_geometries(self) -> List[Tuple]:
        """Load country boundaries from africa_boundaries.gpkg."""
        boundaries_path = ConfigLoader.get_africa_boundaries_path()
        columns = ConfigLoader.get_africa_boundaries_columns()

        if not boundaries_path.exists():
            raise FileNotFoundError(f"Africa boundaries file not found: {boundaries_path}")

        self.logger.info(f"Loading Africa country boundaries from: {boundaries_path}")
        boundaries = gpd.read_file(boundaries_path)
        self.aoi_crs = boundaries.crs

        # Filter by country (no city filter for countries mode)
        filtered = boundaries.copy()

        if isinstance(self.imagery_config.country, str) and \
           self.imagery_config.country.lower() == "all":
            pass
        elif isinstance(self.imagery_config.country, list):
            # Try filtering by ISO3 column first
            if columns["iso3"] in filtered.columns:
                filtered = filtered[filtered[columns["iso3"]].isin(self.imagery_config.country)]
            else:
                self.logger.warning(
                    f"ISO3 column '{columns['iso3']}' not found, cannot filter by country"
                )

        if len(filtered) == 0:
            raise ValueError("No countries match the specified filters")

        self.logger.info(f"Selected {len(filtered)} countries")

        # Return list of (geometry, name, iso3)
        return [
            (
                row.geometry,
                row.get(columns["iso3"], row.get(columns["name"], f"Country_{idx}")),
                row.get(columns["iso3"], "UNK")
            )
            for idx, row in filtered.iterrows()
        ]

    def _load_single_geometry(self) -> List[Tuple]:
        """Load single AOI geometry."""
        aoi_gdf = gpd.read_file(self.imagery_config.aoi_file)
        self.aoi_crs = aoi_gdf.crs

        # Use filename as name, generate placeholder ISO3
        name = self.imagery_config.aoi_file.stem
        iso3 = "AOI"  # Placeholder for single AOI mode

        # Return list with single item: (geometry, name, iso3)
        return [(aoi_gdf.union_all(), name, iso3)]
