"""
ODIAC fossil fuel CO2 emissions extraction processor.

This processor extracts CO2 emissions data from the ODIAC dataset via
NASA's GHG Center STAC and Raster APIs. It follows the established
TemplateMethodProcessor pattern used in other extraction processors.

Key features:
- Batch processing with AfricaPolis agglomerations or country boundaries
- Monthly and annual average extraction
- Zonal statistics computation
- Cloud-optimized raster export

Performance:
- Small urban area (10 km²): ~5-10 seconds per month
- Medium city (100 km²): ~10-20 seconds per month
- Country-level: ~30-120 seconds per month (varies by size)
- API-based (no large downloads required)

Example (AfricaPolis mode):
    config = ODIACConfig(
        aoi_file="africapolis",
        country=["KEN"],
        city=["Nairobi"],
        year=2022,
        output_dir=Path("./outputs")
    )

    processor = ODIACProcessor(config)
    result = processor.process()
    print(f"Processed {result.succeeded_count} cities")

Example (Countries mode):
    config = ODIACConfig(
        aoi_file="countries",
        country=["KEN", "TZA", "UGA"],
        year=2022,
        output_dir=Path("./outputs")
    )

    processor = ODIACProcessor(config)
    result = processor.process()
    print(f"Processed {result.succeeded_count} countries")
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from datetime import datetime
import json
import calendar

try:
    import requests
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    from pystac_client import Client
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.io import MemoryFile
    from rasterio.mask import mask
    from shapely.geometry import mapping
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
from geoworkflow.schemas.odiac_config import ODIACConfig
from geoworkflow.core.base import ProcessingResult


class ODIACProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Extract CO2 emissions from ODIAC dataset via NASA GHG Center APIs.

    This processor provides efficient extraction of monthly and annual
    CO2 emissions data for arbitrary AOIs using cloud-based APIs.

    Workflow:
    1. Load AOI (single or batch AfricaPolis mode)
    2. Query STAC API for monthly items
    3. For each geometry:
       a. Download clipped monthly TIFFs via Raster API
       b. Compute annual average
       c. Generate zonal statistics
       d. Export results

    The processor handles batch processing automatically with parallel
    execution using thread-based workers.
    """

    def __init__(
        self,
        config: Union[ODIACConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ODIAC processor.

        Args:
            config: Configuration (ODIACConfig or dict)
            logger: Optional logger instance
        """
        # Check dependencies
        if not HAS_REQUIRED_LIBS:
            raise ImportError(
                "Required libraries not available. Install with: "
                "pip install pystac-client requests geopandas rasterio shapely"
            )

        # Convert Pydantic model to dict for base class
        if isinstance(config, ODIACConfig):
            config_dict = config.model_dump(mode='json')
            self.odiac_config = config
        else:
            config_dict = config
            self.odiac_config = ODIACConfig(**config_dict)

        super().__init__(config_dict, logger)

        # Processing state
        self.aoi_crs = None
        self.stac_items: List[Any] = []
        self.monthly_data: Dict[str, Dict] = {}

    def _get_max_workers(self) -> int:
        """
        Determine optimal number of parallel workers for API calls.

        Since we're using threads for I/O-bound API operations,
        we can safely use more workers than CPU cores.

        Returns configured num_workers.
        """
        return self.odiac_config.num_workers

    def process(self):
        """Main entry point for processing."""
        # Query STAC API once for all monthly items
        self._query_stac_items()

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

    def _is_africapolis_mode(self) -> bool:
        """Check if running in AfricaPolis batch mode."""
        return isinstance(self.odiac_config.aoi_file, str) and self.odiac_config.aoi_file == "africapolis"

    def _is_countries_mode(self) -> bool:
        """Check if running in Countries batch mode."""
        return isinstance(self.odiac_config.aoi_file, str) and self.odiac_config.aoi_file == "countries"

    def _query_stac_items(self):
        """Query STAC API for all monthly items for the specified year."""
        self.logger.info(f"Querying STAC API for {self.odiac_config.year} data...")

        try:
            catalog = Client.open(self.odiac_config.stac_api_url)

            search = catalog.search(
                collections=self.odiac_config.collection_name,
                datetime=[
                    f'{self.odiac_config.year}-01-01T00:00:00Z',
                    f'{self.odiac_config.year}-12-31T23:59:59Z'
                ]
            )

            self.stac_items = list(search.item_collection())

            if not self.stac_items:
                raise ExtractionError(
                    f"No ODIAC data found for year {self.odiac_config.year}"
                )

            # Sort by date
            self.stac_items.sort(key=lambda x: x.properties['start_datetime'])

            # Build monthly data dictionary
            for item in self.stac_items:
                month_str = item.properties['start_datetime'][:7]  # YYYY-MM
                s3_path = item.assets[self.odiac_config.asset_name].href

                self.monthly_data[month_str] = {
                    'item_id': item.id,
                    's3_path': s3_path,
                    'item': item
                }

            self.logger.info(f"Found {len(self.stac_items)} monthly items")

        except Exception as e:
            raise ExtractionError(f"STAC API query failed: {str(e)}")

    def _process_single_geometry(
        self,
        geometry,
        name: str,
        iso3: str
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Process a single geometry (agglomeration) - thread-safe.

        This method is designed to be called by multiple threads concurrently.
        Each thread makes independent API calls.

        Args:
            geometry: Geometry to clip to
            name: Name of the agglomeration
            iso3: Country ISO3 code

        Returns:
            Tuple of (name, success, error_message)
        """
        try:
            # Reproject geometry to WGS84 for API
            temp_gdf = gpd.GeoDataFrame(
                {'geometry': [geometry]},
                crs=self.aoi_crs
            )
            if temp_gdf.crs != "EPSG:4326":
                temp_gdf = temp_gdf.to_crs("EPSG:4326")

            geom_wgs84 = temp_gdf.geometry.iloc[0]

            # Apply buffer if specified
            if self.odiac_config.buffer_aoi_meters > 0:
                # Buffer in equal area projection
                temp_albers = temp_gdf.to_crs("ESRI:102022")
                buffered = temp_albers.buffer(self.odiac_config.buffer_aoi_meters)
                temp_buffered = gpd.GeoDataFrame(
                    {'geometry': buffered},
                    crs="ESRI:102022"
                )
                temp_wgs84 = temp_buffered.to_crs("EPSG:4326")
                geom_wgs84 = temp_wgs84.geometry.iloc[0]

            # Extract monthly TIFFs and statistics
            monthly_tiffs = []
            monthly_stats = []

            for month_str, data_info in self.monthly_data.items():
                # Get clipped TIFF data via Raster API
                tiff_data = self._download_clipped_tiff(month_str, geom_wgs84)

                # Compute statistics
                stats = self._compute_statistics(
                    tiff_data,
                    geom_wgs84,
                    name,
                    iso3,
                    month_str
                )

                monthly_tiffs.append((month_str, tiff_data))
                monthly_stats.append(stats)

                # Export monthly TIFF if requested
                if self.odiac_config.export_monthly:
                    self._export_tiff(tiff_data, name, iso3, month_str)

            # Compute and export annual average if requested
            if self.odiac_config.export_annual:
                annual_tiff = self._compute_annual_average(monthly_tiffs)
                self._export_tiff(annual_tiff, name, iso3, "annual")

                # Add annual statistics
                annual_stats = self._compute_statistics(
                    annual_tiff,
                    geom_wgs84,
                    name,
                    iso3,
                    "annual"
                )
                monthly_stats.append(annual_stats)

            # Export statistics if requested
            if self.odiac_config.export_statistics:
                self._export_statistics(monthly_stats, name, iso3)

            return (name, True, None)

        except Exception as e:
            return (name, False, str(e))

    def _download_clipped_tiff(
        self,
        month_str: str,
        geometry
    ) -> Dict[str, Any]:
        """
        Download and clip raster data via NASA GHG Center Raster API.

        The Raster API provides authenticated access to the S3 data and
        supports geometry-based clipping, returning only the requested region.

        Args:
            month_str: Month string in format 'YYYY-MM' (e.g., '2022-01')
            geometry: Shapely geometry (WGS84)

        Returns:
            Dictionary with raster data and metadata
        """
        try:
            # Get the item ID for this month
            if month_str not in self.monthly_data:
                raise ExtractionError(f"No STAC item found for month {month_str}")

            item_id = self.monthly_data[month_str]['item_id']

            # Simplify geometry if too complex (Raster API has geometry complexity limits)
            # Count total coordinates in geometry (handles both Polygon and MultiPolygon)
            def count_coords(geom):
                if hasattr(geom, 'exterior'):
                    return len(geom.exterior.coords)
                elif hasattr(geom, 'geoms'):  # MultiPolygon
                    return sum(len(g.exterior.coords) for g in geom.geoms)
                return 0

            coord_count = count_coords(geometry)

            # Simplify if geometry is too complex (>300 coords)
            # Raster API has strict geometry complexity limits (~300 coords max)
            if coord_count > 300:
                original_count = coord_count
                # Start with higher tolerance for very complex geometries
                # For 1km resolution ODIAC data, tolerances up to ~30km are acceptable
                if coord_count > 50000:
                    tolerance = 0.1  # Start at ~10km for very complex geometries
                else:
                    tolerance = 0.01  # Start at ~1km

                geometry = geometry.simplify(tolerance, preserve_topology=True)
                new_count = count_coords(geometry)

                # Iteratively increase tolerance until we're under 300 coords
                while new_count > 300 and tolerance < 0.5:
                    tolerance *= 1.5
                    geometry = geometry.simplify(tolerance, preserve_topology=True)
                    new_count = count_coords(geometry)

                # If still too complex, use convex hull as last resort
                if new_count > 300:
                    self.logger.warning(
                        f"Geometry still has {new_count} coordinates after max simplification. "
                        f"Using convex hull for API compatibility."
                    )
                    geometry = geometry.convex_hull
                    new_count = count_coords(geometry)

                self.logger.info(
                    f"Simplified geometry from {original_count} to {new_count} coordinates "
                    f"(tolerance: {tolerance:.3f}°) for API compatibility"
                )

            # Convert geometry to GeoJSON Feature (required format for API)
            feature = {
                'type': 'Feature',
                'geometry': mapping(geometry),
                'properties': {}
            }

            # Build API request (item-level endpoint)
            url = (f'{self.odiac_config.raster_api_url}/collections/'
                   f'{self.odiac_config.collection_name}/items/{item_id}/feature.tif')
            params = {
                'assets': self.odiac_config.asset_name
            }

            # Make request to Raster API
            response = requests.post(
                url,
                json=feature,
                params=params,
                timeout=self.odiac_config.api_timeout
            )

            if response.status_code != 200:
                raise ExtractionError(
                    f"Raster API request failed with status {response.status_code}: {response.text[:200]}"
                )

            # Check if response is actually a GeoTIFF (not HTML error page)
            content_type = response.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                raise ExtractionError(
                    f"Raster API returned HTML instead of GeoTIFF. "
                    f"Content-Type: {content_type}, Response: {response.text[:300]}"
                )

            # Validate TIFF magic bytes
            if len(response.content) < 4 or response.content[:2] not in [b'II', b'MM']:
                raise ExtractionError(
                    f"Raster API response is not a valid TIFF. "
                    f"Content-Type: {content_type}, "
                    f"Size: {len(response.content)} bytes, "
                    f"First bytes: {response.content[:50]}"
                )

            # Read the returned GeoTIFF from memory
            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    # Read the first band
                    data = src.read(1)

                    return {
                        'data': data,
                        'transform': src.transform,
                        'crs': str(src.crs),
                        'nodata': src.nodata,
                        'width': src.width,
                        'height': src.height
                    }

        except requests.RequestException as e:
            raise ExtractionError(f"Failed to request data from Raster API: {str(e)}")
        except Exception as e:
            raise ExtractionError(f"Failed to process raster data: {str(e)}")

    def _compute_statistics(
        self,
        tiff_data: Dict[str, Any],
        geometry,
        name: str,
        iso3: str,
        period: str
    ) -> Dict[str, Any]:
        """
        Compute zonal statistics for the raster data.

        Args:
            tiff_data: Dictionary with raster data
            geometry: Shapely geometry (WGS84)
            name: City name
            iso3: Country code
            period: Month string (YYYY-MM) or "annual"

        Returns:
            Dictionary with statistics
        """
        data = tiff_data['data']
        nodata = tiff_data.get('nodata')

        # Mask nodata values
        if nodata is not None:
            valid_data = data[data != nodata]
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return {
                'country': name,
                'city': name,
                'iso3': iso3,
                'year': self.odiac_config.year,
                'period': period,
                'mean': np.nan,
                'min': np.nan,
                'max': np.nan,
                'std': np.nan,
                'sum': np.nan,
                'count': 0
            }

        return {
            'country': name,
            'city': name,
            'iso3': iso3,
            'year': self.odiac_config.year,
            'period': period,
            'mean': float(np.mean(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'std': float(np.std(valid_data)),
            'sum': float(np.sum(valid_data)),
            'count': int(len(valid_data))
        }

    def _compute_annual_average(
        self,
        monthly_tiffs: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compute annual average from monthly TIFFs.

        Args:
            monthly_tiffs: List of (month_str, tiff_data) tuples

        Returns:
            Dictionary with averaged raster data
        """
        # Stack all monthly data
        arrays = [tiff['data'] for _, tiff in monthly_tiffs]
        stacked = np.stack(arrays, axis=0)

        # Compute mean along time axis, handling nodata
        nodata = monthly_tiffs[0][1].get('nodata')
        if nodata is not None:
            masked = np.ma.masked_equal(stacked, nodata)
            annual_mean = masked.mean(axis=0).filled(nodata)
        else:
            annual_mean = np.nanmean(stacked, axis=0)

        # Return using first month's metadata
        first_tiff = monthly_tiffs[0][1]
        return {
            'data': annual_mean,
            'transform': first_tiff['transform'],
            'crs': first_tiff['crs'],
            'nodata': first_tiff['nodata'],
            'width': first_tiff['width'],
            'height': first_tiff['height']
        }

    def _export_tiff(
        self,
        tiff_data: Dict[str, Any],
        name: str,
        iso3: str,
        period: str
    ):
        """
        Export TIFF to file, reprojecting to target CRS.

        Args:
            tiff_data: Dictionary with raster data
            name: City name
            iso3: Country code
            period: Month string (YYYY-MM) or "annual"
        """
        safe_name = name.replace(" ", "_").replace("/", "_")
        period_str = period.replace("-", "_")

        output_file = (
            self.odiac_config.output_dir /
            f"{iso3}_{safe_name}_odiac_{self.odiac_config.year}_{period_str}.tif"
        )

        # Check if file exists
        if output_file.exists() and not self.odiac_config.overwrite_existing:
            self.logger.debug(f"Skipping existing file: {output_file.name}")
            return

        # Reproject to target CRS if needed
        src_crs = tiff_data['crs']
        dst_crs = self.odiac_config.output_crs

        if src_crs != dst_crs:
            # Calculate transform for target CRS
            transform, width, height = calculate_default_transform(
                src_crs,
                dst_crs,
                tiff_data['width'],
                tiff_data['height'],
                left=tiff_data['transform'].c,
                bottom=tiff_data['transform'].f + tiff_data['transform'].e * tiff_data['height'],
                right=tiff_data['transform'].c + tiff_data['transform'].a * tiff_data['width'],
                top=tiff_data['transform'].f
            )

            # Create destination array
            dst_data = np.empty((height, width), dtype=tiff_data['data'].dtype)

            # Reproject
            reproject(
                source=tiff_data['data'],
                destination=dst_data,
                src_transform=tiff_data['transform'],
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=tiff_data['nodata'],
                dst_nodata=tiff_data['nodata']
            )

            # Update for writing
            write_data = dst_data
            write_transform = transform
            write_width = width
            write_height = height
        else:
            write_data = tiff_data['data']
            write_transform = tiff_data['transform']
            write_width = tiff_data['width']
            write_height = tiff_data['height']

        # Write to file
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=write_height,
            width=write_width,
            count=1,
            dtype=write_data.dtype,
            crs=dst_crs,
            transform=write_transform,
            nodata=tiff_data['nodata'],
            compress=self.odiac_config.compression
        ) as dst:
            dst.write(write_data, 1)

    def _export_statistics(
        self,
        stats_list: List[Dict[str, Any]],
        name: str,
        iso3: str
    ):
        """
        Export statistics to CSV.

        Args:
            stats_list: List of statistics dictionaries
            name: City name
            iso3: Country code
        """
        safe_name = name.replace(" ", "_").replace("/", "_")
        output_file = (
            self.odiac_config.output_dir /
            f"{iso3}_{safe_name}_odiac_{self.odiac_config.year}_stats.csv"
        )

        # Check if file exists
        if output_file.exists() and not self.odiac_config.overwrite_existing:
            self.logger.debug(f"Skipping existing file: {output_file.name}")
            return

        df = pd.DataFrame(stats_list)
        df.to_csv(output_file, index=False)

    def _process_geometries(self, geometries: List[Tuple]) -> Union[BatchProcessResult, ProcessingResult]:
        """Process all geometries with thread-based parallelization."""
        succeeded = []
        failed = {}

        max_workers = self._get_max_workers()

        self.logger.info(f"Processing {len(geometries)} geometries with {max_workers} parallel threads")

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
                name, success, error_msg = future.result()

                if success:
                    succeeded.append(name)
                    self.logger.info(f"  ✓ {name}")
                else:
                    failed[name] = error_msg
                    self.logger.error(f"  ✗ {name}: {error_msg}")

        # Return appropriate result type
        if self._is_africapolis_mode() or self._is_countries_mode():
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

    def _filter_agglomerations(
        self,
        gdf: gpd.GeoDataFrame,
        columns: Dict[str, str]
    ) -> gpd.GeoDataFrame:
        """Filter by country and city using AND logic."""
        filtered = gdf.copy()

        if isinstance(self.odiac_config.country, str) and self.odiac_config.country.lower() == "all":
            pass
        elif isinstance(self.odiac_config.country, list):
            filtered = filtered[filtered[columns["iso3"]].isin(self.odiac_config.country)]

        if self.odiac_config.city:
            filtered = filtered[filtered[columns["name"]].isin(self.odiac_config.city)]

        if len(filtered) == 0:
            raise ValueError("No agglomerations match the specified filters")

        self.logger.info(
            f"Filters: country={self.odiac_config.country}, "
            f"city={self.odiac_config.city}"
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

        if isinstance(self.odiac_config.country, str) and self.odiac_config.country.lower() == "all":
            pass
        elif isinstance(self.odiac_config.country, list):
            # Try filtering by ISO3 column first
            if columns["iso3"] in filtered.columns:
                filtered = filtered[filtered[columns["iso3"]].isin(self.odiac_config.country)]
            else:
                self.logger.warning(f"ISO3 column '{columns['iso3']}' not found, cannot filter by country")

        if len(filtered) == 0:
            raise ValueError("No countries match the specified filters")

        self.logger.info(f"Selected {len(filtered)} countries")

        # Return list of (geometry, name, iso3)
        # Use country name as the name, and ISO3 code
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
        aoi_gdf = gpd.read_file(self.odiac_config.aoi_file)
        self.aoi_crs = aoi_gdf.crs

        # Use filename as name, generate placeholder ISO3
        name = self.odiac_config.aoi_file.stem
        iso3 = "AOI"  # Placeholder for single AOI mode

        # Return list with single item: (geometry, name, iso3)
        return [(aoi_gdf.union_all(), name, iso3)]
