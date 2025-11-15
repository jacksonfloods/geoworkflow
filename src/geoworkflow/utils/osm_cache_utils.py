"""
OSM S2 cache utilities for fast spatial queries.

This module provides S2-based spatial partitioning of OSM PBF files,
inspired by the Google Open Buildings GCS architecture. By pre-partitioning
highway data into S2 cells, we can achieve 10-100x faster extraction for
repeated queries.

Architecture:
    PBF file (entire country) → S2-partitioned GeoParquet files
    Query time: Load only relevant S2 cells instead of entire country

Example:
    # One-time preprocessing
    partition_pbf_to_s2_cache(
        pbf_path=Path("kenya.osm.pbf"),
        cache_dir=Path("cache/osm_s2"),
        s2_level=6
    )

    # Fast querying
    highways = get_highways_from_s2_cache(
        country_iso3="KEN",
        aoi_geometry=nairobi_boundary,
        cache_dir=Path("cache/osm_s2")
    )
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from shapely.prepared import prep
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from geoworkflow.utils.s2_utils import (
    get_bounding_box_s2_covering_tokens,
    s2_token_to_shapely_polygon
)

logger = logging.getLogger(__name__)


# ==================== S2 CACHE PARTITIONING ====================


def partition_pbf_to_s2_cache(
    pbf_path: Path,
    cache_dir: Path,
    country_iso3: str,
    s2_level: int = 6,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Partition PBF file into S2 cell-based GeoParquet cache.

    This is a one-time preprocessing step that converts a country PBF file
    into spatially-partitioned files for fast querying.

    Args:
        pbf_path: Path to PBF file (e.g., kenya.osm.pbf)
        cache_dir: Directory to store S2-partitioned cache
        country_iso3: ISO3 country code (e.g., "KEN")
        s2_level: S2 cell level (default 6, ~1.27° cells)
        overwrite: If True, regenerate cache even if exists

    Returns:
        Dictionary with cache statistics

    Example:
        stats = partition_pbf_to_s2_cache(
            pbf_path=Path("kenya.osm.pbf"),
            cache_dir=Path("cache/osm_s2"),
            country_iso3="KEN",
            s2_level=6
        )
        print(f"Created cache with {stats['num_cells']} S2 cells")
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for S2 cache partitioning")

    # Create country-specific cache directory
    country_cache_dir = cache_dir / country_iso3

    # Check if cache already exists
    metadata_file = country_cache_dir / "metadata.json"
    if metadata_file.exists() and not overwrite:
        logger.info(f"S2 cache already exists for {country_iso3} at {country_cache_dir}")
        with open(metadata_file, 'r') as f:
            return json.load(f)

    logger.info(f"Partitioning {pbf_path.name} into S2 level-{s2_level} cells...")
    country_cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load highways from PBF using pyrosm
    try:
        import pyrosm
    except ImportError:
        raise ImportError("pyrosm is required for PBF partitioning. Install with: pip install pyrosm")

    logger.info("  Loading highways from PBF...")
    osm = pyrosm.OSM(str(pbf_path))
    highways_gdf = osm.get_network(network_type="all")

    if len(highways_gdf) == 0:
        raise ValueError(f"No highways found in {pbf_path}")

    total_highways = len(highways_gdf)
    logger.info(f"  Loaded {total_highways:,} highways")

    # Step 2: Compute S2 covering for entire country
    logger.info("  Computing S2 cell covering...")
    country_bounds = box(*highways_gdf.total_bounds)
    s2_tokens = get_bounding_box_s2_covering_tokens(country_bounds, level=s2_level)
    logger.info(f"  Country covered by {len(s2_tokens)} S2 cells")

    # Step 3: Assign each highway to S2 cells
    logger.info("  Assigning highways to S2 cells...")
    highways_gdf['s2_token'] = highways_gdf.geometry.apply(
        lambda geom: _assign_geometry_to_s2_token(geom, s2_level)
    )

    # Step 4: Partition and save to GeoParquet files
    logger.info("  Writing S2-partitioned GeoParquet files...")
    cells_written = 0
    highways_per_cell = {}

    for s2_token in s2_tokens:
        # Get highways in this cell
        cell_highways = highways_gdf[highways_gdf['s2_token'] == s2_token]

        if len(cell_highways) == 0:
            continue

        # Remove temporary s2_token column
        cell_highways = cell_highways.drop(columns=['s2_token'])

        # Write to GeoParquet with error handling
        cell_file = country_cache_dir / f"{s2_token}.parquet"
        try:
            cell_highways.to_parquet(cell_file, compression='snappy')
            cells_written += 1
            highways_per_cell[s2_token] = len(cell_highways)
        except Exception as e:
            logger.error(f"Failed to write {cell_file.name}: {e}")
            raise RuntimeError(
                f"GeoParquet write failed. This usually means PyArrow is not installed.\n"
                f"Install with: conda install pyarrow>=12.0.0\n"
                f"Original error: {e}"
            ) from e

    # Step 5: Validate that files were actually written
    if cells_written == 0:
        raise RuntimeError(
            f"No cache files written for {country_iso3}. This indicates a problem with:\n"
            f"1. PyArrow installation (required for GeoParquet)\n"
            f"2. S2 token assignment (no highways matched any cells)\n"
            f"3. Disk write permissions\n"
            f"Install PyArrow with: conda install pyarrow>=12.0.0"
        )

    # Step 6: Write metadata
    metadata = {
        'country_iso3': country_iso3,
        'pbf_file': pbf_path.name,
        'pbf_path': str(pbf_path),
        's2_level': s2_level,
        'total_highways': total_highways,
        'num_cells': cells_written,
        'cells': highways_per_cell,
        'created_at': datetime.now().isoformat(),
        'cache_version': '1.0'
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"✓ Created S2 cache: {cells_written} cells, "
        f"{total_highways:,} highways → {country_cache_dir}"
    )

    return metadata


def _assign_geometry_to_s2_token(geometry, s2_level: int) -> str:
    """
    Assign a geometry to its primary S2 cell token.

    Uses the geometry's centroid to determine the S2 cell.
    For LineStrings, this gives a single representative cell.
    """
    from s2sphere import CellId, LatLng

    # Get centroid
    centroid = geometry.centroid

    # Convert to S2 cell
    lat_lng = LatLng.from_degrees(centroid.y, centroid.x)
    cell_id = CellId.from_lat_lng(lat_lng)

    # Get token at specified level
    cell_id_at_level = cell_id.parent(s2_level)
    return cell_id_at_level.to_token()


# ==================== S2 CACHE QUERYING ====================


def get_highways_from_s2_cache(
    country_iso3: str,
    aoi_geometry,
    cache_dir: Path,
    buffer_meters: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Query S2 cache for highways intersecting AOI.

    This is the fast query path that only loads relevant S2 cells
    instead of the entire country PBF.

    Args:
        country_iso3: ISO3 country code (e.g., "KEN")
        aoi_geometry: Shapely geometry or GeoDataFrame for AOI
        cache_dir: Directory containing S2 cache
        buffer_meters: Buffer AOI before querying (optional)

    Returns:
        GeoDataFrame of highways intersecting AOI

    Example:
        highways = get_highways_from_s2_cache(
            country_iso3="KEN",
            aoi_geometry=nairobi_boundary,
            cache_dir=Path("cache/osm_s2")
        )
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for S2 cache queries")

    country_cache_dir = cache_dir / country_iso3

    # Check if cache exists
    metadata_file = country_cache_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"S2 cache not found for {country_iso3} at {country_cache_dir}. "
            f"Run partition_pbf_to_s2_cache() first."
        )

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Querying S2 cache for {country_iso3}...")

    # Extract geometry if GeoDataFrame
    if isinstance(aoi_geometry, gpd.GeoDataFrame):
        aoi_geom = aoi_geometry.union_all()
        aoi_crs = aoi_geometry.crs
    else:
        aoi_geom = aoi_geometry
        aoi_crs = "EPSG:4326"  # Assume WGS84

    # Apply buffer if requested
    if buffer_meters > 0:
        # Simple degree approximation (good enough for S2 cell selection)
        degree_buffer = buffer_meters / 111000  # ~111km per degree at equator
        aoi_geom = aoi_geom.buffer(degree_buffer)

    # Get S2 cells covering AOI
    s2_level = metadata['s2_level']
    s2_tokens = get_bounding_box_s2_covering_tokens(aoi_geom, level=s2_level)
    logger.info(f"  AOI covered by {len(s2_tokens)} S2 cells")

    # Load only relevant S2 cell files
    highways_parts = []
    cells_loaded = 0
    highways_loaded = 0

    for s2_token in s2_tokens:
        cell_file = country_cache_dir / f"{s2_token}.parquet"

        if not cell_file.exists():
            # Cell has no highways, skip
            continue

        # Load cell's highways with error handling
        try:
            cell_highways = gpd.read_parquet(cell_file)
            highways_parts.append(cell_highways)
            cells_loaded += 1
            highways_loaded += len(cell_highways)
        except Exception as e:
            logger.error(f"Failed to read {cell_file.name}: {e}")
            raise RuntimeError(
                f"Failed to read S2 cache file {cell_file.name}. "
                f"The cache may be corrupt. Try deleting {country_cache_dir} and recreating. "
                f"Error: {e}"
            ) from e

    if len(highways_parts) == 0:
        logger.warning(f"No highways found in S2 cache for AOI")
        return gpd.GeoDataFrame(columns=['osm_id', 'highway', 'geometry'], crs="EPSG:4326")

    # Concatenate all cell data - use gpd.GeoDataFrame to preserve geometry
    highways_gdf = gpd.GeoDataFrame(pd.concat(highways_parts, ignore_index=True), crs="EPSG:4326")

    logger.info(
        f"  Loaded {highways_loaded:,} highways from {cells_loaded} S2 cells "
        f"({100 * highways_loaded / metadata['total_highways']:.1f}% of country total)"
    )

    return highways_gdf


def check_s2_cache_exists(country_iso3: str, cache_dir: Path) -> bool:
    """
    Check if S2 cache exists for a country.

    Args:
        country_iso3: ISO3 country code
        cache_dir: Cache directory

    Returns:
        True if cache exists, False otherwise
    """
    metadata_file = cache_dir / country_iso3 / "metadata.json"
    return metadata_file.exists()


def get_s2_cache_metadata(country_iso3: str, cache_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Get metadata for S2 cache.

    Args:
        country_iso3: ISO3 country code
        cache_dir: Cache directory

    Returns:
        Metadata dictionary or None if cache doesn't exist
    """
    metadata_file = cache_dir / country_iso3 / "metadata.json"

    if not metadata_file.exists():
        return None

    with open(metadata_file, 'r') as f:
        return json.load(f)


def list_s2_caches(cache_dir: Path) -> List[str]:
    """
    List all available S2 caches.

    Args:
        cache_dir: Cache directory

    Returns:
        List of ISO3 country codes with caches
    """
    if not cache_dir.exists():
        return []

    caches = []
    for country_dir in cache_dir.iterdir():
        if country_dir.is_dir():
            metadata_file = country_dir / "metadata.json"
            if metadata_file.exists():
                caches.append(country_dir.name)

    return sorted(caches)


def clean_incomplete_caches(cache_dir: Path, dry_run: bool = True) -> Dict[str, Any]:
    """
    Detect and optionally remove incomplete S2 cache directories.

    Incomplete caches are directories that exist but:
    - Have no metadata.json file
    - Have no .parquet files

    Args:
        cache_dir: Cache directory
        dry_run: If True, only report what would be deleted (default)

    Returns:
        Dictionary with cleanup statistics

    Example:
        # Check for incomplete caches
        stats = clean_incomplete_caches(Path("cache/osm_s2"), dry_run=True)
        print(f"Found {len(stats['incomplete'])} incomplete caches")

        # Actually remove them
        stats = clean_incomplete_caches(Path("cache/osm_s2"), dry_run=False)
    """
    import shutil

    if not cache_dir.exists():
        return {'incomplete': [], 'removed': [], 'errors': []}

    incomplete = []
    removed = []
    errors = []

    for country_dir in cache_dir.iterdir():
        if not country_dir.is_dir():
            continue

        country_iso3 = country_dir.name
        metadata_file = country_dir / "metadata.json"

        # Check if cache is incomplete
        is_incomplete = False

        if not metadata_file.exists():
            # No metadata = incomplete
            is_incomplete = True
            reason = "missing metadata.json"
        else:
            # Has metadata, but check if parquet files exist
            parquet_files = list(country_dir.glob("*.parquet"))
            if len(parquet_files) == 0:
                is_incomplete = True
                reason = "no parquet files"
            else:
                # Cache looks complete
                continue

        incomplete.append({'iso3': country_iso3, 'reason': reason, 'path': str(country_dir)})

        if not dry_run:
            try:
                shutil.rmtree(country_dir)
                removed.append(country_iso3)
                logger.info(f"Removed incomplete cache: {country_iso3} ({reason})")
            except Exception as e:
                errors.append({'iso3': country_iso3, 'error': str(e)})
                logger.error(f"Failed to remove {country_iso3}: {e}")

    if dry_run and len(incomplete) > 0:
        logger.warning(
            f"Found {len(incomplete)} incomplete cache directories. "
            f"Run with dry_run=False to remove them."
        )

    return {
        'incomplete': incomplete,
        'removed': removed,
        'errors': errors
    }
