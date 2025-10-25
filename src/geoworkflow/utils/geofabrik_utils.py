"""
Utilities for downloading and managing Geofabrik OSM PBF files.


This module handles:
- Region name mapping to Geofabrik URLs
- PBF file downloading with progress bars
- Cache management with metadata
- Auto-detection of regions from AOI geometry
"""


from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import requests
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box




logger = logging.getLogger(__name__)




# ==================== GEOFABRIK REGION MAPPING ====================


# Map region names to Geofabrik download paths
# Source: https://download.geofabrik.de/
GEOFABRIK_REGIONS = {
    # Africa
    'algeria': 'africa/algeria-latest.osm.pbf',
    'angola': 'africa/angola-latest.osm.pbf',
    'benin': 'africa/benin-latest.osm.pbf',
    'botswana': 'africa/botswana-latest.osm.pbf',
    'burkina-faso': 'africa/burkina-faso-latest.osm.pbf',
    'burundi': 'africa/burundi-latest.osm.pbf',
    'cameroon': 'africa/cameroon-latest.osm.pbf',
    'cape-verde': 'africa/cape-verde-latest.osm.pbf',
    'central-african-republic': 'africa/central-african-republic-latest.osm.pbf',
    'chad': 'africa/chad-latest.osm.pbf',
    'comoros': 'africa/comoros-latest.osm.pbf',
    'congo-brazzaville': 'africa/congo-brazzaville-latest.osm.pbf',
    'congo-democratic-republic': 'africa/congo-democratic-republic-latest.osm.pbf',
    'djibouti': 'africa/djibouti-latest.osm.pbf',
    'egypt': 'africa/egypt-latest.osm.pbf',
    'equatorial-guinea': 'africa/equatorial-guinea-latest.osm.pbf',
    'eritrea': 'africa/eritrea-latest.osm.pbf',
    'ethiopia': 'africa/ethiopia-latest.osm.pbf',
    'gabon': 'africa/gabon-latest.osm.pbf',
    'ghana': 'africa/ghana-latest.osm.pbf',
    'guinea': 'africa/guinea-latest.osm.pbf',
    'guinea-bissau': 'africa/guinea-bissau-latest.osm.pbf',
    'ivory-coast': 'africa/ivory-coast-latest.osm.pbf',
    'kenya': 'africa/kenya-latest.osm.pbf',
    'lesotho': 'africa/lesotho-latest.osm.pbf',
    'liberia': 'africa/liberia-latest.osm.pbf',
    'libya': 'africa/libya-latest.osm.pbf',
    'madagascar': 'africa/madagascar-latest.osm.pbf',
    'malawi': 'africa/malawi-latest.osm.pbf',
    'mali': 'africa/mali-latest.osm.pbf',
    'mauritania': 'africa/mauritania-latest.osm.pbf',
    'mauritius': 'africa/mauritius-latest.osm.pbf',
    'morocco': 'africa/morocco-latest.osm.pbf',
    'mozambique': 'africa/mozambique-latest.osm.pbf',
    'namibia': 'africa/namibia-latest.osm.pbf',
    'niger': 'africa/niger-latest.osm.pbf',
    'nigeria': 'africa/nigeria-latest.osm.pbf',
    'rwanda': 'africa/rwanda-latest.osm.pbf',
    'saint-helena-ascension-and-tristan-da-cunha': 'africa/saint-helena-ascension-and-tristan-da-cunha-latest.osm.pbf',
    'sao-tome-and-principe': 'africa/sao-tome-and-principe-latest.osm.pbf',
    'senegal': 'africa/senegal-latest.osm.pbf',
    'seychelles': 'africa/seychelles-latest.osm.pbf',
    'sierra-leone': 'africa/sierra-leone-latest.osm.pbf',
    'somalia': 'africa/somalia-latest.osm.pbf',
    'south-africa': 'africa/south-africa-latest.osm.pbf',
    'south-sudan': 'africa/south-sudan-latest.osm.pbf',
    'sudan': 'africa/sudan-latest.osm.pbf',
    'tanzania': 'africa/tanzania-latest.osm.pbf',
    'togo': 'africa/togo-latest.osm.pbf',
    'tunisia': 'africa/tunisia-latest.osm.pbf',
    'uganda': 'africa/uganda-latest.osm.pbf',
    'zambia': 'africa/zambia-latest.osm.pbf',
    'zimbabwe': 'africa/zimbabwe-latest.osm.pbf',
    
    # Add more continents as needed...
    # For now, focusing on Africa for sub-Saharan use case
}


# Approximate bounding boxes for region detection
# Format: (min_lon, min_lat, max_lon, max_lat)
REGION_BOUNDS = {
    'kenya': (33.908859, -4.678047, 41.899078, 5.506),
    'tanzania': (29.327, -11.745, 40.443, -0.984),
    'uganda': (29.573, -1.475, 35.000, 4.234),
    'rwanda': (28.861, -2.840, 30.899, -1.047),
    'ethiopia': (32.997, 3.397, 47.986, 14.880),
    'nigeria': (2.668, 4.240, 14.680, 13.892),
    'ghana': (-3.260, 4.710, 1.191, 11.174),
    'south-africa': (16.458, -34.819, 32.895, -22.126),
    # Add more as needed...
}




# ==================== METADATA MANAGEMENT ====================


class PBFMetadata:
    """Metadata for cached PBF files."""
    
    def __init__(
        self,
        region: str,
        download_date: datetime,
        file_size_mb: float,
        geofabrik_url: str,
        pbf_path: Path
    ):
        self.region = region
        self.download_date = download_date
        self.file_size_mb = file_size_mb
        self.geofabrik_url = geofabrik_url
        self.pbf_path = str(pbf_path)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'region': self.region,
            'download_date': self.download_date.isoformat(),
            'file_size_mb': round(self.file_size_mb, 2),
            'geofabrik_url': self.geofabrik_url,
            'pbf_path': self.pbf_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PBFMetadata':
        """Load from dictionary."""
        return cls(
            region=data['region'],
            download_date=datetime.fromisoformat(data['download_date']),
            file_size_mb=data['file_size_mb'],
            geofabrik_url=data['geofabrik_url'],
            pbf_path=Path(data['pbf_path'])
        )
    
    def age_days(self) -> int:
        """Calculate age in days."""
        return (datetime.now() - self.download_date).days




def save_metadata(pbf_path: Path, metadata: PBFMetadata) -> None:
    """Save metadata JSON file alongside PBF."""
    meta_path = pbf_path.with_suffix('.osm.pbf.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    logger.debug(f"Saved metadata: {meta_path}")




def load_metadata(pbf_path: Path) -> Optional[PBFMetadata]:
    """Load metadata from JSON file if it exists."""
    meta_path = pbf_path.with_suffix('.osm.pbf.meta.json')
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path, 'r') as f:
            data = json.load(f)
        return PBFMetadata.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load metadata from {meta_path}: {e}")
        return None




# ==================== REGION DETECTION ====================


def detect_regions_from_aoi(aoi_geometry) -> List[str]:
    """
    Auto-detect Geofabrik regions that intersect with AOI.
    
    Args:
        aoi_geometry: Shapely geometry or GeoDataFrame
        
    Returns:
        List of region names that intersect the AOI
        
    Example:
        regions = detect_regions_from_aoi(nairobi_polygon)
        # Returns: ['kenya']
    """
    import geopandas as gpd
    
    # Ensure we're working in WGS84 for comparison with REGION_BOUNDS
    if hasattr(aoi_geometry, 'crs'):
        # It's a GeoDataFrame or GeoSeries
        if aoi_geometry.crs is None:
            logger.warning("AOI has no CRS, assuming EPSG:4326")
            aoi_geometry = aoi_geometry.copy()
            aoi_geometry.set_crs("EPSG:4326", inplace=True)
        elif aoi_geometry.crs != "EPSG:4326":
            logger.info(f"Reprojecting AOI from {aoi_geometry.crs} to EPSG:4326 for region detection")
            aoi_geometry = aoi_geometry.to_crs("EPSG:4326")
        
        bounds = aoi_geometry.unary_union.bounds
    else:
        # Shapely geometry - assume it's already in WGS84
        bounds = aoi_geometry.bounds
    
    aoi_box = box(*bounds)
    
    # Check intersection with known regions
    intersecting = []
    for region, region_bounds in REGION_BOUNDS.items():
        region_box = box(*region_bounds)
        if aoi_box.intersects(region_box):
            intersecting.append(region)
    
    if not intersecting:
        # Fallback: try to guess from centroid
        logger.warning(
            "Could not auto-detect region from AOI bounds. "
            "Please specify geofabrik_regions manually."
        )
        raise ValueError(
            "Unable to auto-detect Geofabrik region. "
            "Please set 'geofabrik_regions' in config. "
            f"AOI bounds: {bounds}"
        )
    
    logger.info(f"Auto-detected regions: {intersecting}")
    return intersecting




# ==================== DOWNLOAD MANAGEMENT ====================


def get_geofabrik_url(region: str) -> str:
    """
    Get Geofabrik download URL for region.
    
    Args:
        region: Region name (e.g., 'kenya')
        
    Returns:
        Full download URL
        
    Raises:
        ValueError: If region not found
    """
    if region not in GEOFABRIK_REGIONS:
        raise ValueError(
            f"Unknown region: {region}. "
            f"Available: {', '.join(sorted(GEOFABRIK_REGIONS.keys()))}"
        )
    
    path = GEOFABRIK_REGIONS[region]
    return f"https://download.geofabrik.de/{path}"




def download_pbf(
    region: str,
    cache_dir: Path,
    force: bool = False
) -> Path:
    """
    Download PBF file from Geofabrik with progress bar.
    
    Args:
        region: Region name
        cache_dir: Directory to save PBF
        force: Force re-download even if cached
        
    Returns:
        Path to downloaded PBF file
        
    Example:
        pbf_path = download_pbf('kenya', Path('~/.geoworkflow/osm_cache'))
    """
    pbf_path = cache_dir / f"{region}-latest.osm.pbf"
    
    # Check if already cached
    if pbf_path.exists() and not force:
        metadata = load_metadata(pbf_path)
        if metadata:
            age = metadata.age_days()
            logger.info(
                f"Using cached PBF: {pbf_path} "
                f"(downloaded {age} days ago, {metadata.file_size_mb:.1f} MB)"
            )
            return pbf_path
        else:
            logger.info(f"Using cached PBF: {pbf_path} (no metadata)")
            return pbf_path
    
    # Download
    url = get_geofabrik_url(region)
    logger.info(f"Downloading {region} from Geofabrik: {url}")
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(pbf_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {region}",
                leave=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Save metadata
        file_size_mb = pbf_path.stat().st_size / (1024 ** 2)
        metadata = PBFMetadata(
            region=region,
            download_date=datetime.now(),
            file_size_mb=file_size_mb,
            geofabrik_url=url,
            pbf_path=pbf_path
        )
        save_metadata(pbf_path, metadata)
        
        logger.info(f"Downloaded {file_size_mb:.1f} MB to {pbf_path}")
        return pbf_path
        
    except Exception as e:
        logger.error(f"Failed to download {region}: {e}")
        # Clean up partial download
        if pbf_path.exists():
            pbf_path.unlink()
        raise




def get_cached_pbf(
    region: str,
    cache_dir: Path,
    force_redownload: bool = False,
    max_age_days: Optional[int] = None
) -> Tuple[Path, PBFMetadata]:
    """
    Get cached PBF or download if not present.
    
    Args:
        region: Region name
        cache_dir: Cache directory
        force_redownload: Force re-download
        max_age_days: Warn if older than N days (but still use it)
        
    Returns:
        Tuple of (pbf_path, metadata)
        
    Example:
        pbf_path, meta = get_cached_pbf('kenya', cache_dir, max_age_days=30)
        if meta.age_days() > 30:
            print("Warning: Data is old")
    """
    pbf_path = download_pbf(region, cache_dir, force=force_redownload)
    metadata = load_metadata(pbf_path)
    
    # Check age and warn if needed
    if metadata and max_age_days is not None:
        age = metadata.age_days()
        if age > max_age_days:
            logger.warning(
                f"Cached PBF for {region} is {age} days old "
                f"(threshold: {max_age_days} days). "
                f"Consider setting force_redownload=True for fresh data."
            )
    
    return pbf_path, metadata




def list_cached_pbfs(cache_dir: Path) -> List[Tuple[str, PBFMetadata]]:
    """
    List all cached PBF files with metadata.
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        List of (region, metadata) tuples
        
    Example:
        for region, meta in list_cached_pbfs(cache_dir):
            print(f"{region}: {meta.age_days()} days old, {meta.file_size_mb} MB")
    """
    cached = []
    for pbf_file in cache_dir.glob("*.osm.pbf"):
        # Extract region name
        region = pbf_file.stem.replace('-latest.osm', '')
        metadata = load_metadata(pbf_file)
        if metadata:
            cached.append((region, metadata))
    
    return sorted(cached, key=lambda x: x[1].download_date, reverse=True)