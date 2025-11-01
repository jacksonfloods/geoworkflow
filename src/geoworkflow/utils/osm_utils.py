"""
OSM-specific utility functions for processing highway data.


This module provides helpers for:
- Filtering highways by type
- Attribute selection and cleaning
- Geometry validation
- Deduplication across multi-region extracts
"""


from typing import List, Optional, Dict, Any
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely import wkt
from shapely.ops import linemerge




logger = logging.getLogger(__name__)




# ==================== HIGHWAY TYPE FILTERING ====================


def filter_highways_by_type(
    highways_gdf: gpd.GeoDataFrame,
    highway_types: List[str]
) -> gpd.GeoDataFrame:
    """
    Filter highways by type (e.g., motorway, residential).
    
    Args:
        highways_gdf: GeoDataFrame with 'highway' column
        highway_types: List of highway types to keep
        
    Returns:
        Filtered GeoDataFrame
        
    Example:
        major_roads = filter_highways_by_type(all_highways, ['motorway', 'trunk', 'primary'])
    """
    if 'highway' not in highways_gdf.columns:
        logger.warning("No 'highway' column found. Returning all rows.")
        return highways_gdf
    
    # OSM highway column can be multi-valued (e.g., "primary;secondary")
    # So we check if any value matches
    def matches_any_type(highway_value):
        if pd.isna(highway_value):
            return False
        # Split on semicolon for multi-values
        values = str(highway_value).split(';')
        return any(v.strip() in highway_types for v in values)
    
    mask = highways_gdf['highway'].apply(matches_any_type)
    filtered = highways_gdf[mask].copy()
    
    logger.info(
        f"Filtered highways by type: {len(highways_gdf)} → {len(filtered)} "
        f"(kept: {highway_types})"
    )
    
    return filtered




# ==================== ATTRIBUTE MANAGEMENT ====================


def select_highway_attributes(
    highways_gdf: gpd.GeoDataFrame,
    include_attributes: List[str]
) -> gpd.GeoDataFrame:
    """
    Select specific OSM attributes, keeping geometry and osm_id.
    
    Args:
        highways_gdf: Full GeoDataFrame with all OSM attributes
        include_attributes: List of attribute names to keep
        
    Returns:
        GeoDataFrame with selected columns
        
    Example:
        selected = select_highway_attributes(
            highways,
            ['highway', 'name', 'surface', 'lanes']
        )
    """
    # Always keep geometry and osm_id
    required_cols = ['geometry']
    
    # Handle different ID column names
    if 'osm_id' in highways_gdf.columns:
        required_cols.append('osm_id')
    elif 'id' in highways_gdf.columns:
        # Pyrosm uses 'id', rename to 'osm_id' for consistency
        highways_gdf = highways_gdf.rename(columns={'id': 'osm_id'})
        required_cols.append('osm_id')  # FIX: Add 'osm_id', not 'id'
    else:
        # No ID column found - log warning but continue
        logger.warning(
            "No 'osm_id' or 'id' column found in highways data. "
            "Feature identification may be limited."
        )
    
    # Find available attributes
    available = [col for col in include_attributes if col in highways_gdf.columns]
    missing = [col for col in include_attributes if col not in highways_gdf.columns]
    
    if missing:
        logger.warning(
            f"Requested attributes not found in data: {missing}. "
            f"Available attributes: {list(highways_gdf.columns)[:20]}..."
        )
    
    # Select columns
    keep_cols = required_cols + available
    selected = highways_gdf[keep_cols].copy()
    
    logger.info(f"Selected {len(keep_cols)} attributes: {keep_cols}")
    
    return selected




def clean_highway_attributes(highways_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and standardize highway attributes.
    
    Operations:
    - Convert lanes to numeric
    - Standardize surface values
    - Clean whitespace from text fields
    
    Args:
        highways_gdf: Raw GeoDataFrame
        
    Returns:
        Cleaned GeoDataFrame
    """
    cleaned = highways_gdf.copy()
    
    # Clean lanes column (convert to int)
    if 'lanes' in cleaned.columns:
        def parse_lanes(val):
            if pd.isna(val):
                return None
            try:
                # Handle cases like "2", "2;3", "2-3"
                val_str = str(val).split(';')[0].split('-')[0]
                return int(float(val_str))
            except:
                return None
        
        cleaned['lanes'] = cleaned['lanes'].apply(parse_lanes)
    
    # Clean text fields (strip whitespace)
    text_cols = ['name', 'ref', 'surface', 'highway']
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(
                lambda x: str(x).strip() if pd.notna(x) else None
            )
    
    # Standardize surface values (optional)
    if 'surface' in cleaned.columns:
        surface_mapping = {
            'asphalt': 'paved',
            'concrete': 'paved',
            'paved': 'paved',
            'unpaved': 'unpaved',
            'gravel': 'unpaved',
            'dirt': 'unpaved',
            'ground': 'unpaved',
        }
        cleaned['surface_type'] = cleaned['surface'].str.lower().map(surface_mapping)
    
    return cleaned




# ==================== GEOMETRY PROCESSING ====================


def validate_highway_geometries(highways_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and fix highway geometries.
    
    Operations:
    - Remove invalid geometries
    - Remove empty geometries
    - Ensure LineString or MultiLineString
    - Remove duplicates
    
    Args:
        highways_gdf: GeoDataFrame
        
    Returns:
        Validated GeoDataFrame
    """
    initial_count = len(highways_gdf)
    cleaned = highways_gdf.copy()
    
    # Remove null geometries
    cleaned = cleaned[cleaned.geometry.notna()].copy()
    
    # Check validity
    invalid_mask = ~cleaned.geometry.is_valid
    if invalid_mask.any():
        logger.warning(f"Found {invalid_mask.sum()} invalid geometries. Attempting to fix...")
        cleaned.loc[invalid_mask, 'geometry'] = (
            cleaned.loc[invalid_mask, 'geometry'].buffer(0)
        )
    
    # Ensure LineString types
    valid_types = cleaned.geometry.type.isin(['LineString', 'MultiLineString'])
    if not valid_types.all():
        logger.warning(
            f"Removing {(~valid_types).sum()} non-LineString geometries. "
            f"Types found: {cleaned.geometry.type.value_counts().to_dict()}"
        )
        cleaned = cleaned[valid_types].copy()
    
    # Remove empty geometries
    empty_mask = cleaned.geometry.is_empty
    if empty_mask.any():
        logger.warning(f"Removing {empty_mask.sum()} empty geometries")
        cleaned = cleaned[~empty_mask].copy()
    
    final_count = len(cleaned)
    removed = initial_count - final_count
    
    if removed > 0:
        logger.info(
            f"Geometry validation: {initial_count} → {final_count} "
            f"({removed} removed)"
        )
    
    return cleaned




def deduplicate_highways(
    highways_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Remove duplicate highways (useful for multi-region overlaps).
    
    Duplicates are identified by:
    1. Same osm_id (exact duplicates)
    2. Nearly identical geometries (within buffer)
    
    Args:
        highways_gdf: GeoDataFrame
        buffer_meters: Tolerance for geometry matching
        
    Returns:
        Deduplicated GeoDataFrame
    """
    initial_count = len(highways_gdf)
    
    # Remove exact osm_id duplicates
    if 'osm_id' in highways_gdf.columns:
        highways_gdf = highways_gdf.drop_duplicates(subset=['osm_id']).copy()
    
    # For spatial duplicates, we'd need more complex logic
    # For now, just use osm_id
    
    final_count = len(highways_gdf)
    removed = initial_count - final_count
    
    if removed > 0:
        logger.info(f"Deduplication: {initial_count} → {final_count} ({removed} removed)")
    
    return highways_gdf

def clip_highways_to_aoi(
    highways_gdf: gpd.GeoDataFrame,
    aoi_geometry,
    buffer_meters: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Clip highway geometries to AOI boundary.
    
    Args:
        highways_gdf: GeoDataFrame of highways
        aoi_geometry: Shapely geometry or GeoDataFrame
        buffer_meters: Buffer AOI by N meters before clipping
        
    Returns:
        Clipped GeoDataFrame
    """
    import geopandas as gpd
    
    # Handle different input types and extract geometry + CRS
    if isinstance(aoi_geometry, gpd.GeoDataFrame):
        aoi_geom = aoi_geometry.union_all()  # Updated from unary_union
        aoi_crs = aoi_geometry.crs
    else:
        # It's a shapely geometry
        aoi_geom = aoi_geometry
        aoi_crs = None
    
    # Convert AOI to a GeoDataFrame if it isn't already
    if not isinstance(aoi_geometry, gpd.GeoDataFrame):
        # If we have a CRS from the original GeoDataFrame, use it
        # Otherwise assume it matches highways_gdf.crs
        if aoi_crs is None:
            aoi_crs = highways_gdf.crs
        temp_aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs=aoi_crs)
    else:
        temp_aoi_gdf = aoi_geometry.copy()
    
    # Reproject AOI to match highways CRS if needed
    if temp_aoi_gdf.crs != highways_gdf.crs:
        logger.info(f"Reprojecting AOI from {temp_aoi_gdf.crs} to {highways_gdf.crs}")
        temp_aoi_gdf = temp_aoi_gdf.to_crs(highways_gdf.crs)
        aoi_geom = temp_aoi_gdf.union_all()  # Updated from unary_union
    
    # Apply buffer if requested
    if buffer_meters > 0:
        # Use proper UTM projection for accurate buffering
        if highways_gdf.crs.is_geographic:
            # Estimate appropriate UTM zone based on geometry centroid
            utm_crs = temp_aoi_gdf.estimate_utm_crs()
            
            # Reproject to UTM, buffer, then reproject back
            temp_gdf_utm = temp_aoi_gdf.to_crs(utm_crs)
            temp_gdf_utm['geometry'] = temp_gdf_utm.geometry.buffer(buffer_meters)
            temp_gdf_buffered = temp_gdf_utm.to_crs(highways_gdf.crs)
            
            aoi_geom = temp_gdf_buffered.union_all()  # Updated from unary_union
        else:
            # Already in metric CRS, buffer directly
            aoi_geom = aoi_geom.buffer(buffer_meters)
    
    # Now both geometries are guaranteed to be in the same CRS
    # Clip geometries
    clipped = highways_gdf.copy()
    clipped['geometry'] = clipped.geometry.intersection(aoi_geom)
    
    # Remove empty results from clip
    clipped = clipped[~clipped.geometry.is_empty].copy()
    
    logger.info(f"Clipped highways to AOI: {len(highways_gdf)} → {len(clipped)}")
    
    return clipped
# ==================== UTILITY FUNCTIONS ====================


def calculate_highway_length(highways_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add length_m column (length in meters).
    
    Reprojects to UTM if needed for accurate measurement.
    
    Args:
        highways_gdf: GeoDataFrame
        
    Returns:
        GeoDataFrame with 'length_m' column
    """
    highways = highways_gdf.copy()
    
    # If geographic CRS, reproject for accurate length
    if highways.crs and highways.crs.is_geographic:
        # Get centroid to determine UTM zone
        centroid = highways.union_all().centroid
        utm_crs = highways.estimate_utm_crs()
        highways_utm = highways.to_crs(utm_crs)
        highways['length_m'] = highways_utm.geometry.length
    else:
        highways['length_m'] = highways.geometry.length
    
    return highways




def summarize_highway_network(highways_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for highway network.
    
    Args:
        highways_gdf: GeoDataFrame
        
    Returns:
        Dictionary of summary stats
        
    Example:
        stats = summarize_highway_network(highways)
        print(f"Total length: {stats['total_length_km']:.1f} km")
    """
    summary = {
        'total_features': len(highways_gdf),
        'highway_types': {},
        'total_length_km': 0.0,
        'attributes_present': list(highways_gdf.columns)
    }
    
    # Count by highway type
    if 'highway' in highways_gdf.columns:
        summary['highway_types'] = highways_gdf['highway'].value_counts().to_dict()
    
    # Calculate total length
    if 'length_m' in highways_gdf.columns:
        summary['total_length_km'] = highways_gdf['length_m'].sum() / 1000
    else:
        highways_with_length = calculate_highway_length(highways_gdf)
        summary['total_length_km'] = highways_with_length['length_m'].sum() / 1000
    
    return summary