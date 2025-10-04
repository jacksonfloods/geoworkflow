"""
S2 geometry utilities using s2sphere (Windows-compatible).
Replaces s2geometry functionality from Google Colab notebook.
"""
from typing import List, Tuple, Optional
from s2sphere import CellId, Cell, RegionCoverer, LatLng, LatLngRect
import shapely.geometry
from shapely.geometry.base import BaseGeometry


def get_bounding_box_s2_covering_tokens(
    geometry: BaseGeometry, 
    level: int = 6,
    max_cells: int = 1_000_000
) -> List[str]:
    """
    Get S2 cell tokens covering a geometry's bounding box.
    
    Args:
        geometry: Shapely geometry to cover
        level: S2 cell level (default 6, matches GCS bucket structure)
        max_cells: Maximum number of cells to return
        
    Returns:
        List of S2 token strings
        
    Example:
        >>> from shapely.geometry import box
        >>> bbox = box(-1, 5, 0, 6)
        >>> tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
    """
    # Get bounding box
    minx, miny, maxx, maxy = geometry.bounds
    
    # Create S2 LatLngRect from bounds
    # s2geometry: s2.S2LatLng_FromDegrees(lat, lng)
    # s2sphere: LatLng.from_degrees(lat, lng)
    lo = LatLng.from_degrees(miny, minx)
    hi = LatLng.from_degrees(maxy, maxx)
    
    # s2geometry: s2.S2LatLngRect_FromPointPair(lo, hi)
    # s2sphere: LatLngRect.from_point_pair(lo, hi)
    rect = LatLngRect.from_point_pair(lo, hi)
    
    # Create region coverer
    # s2geometry: s2.S2RegionCoverer()
    # s2sphere: RegionCoverer()
    coverer = RegionCoverer()
    
    # Set level
    # s2geometry: coverer.set_fixed_level(6)
    # s2sphere: coverer.min_level = coverer.max_level = 6
    coverer.min_level = level
    coverer.max_level = level
    
    # Set max cells
    # s2geometry: coverer.set_max_cells(max_cells)
    # s2sphere: coverer.max_cells = max_cells
    coverer.max_cells = max_cells
    
    # Get covering
    covering = coverer.get_covering(rect)
    
    # Convert to tokens
    # s2geometry: cell.ToToken()
    # s2sphere: cell.to_token()
    tokens = [cell.to_token() for cell in covering]
    
    return tokens


def s2_token_to_shapely_polygon(token: str) -> shapely.geometry.Polygon:
    """
    Convert S2 token to Shapely polygon.
    
    Args:
        token: S2 cell token string
        
    Returns:
        Shapely Polygon representing the S2 cell
        
    Example:
        >>> polygon = s2_token_to_shapely_polygon("89c25")
        >>> print(polygon.bounds)
    """
    # s2geometry: s2.S2CellId_FromToken(token)
    # s2sphere: CellId.from_token(token)
    cell_id = CellId.from_token(token)
    cell = Cell(cell_id)
    
    # Get vertices (4 corners)
    vertices = []
    for i in range(4):
        vertex = cell.get_vertex(i)
        lat_lng = LatLng.from_point(vertex)
        vertices.append((lat_lng.lng().degrees, lat_lng.lat().degrees))
    
    # Close the polygon
    vertices.append(vertices[0])
    
    return shapely.geometry.Polygon(vertices)


def estimate_cells_for_geometry(geometry: BaseGeometry, level: int = 6) -> int:
    """
    Estimate number of S2 cells needed to cover geometry.
    
    Args:
        geometry: Shapely geometry
        level: S2 cell level
        
    Returns:
        Estimated number of cells
        
    Note:
        This is a rough estimate based on bounding box area.
        Actual cell count may vary.
    """
    # Get bounding box area in square degrees
    minx, miny, maxx, maxy = geometry.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    
    # Approximate cell size at different levels (in square degrees)
    # Level 6: ~1.27 degrees, Level 7: ~0.63 degrees, Level 8: ~0.32 degrees
    cell_sizes = {
        4: 5.09,
        5: 2.54,
        6: 1.27,
        7: 0.635,
        8: 0.318,
    }
    
    cell_size = cell_sizes.get(level, 1.27)
    estimated_cells = int(bbox_area / (cell_size * cell_size)) + 1
    
    return estimated_cells


def validate_s2_level(level: int) -> bool:
    """
    Validate S2 level is appropriate for Open Buildings data.
    
    Args:
        level: S2 cell level
        
    Returns:
        True if valid, False otherwise
        
    Note:
        Open Buildings v3 uses level 6 in GCS bucket structure.
        Other levels may not have corresponding data.
    """
    return 4 <= level <= 8


def get_s2_cell_area(level: int) -> float:
    """
    Get approximate area of S2 cell at given level in km².
    
    Args:
        level: S2 cell level
        
    Returns:
        Approximate area in km²
    """
    # Earth's surface area: ~510 million km²
    # Number of cells at level: 6 * 4^level
    earth_area_km2 = 510_000_000
    num_cells = 6 * (4 ** level)
    return earth_area_km2 / num_cells
