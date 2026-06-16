"""Lazy hex geometry: read per-city GeoPackages and join warehouse values.

Geometry stays in the per-city ``.gpkg`` (each in its own UTM — the source of
truth) and is never duplicated into the warehouse. For plotting we read just the
needed columns and map values onto them by ``GridID`` — the same idiom as
``hexcube.to_geodataframe``. Plotting several cities together must first reproject
each out of its own UTM into a shared CRS (the one real cross-city correctness
trap), which :func:`concat_for_plot` centralizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from geoworkflow.core.exceptions import GeoWorkflowError

try:
    import geopandas as gpd
    import pandas as pd
    HAS_GEO = True
except ImportError:  # pragma: no cover
    HAS_GEO = False

GRID_ID = "GridID"


def load_hex_geometry(
    gpkg: Path,
    *,
    gridids: Optional[Sequence[str]] = None,
    grid_id_column: str = GRID_ID,
) -> "gpd.GeoDataFrame":
    """Read ``[GridID, geometry]`` from a city's hex GeoPackage (native UTM)."""
    if not HAS_GEO:  # pragma: no cover
        raise GeoWorkflowError("geopandas is required for geometry/plotting")
    gpkg = Path(gpkg)
    if not gpkg.exists():
        raise GeoWorkflowError(f"hex geometry not found: {gpkg}")
    gdf = gpd.read_file(gpkg, columns=[grid_id_column])[[grid_id_column, "geometry"]]
    if gridids is not None:
        gdf = gdf[gdf[grid_id_column].isin(set(gridids))]
    return gdf


def join_values(
    gdf: "gpd.GeoDataFrame",
    values: "pd.DataFrame",
    *,
    value_col: str = "value",
    out_col: str = "value",
    grid_id_column: str = GRID_ID,
) -> "gpd.GeoDataFrame":
    """Attach a per-hex value column to geometry, mapped by GridID.

    ``values`` is a DataFrame with a GridID column and a value column; hexes
    absent from ``values`` get NaN (uncoloured), matching the cube behaviour.
    """
    series = pd.Series(values[value_col].values,
                       index=values[grid_id_column].astype(str).values)
    out = gdf.copy()
    out[out_col] = out[grid_id_column].astype(str).map(series)
    return out


def concat_for_plot(
    gdfs: List["gpd.GeoDataFrame"],
    target_crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """Reproject each city's geometry to a shared CRS, then concatenate.

    Each city is stored in its own UTM, so they must be reprojected to one CRS
    before they can share a plot/frame.
    """
    if not gdfs:
        raise GeoWorkflowError("nothing to concatenate")
    projected = [g.to_crs(target_crs) for g in gdfs]
    return gpd.GeoDataFrame(pd.concat(projected, ignore_index=True), crs=target_crs)
