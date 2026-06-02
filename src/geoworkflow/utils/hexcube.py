"""
Query helpers for hexagon (cell, time) NetCDF cubes built by Pipeline 2.

The cube has dims ``(cell, time)`` with ``cell`` indexed by GridID and ``lat`` /
``lon`` / ``q`` / ``r`` as per-cell coordinates. These helpers wrap the common
xarray idioms (and the two operations that differ from a regular lat/lon raster:
nearest-hex lookup and joining values back to hex geometry for mapping).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

try:
    import numpy as np
    import pandas as pd
    import xarray as xr
    HAS_LIBS = True
except ImportError:  # pragma: no cover
    HAS_LIBS = False


def open_hex_cube(path: Union[str, Path]) -> "xr.Dataset":
    """Open a hex cube NetCDF written by the GridToNetCDFProcessor."""
    return xr.open_dataset(path)


def _squeeze_time(da: "xr.DataArray") -> "xr.DataArray":
    """Drop a length-1 time dim so a single-month selection is a clean (cell,) map."""
    if "time" in da.dims and da.sizes["time"] == 1:
        return da.squeeze("time", drop=True)
    return da


def select_month(ds: "xr.Dataset", variable: str, year: int, month: int) -> "xr.DataArray":
    return _squeeze_time(ds[variable].sel(time=f"{year}-{month:02d}"))
    # one month, value per hexagon, shape (cell,)


def annual_mean(ds: "xr.Dataset", variable: str, year: int) -> "xr.DataArray":
    return ds[variable].sel(time=slice(f"{year}-01", f"{year}-12")).mean("time")
    # per-hex mean across the 12 months of `year`, shape (cell,)


def month_across_years(ds: "xr.Dataset", variable: str, month: int) -> "xr.DataArray":
    return ds[variable].sel(time=ds.time.dt.month == month).mean("time")
    # per-hex mean across every occurrence of `month` (e.g. all Junes), shape (cell,)


def monthly_climatology(ds: "xr.Dataset", variable: str) -> "xr.DataArray":
    return ds[variable].groupby("time.month").mean()
    # per-hex mean for each calendar month 1..12, shape (cell, month)


def city_mean(ds: "xr.Dataset", variable: str) -> "xr.DataArray":
    return ds[variable].mean("cell")
    # spatial average over all hexes, one value per time step, shape (time,)


def nearest_cell(ds: "xr.Dataset", lon: float, lat: float) -> "xr.Dataset":
    """Select the single hexagon whose centroid is nearest to (lon, lat).

    The regular-raster ``.sel(x=..., y=..., method="nearest")`` does not apply to
    an irregular hex cube, so we find the nearest centroid explicitly.
    """
    d2 = (ds["lon"] - lon) ** 2 + (ds["lat"] - lat) ** 2
    return ds.isel(cell=int(d2.argmin().item()))
    # all variables/time for the closest hex; e.g. result["PM25"] is shape (time,)


def to_geodataframe(
    ds: "xr.Dataset",
    grid_file: Union[str, Path],
    variable: Optional[str] = None,
    time: Optional[str] = None,
    grid_id_column: str = "GridID",
):
    """Join cube values back onto hex geometry for mapping.

    Reads the hex geometry from ``grid_file`` and attaches either a single
    (variable, time) slice as one column, or — when ``time`` is omitted — the
    time-mean of ``variable``. Returns a GeoDataFrame ready for ``.plot(...)``.
    """
    import geopandas as gpd

    grid = gpd.read_file(grid_file)[[grid_id_column, "geometry"]].copy()

    if variable is None:
        variable = list(ds.data_vars)[0]
    da = ds[variable]
    if "time" in da.dims:
        da = _squeeze_time(da.sel(time=time)) if time is not None else da.mean("time")

    values = pd.Series(np.asarray(da.values), index=np.asarray(ds["cell"].values))
    col = variable if time is None else f"{variable}_{time}"
    grid[col] = grid[grid_id_column].astype(str).map(values)
    return grid
