"""
Coverage-weighted zonal statistics over raster cells, powered by exactextract.

This is the computational core behind the grid-statistics pipeline. Given a
single-band raster and a set of polygons (e.g. a hex grid), it returns one value
per (feature, statistic). Unlike a pixel-centroid approach, exactextract weights
each cell by the fraction of the cell covered by the polygon, so a hexagon
smaller than a raster pixel still receives that pixel's value.

The set of available statistics — and how to add new ones — is defined in
:mod:`geoworkflow.core.statistics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import geopandas as gpd  # noqa: F401  (used for type context / CRS handling)
    import pandas as pd
    import rasterio
    from exactextract import exact_extract
    HAS_ZONAL_LIBS = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    HAS_ZONAL_LIBS = False

from geoworkflow.core.statistics import ZonalStatistic, resolve_statistics

RasterInput = Union[str, Path, "rasterio.io.DatasetReader"]


def _op_to_column(op: str) -> str:
    """Map an exactextract op string to the column name it produces.

    Simple ops use the op name verbatim (``mean`` -> ``mean``). Quantiles use
    the percentage form exactextract emits (``quantile(q=0.25)`` -> ``quantile_25``).
    """
    if op.startswith("quantile(q="):
        q = float(op[len("quantile(q="):-1])
        pct = q * 100.0
        pct_str = str(int(pct)) if float(pct).is_integer() else str(pct)
        return f"quantile_{pct_str}"
    return op


def _safe_reduce(reducer, values: np.ndarray, coverage: np.ndarray) -> float:
    values = np.asarray(values)
    coverage = np.asarray(coverage)
    if values.size == 0:
        return float("nan")
    return float(reducer(values, coverage))


def compute_zonal_statistics(
    raster: RasterInput,
    vectors: "gpd.GeoDataFrame",
    stat_names: List[str],
    *,
    include_cols: Optional[List[str]] = None,
    reproject: bool = True,
) -> "pd.DataFrame":
    """Compute coverage-weighted zonal statistics for one raster band.

    Operates on the first band of ``raster``. Multi-band and temporal iteration
    is the responsibility of the raster-source layer, which hands single-band
    sources to this engine one slice at a time.

    Args:
        raster: Path to a raster, or an open single-band ``rasterio`` dataset.
        vectors: Polygons to summarize over (the hex grid). Reprojected to the
            raster CRS automatically when ``reproject`` is True.
        stat_names: Statistic names to compute (see
            :func:`geoworkflow.core.statistics.resolve_statistics`).
        include_cols: Vector attribute columns to carry through to the output
            (e.g. ``["GridID"]``).
        reproject: Reproject ``vectors`` to the raster CRS when they differ.

    Returns:
        A DataFrame with one row per input feature (same order as ``vectors``),
        the requested ``include_cols``, and one column per statistic name.

    Raises:
        ImportError: if the zonal stack (exactextract/rasterio) is unavailable.
    """
    if not HAS_ZONAL_LIBS:
        raise ImportError(
            "Zonal statistics require exactextract, rasterio, geopandas and "
            "pandas. Install with: pip install exactextract rasterio geopandas"
        )

    stats: List[ZonalStatistic] = resolve_statistics(stat_names)
    include_cols = list(include_cols or [])

    op_stats = [s for s in stats if not s.is_reducer]
    reducer_stats = [s for s in stats if s.is_reducer]

    # Unique exactextract ops to request (multiple stat names may share an op,
    # e.g. weighted_mean and mean both use "mean").
    ee_ops: List[str] = []
    seen: set[str] = set()
    for s in op_stats:
        if s.exactextract_op not in seen:
            ee_ops.append(s.exactextract_op)
            seen.add(s.exactextract_op)
    # Reducers need the raw per-cell arrays.
    if reducer_stats:
        for extra in ("values", "coverage"):
            if extra not in seen:
                ee_ops.append(extra)
                seen.add(extra)

    close_after = False
    if isinstance(raster, (str, Path)):
        src = rasterio.open(str(raster))
        close_after = True
    else:
        src = raster

    try:
        raster_crs = src.crs
        vecs = vectors
        if (
            reproject
            and raster_crs is not None
            and vectors.crs is not None
            and str(vectors.crs) != str(raster_crs)
        ):
            vecs = vectors.to_crs(raster_crs)

        raw = exact_extract(
            src,
            vecs,
            ee_ops,
            output="pandas",
            include_cols=include_cols,
        )
    finally:
        if close_after:
            src.close()

    out = pd.DataFrame(index=raw.index)
    for col in include_cols:
        out[col] = raw[col].to_numpy()

    for s in op_stats:
        column = _op_to_column(s.exactextract_op)
        out[s.name] = raw[column].to_numpy()

    if reducer_stats:
        values_arrays = raw["values"].to_numpy()
        coverage_arrays = raw["coverage"].to_numpy()
        for s in reducer_stats:
            out[s.name] = [
                _safe_reduce(s.reducer, v, c)
                for v, c in zip(values_arrays, coverage_arrays)
            ]

    return out
