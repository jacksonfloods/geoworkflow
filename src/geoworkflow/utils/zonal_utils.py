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
    from rasterio.io import MemoryFile
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


def compute_zonal_statistics_stacked(
    stack: "np.ndarray",
    transform,
    crs,
    nodata,
    vectors: "gpd.GeoDataFrame",
    stat_names: List[str],
    band_labels: list,
    *,
    include_cols: Optional[List[str]] = None,
    reproject: bool = True,
) -> "pd.DataFrame":
    """Coverage-weighted zonal stats over a multi-band stack, in one exactextract pass.

    Stacks N single-band rasters that share the same grid (e.g. monthly slices of
    one variable, all clipped to the same AOI) into one N-band image. exactextract
    computes the polygon coverage **once** and reuses it across all bands, so this
    is ~N times faster than calling :func:`compute_zonal_statistics` N times.

    Args:
        stack: 3-D array ``(n_bands, height, width)``.
        transform, crs, nodata: shared georeferencing of the stack.
        vectors: polygons (the hex grid); reprojected to ``crs`` if needed.
        stat_names: statistic names. **Only op-backed stats** are supported here;
            reducer-backed stats must use the single-band engine.
        band_labels: one label per band (e.g. the timestamps), length ``n_bands``.
        include_cols: vector attribute columns to carry through (e.g. ``["GridID"]``).

    Returns:
        Tidy long DataFrame: ``include_cols`` + ``band`` (the label) + ``statistic``
        + ``value`` — one row per (feature, band, statistic).
    """
    if not HAS_ZONAL_LIBS:
        raise ImportError(
            "Zonal statistics require exactextract, rasterio, geopandas and pandas."
        )
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (n_bands, h, w); got shape {stack.shape}")
    n_bands = stack.shape[0]
    if len(band_labels) != n_bands:
        raise ValueError(f"band_labels ({len(band_labels)}) must match n_bands ({n_bands})")

    stats = resolve_statistics(stat_names)
    if any(s.is_reducer for s in stats):
        raise ValueError(
            "compute_zonal_statistics_stacked supports only op-backed statistics; "
            "use compute_zonal_statistics for reducer-backed ones."
        )
    include_cols = list(include_cols or [])

    ee_ops: List[str] = []
    seen: set[str] = set()
    for s in stats:
        if s.exactextract_op not in seen:
            ee_ops.append(s.exactextract_op)
            seen.add(s.exactextract_op)

    profile = {
        "driver": "GTiff",
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "count": n_bands,
        "dtype": stack.dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(stack)
        with memfile.open() as src:
            vecs = vectors
            if (
                reproject
                and src.crs is not None
                and vectors.crs is not None
                and str(vectors.crs) != str(src.crs)
            ):
                vecs = vectors.to_crs(src.crs)
            raw = exact_extract(src, vecs, ee_ops, output="pandas", include_cols=include_cols)

    base = {c: raw[c].to_numpy() for c in include_cols}
    frames = []
    for i in range(n_bands):
        for s in stats:
            col = f"band_{i + 1}_{_op_to_column(s.exactextract_op)}"
            frames.append(pd.DataFrame({
                **base,
                "band": band_labels[i],
                "statistic": s.name,
                "value": raw[col].to_numpy(),
            }))
    return pd.concat(frames, ignore_index=True)


# Ops reconstructable from (coverage, values) in numpy, so the polygon coverage
# can be computed once and reused across all bands.
COVERAGE_REUSE_OPS = {"mean", "sum", "count"}


def compute_zonal_statistics_coverage_reuse(
    stack: "np.ndarray",
    transform,
    crs,
    nodata,
    vectors: "gpd.GeoDataFrame",
    stat_names: List[str],
    band_labels: list,
    *,
    include_cols: Optional[List[str]] = None,
    reproject: bool = True,
) -> "pd.DataFrame":
    """Multi-band coverage-weighted stats, computing the polygon coverage ONCE.

    exactextract's coverage (which cells each polygon overlaps, and the fraction)
    is geometry-only and band-independent. We compute it a single time (via a dummy
    all-valid band, so every covered cell is returned) and then reduce every band in
    vectorized numpy. For N bands this is ~N times faster than re-running exactextract
    per band, because the expensive polygon-vs-grid coverage is not repeated.

    Per-band nodata is handled correctly: a cell masked (NaN / nodata) in one band can
    still be valid in another, so the weighted mean's denominator excludes only the
    cells that are nodata *in that band*.

    Supports the reducible ops in :data:`COVERAGE_REUSE_OPS` (weighted_mean/mean, sum,
    count). For other stats use :func:`compute_zonal_statistics_stacked`.

    Returns a tidy long DataFrame: ``include_cols`` + ``band`` + ``statistic`` + ``value``.
    """
    if not HAS_ZONAL_LIBS:
        raise ImportError(
            "Zonal statistics require exactextract, rasterio, geopandas and pandas."
        )
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (n_bands, h, w); got shape {stack.shape}")
    n_bands, height, width = stack.shape
    if len(band_labels) != n_bands:
        raise ValueError(f"band_labels ({len(band_labels)}) must match n_bands ({n_bands})")

    stats = resolve_statistics(stat_names)
    if any(s.is_reducer or s.exactextract_op not in COVERAGE_REUSE_OPS for s in stats):
        raise ValueError(
            "coverage-reuse supports only weighted_mean/mean, sum and count; "
            "use compute_zonal_statistics_stacked for other statistics."
        )
    include_cols = list(include_cols or [])

    # 1) Coverage computed ONCE on a dummy all-valid band (so all covered cells appear).
    dummy = np.ones((height, width), dtype="float32")
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
               "dtype": "float32", "crs": crs, "transform": transform}
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(dummy, 1)
        with memfile.open() as src:
            vecs = vectors
            if (reproject and src.crs is not None and vectors.crs is not None
                    and str(vectors.crs) != str(src.crs)):
                vecs = vectors.to_crs(src.crs)
            cov = exact_extract(src, vecs, ["cell_id", "coverage"],
                                output="pandas", include_cols=include_cols)

    n_poly = len(cov)
    cell_lists = cov["cell_id"].to_numpy()
    cov_lists = cov["coverage"].to_numpy()
    lengths = np.fromiter((len(c) for c in cell_lists), dtype=np.int64, count=n_poly)
    if lengths.sum():
        all_cells = np.concatenate([np.asarray(c, np.int64) for c in cell_lists])
        all_cov = np.concatenate([np.asarray(c, np.float64) for c in cov_lists])
        poly_idx = np.repeat(np.arange(n_poly), lengths)
    else:
        all_cells = np.empty(0, np.int64)
        all_cov = np.empty(0, np.float64)
        poly_idx = np.empty(0, np.int64)

    # 2) Gather every band's values at the covered cells once.
    stack_flat = stack.reshape(n_bands, height * width).astype(np.float64)
    vals_all = stack_flat[:, all_cells] if all_cells.size else np.empty((n_bands, 0))
    nodata_is_nan = nodata is None or (isinstance(nodata, float) and np.isnan(nodata))
    want = {s.exactextract_op for s in stats}

    base = {c: cov[c].to_numpy() for c in include_cols}
    frames = []
    for bi in range(n_bands):
        if all_cells.size:
            v = vals_all[bi]
            valid = ~np.isnan(v)
            if not nodata_is_nan:
                valid &= (v != nodata)
            cov_valid = np.where(valid, all_cov, 0.0)
            den = np.bincount(poly_idx, weights=cov_valid, minlength=n_poly)
            num = np.bincount(poly_idx, weights=np.where(valid, all_cov * v, 0.0),
                              minlength=n_poly)
        else:
            den = np.zeros(n_poly)
            num = np.zeros(n_poly)

        per_op = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            if "mean" in want:
                per_op["mean"] = np.where(den > 0, num / den, np.nan)
            if "sum" in want:
                per_op["sum"] = np.where(den > 0, num, np.nan)
            if "count" in want:
                per_op["count"] = den

        for s in stats:
            frames.append(pd.DataFrame({
                **base,
                "band": band_labels[bi],
                "statistic": s.name,
                "value": per_op[s.exactextract_op],
            }))
    return pd.concat(frames, ignore_index=True)
