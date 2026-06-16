"""Recipe-driven builder — zonal stats straight into the warehouse (no cubes).

``build(recipe, config)`` runs the pipeline from a recipe: resolve cities ->
(optionally fetch missing rasters) -> ensure grids -> compute zonal statistics
into ``data/hexdb/ISO3=.../aggid=.../part.parquet``. It is resumable
(skip-if-partition-exists, temp-then-rename) and reuses the existing
``compute_grid_statistics`` engine and ``parallel_map`` driver. The cube step is
never called.

``add_city`` / ``remove_city`` / ``add_metric`` / ``remove_metric`` edit the DB
and keep ``_recipe.yaml`` + ``_provenance.jsonl`` current (see
:mod:`geoworkflow.store.provenance`).
"""

from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from geoworkflow.core.exceptions import GeoWorkflowError
from geoworkflow.schemas.config_models import HexDBConfig, HexDBRecipe, MetricSpec
from geoworkflow.store.catalog import Catalog
from geoworkflow.store import provenance
from geoworkflow.utils.parallel import parallel_map, ParallelResult

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False

# ---- worker shared state (set in the parent, inherited via fork) -------
_CFG: Optional[HexDBConfig] = None
_CATALOG: Optional[Catalog] = None
_METRICS: List[MetricSpec] = []
_REGISTRY: Optional[Path] = None
_OVERWRITE = False
_REUSE = True
_MODE = "build"   # "build" (full city) or "add_metric" (append to existing)


def _catalog_for(config: HexDBConfig, *, grid_stats_dir=None) -> Catalog:
    return Catalog(
        Path(config.manifest_csv), Path(config.hexagglo_dir), Path(config.warehouse_dir),
        complexity_csv=config.complexity_csv, grid_stats_dir=grid_stats_dir,
    )


def _write_atomic(df: "pd.DataFrame", part: Path) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, part)


def _metric_inputs(metric: MetricSpec, record):
    """Raster inputs for a metric+city, or None if none are present.

    Global rasters (PM25, odiac) are shared -> pass the dataset dir. Per-city
    rasters live in one big per-ISO3 dir but are filenamed ``{aggid}_...`` -> pass
    ONLY this city's files (a list), so a city never scans its neighbours' tiles
    (the per-ISO3 dir can hold >200k files).
    """
    if metric.scope == "global":
        base = Path(_CFG.global_dir) / metric.folder()
        return [base] if base.exists() else None
    base = Path(_CFG.city_dir) / record.iso3 / metric.folder()
    if not base.exists():
        return None
    files = sorted(base.glob(f"{record.aggid}_*"))
    return files or None


def _compute_metric_frame(metric: MetricSpec, record, tmpdir: str) -> "pd.DataFrame":
    """Run zonal stats for one dataset and return its tidy rows (in memory)."""
    from geoworkflow.processors.integration.grid_statistics import compute_grid_statistics
    inputs = _metric_inputs(metric, record)
    if inputs is None:
        raise FileNotFoundError(f"missing rasters for {metric.name}")
    out = Path(tmpdir) / f"{metric.name}.parquet"
    compute_grid_statistics(
        grid_file=record.gpkg, raster_inputs=inputs, output_file=out,
        statistics=metric.statistics, dataset_registry=_REGISTRY,
    )
    return pd.read_parquet(out)


def _city_stats(record) -> "pd.DataFrame":
    """Compute the full tidy table (all recipe metrics) for one city."""
    with tempfile.TemporaryDirectory() as tmp:
        frames = [_compute_metric_frame(m, record, tmp) for m in _METRICS]
    df = pd.concat(frames, ignore_index=True)
    df["ISO3"] = record.iso3
    df["aggid"] = record.aggid
    return df


def build_city_worker(aggid: int):
    """Pool worker: build (or reuse) one city's partition. -> (aggid, status)."""
    try:
        rec = _CATALOG.resolve_aggid(aggid)
        part = rec.partition
        if part.exists() and not _OVERWRITE:
            return (aggid, "skip-exists")
        if not rec.gpkg.exists():
            return (aggid, "skip-no-grid")
        # Fast path: ingest a pre-computed legacy grid_stats parquet unchanged.
        if _REUSE and rec.combined_parquet and rec.combined_parquet.exists():
            df = pd.read_parquet(rec.combined_parquet)
            df["ISO3"], df["aggid"] = rec.iso3, aggid
            _write_atomic(df, part)
            return (aggid, "reused")
        # Skip cities still missing their per-city rasters (resumable).
        for m in _METRICS:
            if m.scope == "city" and _metric_inputs(m, rec) is None:
                return (aggid, "skip-no-rasters")
        _write_atomic(_city_stats(rec), part)
        return (aggid, "built")
    except Exception as e:  # noqa: BLE001 - keep the pool alive
        return (aggid, f"FAIL: {str(e)[:120]}")


def add_metric_worker(aggid: int):
    """Pool worker: append one new metric's rows to an existing partition."""
    try:
        rec = _CATALOG.resolve_aggid(aggid)
        part = rec.partition
        if not part.exists():
            return (aggid, "skip-no-partition")
        metric = _METRICS[0]
        existing = pd.read_parquet(part)
        present = set(existing["variable"].unique())
        with tempfile.TemporaryDirectory() as tmp:
            new = _compute_metric_frame(metric, rec, tmp)
        new_vars = set(new["variable"].unique())
        if new_vars & present and not _OVERWRITE:
            return (aggid, "skip-exists")
        existing = existing[~existing["variable"].isin(new_vars)]
        new["ISO3"], new["aggid"] = rec.iso3, aggid
        _write_atomic(pd.concat([existing, new], ignore_index=True), part)
        return (aggid, "built")
    except Exception as e:  # noqa: BLE001
        return (aggid, f"FAIL: {str(e)[:120]}")


def _set_globals(config, catalog, metrics, *, overwrite, reuse):
    global _CFG, _CATALOG, _METRICS, _REGISTRY, _OVERWRITE, _REUSE
    _CFG, _CATALOG, _METRICS = config, catalog, list(metrics)
    _REGISTRY = Path(config.raster_registry) if config.raster_registry else None
    _OVERWRITE, _REUSE = overwrite, reuse
    os.environ.setdefault("GEOWORKFLOW_QUIET_PROGRESS", "1")
    logging.getLogger("geoworkflow").setLevel(logging.ERROR)


def _write_metadata(config):
    from geoworkflow.store.metadata import Metadata
    Metadata().to_json(Path(config.warehouse_dir) / "metadata.json")


# ---- public API -------------------------------------------------------
def build(
    recipe: HexDBRecipe,
    config: Optional[HexDBConfig] = None,
    *,
    max_workers: int = 16,
    reuse: bool = True,
    grid_stats_dir: Optional[Path] = None,
    fetch: bool = False,
) -> ParallelResult:
    """Build (or resume) the warehouse from a recipe."""
    if not HAS_PANDAS:  # pragma: no cover
        raise GeoWorkflowError("pandas is required to build the hex database")
    config = config or HexDBConfig()
    warehouse = Path(config.warehouse_dir)

    if not recipe.overwrite:
        provenance.check_consistency(warehouse, recipe)

    catalog = _catalog_for(config, grid_stats_dir=grid_stats_dir)
    records = catalog.resolve_selector(recipe.cities)

    if fetch:
        _fetch_missing(recipe, config, records)

    _set_globals(config, catalog, recipe.metrics,
                 overwrite=recipe.overwrite, reuse=reuse)
    res = parallel_map(build_city_worker, [r.aggid for r in records],
                       max_workers=max_workers, label="cities", progress_every=200)

    _write_metadata(config)
    provenance.write_recipe(warehouse, recipe)
    provenance.append_op(warehouse, "build", cities=len(records),
                         built=res.done, skipped=res.skipped, failed=res.failed,
                         seconds=round(res.seconds, 1))
    return res


def add_city(config: HexDBConfig, iso3: str, name: Optional[str] = None, *,
             aggid: Optional[int] = None, overwrite: bool = False,
             grid_stats_dir: Optional[Path] = None):
    """Build one city consistently with the live recipe; update provenance."""
    warehouse = Path(config.warehouse_dir)
    recipe = provenance.read_recipe(warehouse)
    if recipe is None:
        from geoworkflow.store.recipe import default_recipe
        recipe = default_recipe()
    catalog = _catalog_for(config, grid_stats_dir=grid_stats_dir)
    rec = catalog.resolve(iso3, name, aggid=aggid)
    _set_globals(config, catalog, recipe.metrics, overwrite=overwrite, reuse=False)
    status = build_city_worker(rec.aggid)[1]
    provenance.append_op(warehouse, "add-city", aggid=rec.aggid,
                         iso3=rec.iso3, name=rec.name, status=status)
    return status


def remove_city(config: HexDBConfig, iso3: str, name: Optional[str] = None, *,
                aggid: Optional[int] = None):
    """Delete a city's partition; update provenance."""
    warehouse = Path(config.warehouse_dir)
    catalog = _catalog_for(config)
    rec = catalog.resolve(iso3, name, aggid=aggid)
    if rec.partition.exists():
        rec.partition.unlink()
        try:
            rec.partition.parent.rmdir()
        except OSError:
            pass
        status = "removed"
    else:
        status = "absent"
    provenance.append_op(warehouse, "remove-city", aggid=rec.aggid,
                         iso3=rec.iso3, name=rec.name, status=status)
    return status


def add_metric(config: HexDBConfig, metric: MetricSpec, *, max_workers: int = 8):
    """Compute a new metric for all built cities; append to the live recipe."""
    warehouse = Path(config.warehouse_dir)
    recipe = provenance.read_recipe(warehouse)
    if recipe is None:
        raise GeoWorkflowError("no live recipe; build the database first")
    if any(m.name == metric.name for m in recipe.metrics):
        raise GeoWorkflowError(f"metric '{metric.name}' already in the recipe")

    catalog = _catalog_for(config)
    built = [int(p.parent.name.split("=")[1])
             for p in warehouse.glob("ISO3=*/aggid=*/part.parquet")]
    _set_globals(config, catalog, [metric], overwrite=False, reuse=False)
    res = parallel_map(add_metric_worker, built, max_workers=max_workers,
                       label="cities", progress_every=200)

    recipe.metrics.append(metric)
    provenance.write_recipe(warehouse, recipe)
    provenance.append_op(warehouse, "add-metric", metric=metric.name,
                         cities=len(built), built=res.done, failed=res.failed)
    return res


def remove_metric(config: HexDBConfig, variable: str):
    """Drop a variable's rows from every partition; update the live recipe."""
    warehouse = Path(config.warehouse_dir)
    n = 0
    for part in warehouse.glob("ISO3=*/aggid=*/part.parquet"):
        df = pd.read_parquet(part)
        keep = df[df["variable"] != variable]
        if len(keep) != len(df):
            _write_atomic(keep, part)
            n += 1
    recipe = provenance.read_recipe(warehouse)
    if recipe is not None:
        recipe.metrics = [m for m in recipe.metrics if m.name != variable]
        provenance.write_recipe(warehouse, recipe)
    provenance.append_op(warehouse, "remove-metric", variable=variable, partitions=n)
    return n


def _fetch_missing(recipe: HexDBRecipe, config: HexDBConfig, records) -> None:
    """Download any metric rasters not already present (Earth Engine).

    Uses the existing ``export_gee_rasters`` per metric in grid_dir mode
    (skip-existing), so present inputs are untouched. Only runs for metrics that
    declare a ``source``.
    """
    from geoworkflow.processors.extraction import export_gee_rasters
    for m in recipe.metrics:
        if m.source is None:
            continue
        s = m.source
        out_dir = Path(config.global_dir) if m.scope == "global" else Path(config.city_dir)
        export_gee_rasters(
            source=s.ee_asset, bands=s.bands, band_tags=s.band_tags,
            cadence=s.cadence, start=recipe.time.start, end=recipe.time.end,
            composite=s.composite, qc_band=s.qc_band, qc_bit_mask=s.qc_bit_mask,
            qc_max=s.qc_max, scale_factor=s.scale_factor,
            grid_dir=str(config.hexagglo_dir), grid_pattern="*/*_hex.gpkg",
            output_dir=str(out_dir), dataset=m.folder(),
            filename_template=s.filename_template, static_label=s.static_label,
            scale_m=s.scale_m, output_crs=s.output_crs,
        )
