#!/usr/bin/env python
"""Headless full build of the hex database warehouse (data/hexdb).

Recipe-driven, no cubes. Reuses the ~640 already-computed grid_stats parquets and
computes the rest from rasters via exactextract. Resumable: existing partitions
are skipped, so it is safe to stop and re-run.

    GEE_MAX_WORKERS=16 python notebooks/run_hexdb_build.py
"""
import os
from pathlib import Path

os.environ.setdefault("GEOWORKFLOW_QUIET_PROGRESS", "1")

from geoworkflow.schemas.config_models import HexDBConfig
from geoworkflow.store import builder
from geoworkflow.store.recipe import default_recipe

ROOT = Path(__file__).resolve().parents[2]   # …/africa_cities (geoworkflow/notebooks/ -> up 2)
WORKERS = int(os.environ.get("GEE_MAX_WORKERS", "16"))

cfg = HexDBConfig(
    warehouse_dir=ROOT / "data/hexdb",
    hexagglo_dir=ROOT / "data/boundaries/hexagglo",
    manifest_csv=ROOT / "data/config/grid_zone_manifest.csv",
    complexity_csv=ROOT / "data/boundaries/agglomerations_complexity.csv",
    global_dir=ROOT / "data/global",
    city_dir=ROOT / "data/city",
    raster_registry=ROOT / "data/config/raster_datasets.json",
)

if __name__ == "__main__":
    res = builder.build(default_recipe(), cfg, max_workers=WORKERS,
                        reuse=True, grid_stats_dir=ROOT / "data/grid_stats")
    print(f"HEXDB_BUILD_DONE built={res.done} skipped={res.skipped} "
          f"failed={res.failed} in {res.seconds:.0f}s", flush=True)
