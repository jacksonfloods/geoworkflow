"""
UTM zone master list (the grid zone manifest).

For ``utm_local`` hex grids, each city's UTM zone must be **pinned** — decided
once, recorded, and read on every regeneration — so that a future AfricaPolis
boundary edit can't silently flip a city across a 6° zone line and relocate its
whole grid. This module is that record: a CSV keyed by ``Agglomeration_ID``
mapping each city to its grid spec (``utm_epsg``, ``side_length``, ``origin``).

It is the *grid* counterpart to :mod:`geoworkflow.core.dataset_registry` (which
is the *raster* registry) — both are registries that drive the pipelines.

**Populate strategy (Option A):** the zone is computed once with
``geopandas.estimate_utm_crs`` (a well-tested library, not hand-rolled zone math)
and frozen in the manifest. ``estimate_utm_crs`` was validated to land on the
lab's morphology-grid zone exactly (Dakar spike: same EPSG, residual 0 m). For a
city that already has a morphology grid, :func:`epsg_from_grid_file` can read its
zone to confirm or override.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

try:
    import geopandas as gpd  # noqa: F401
    HAS_GPD = True
except ImportError:  # pragma: no cover
    HAS_GPD = False

MANIFEST_COLUMNS = [
    "Agglomeration_ID", "ISO3", "name", "utm_epsg",
    "side_length", "origin_x", "origin_y", "source",
]

# Conventional location: data/config/ holds the registries that drive the
# pipelines (this manifest + the raster dataset registry), kept apart from data.
DEFAULT_MANIFEST = Path("data/config/grid_zone_manifest.csv")

ORIGIN = (-3_000_000.0, -2_000_000.0)


def load_zone_manifest(path: Union[str, Path] = DEFAULT_MANIFEST) -> Dict[int, dict]:
    """Load the manifest as ``{Agglomeration_ID: row}``; empty dict if absent."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {int(row["Agglomeration_ID"]): row for row in csv.DictReader(f)}


def epsg_from_grid_file(path: Union[str, Path]) -> Optional[int]:
    """Read the EPSG code of an existing grid file (e.g. a morphology .shp)."""
    crs = gpd.read_file(path, rows=1).crs
    return crs.to_epsg() if crs is not None else None


def _append_row(path: Path, row: dict) -> None:
    path = Path(path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def resolve_utm_epsg(
    agglomeration_id: int,
    aoi_gdf: "gpd.GeoDataFrame",
    *,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST,
    iso3: str = "",
    name: str = "",
    side_length: float = 250.0,
    origin: Tuple[float, float] = ORIGIN,
    override_epsg: Optional[int] = None,
    append: bool = True,
) -> int:
    """Return the **pinned** UTM EPSG for a city, recording it on first use.

    If the agglomeration is already in the manifest, its recorded ``utm_epsg`` is
    returned unchanged — so the zone is immune to later AOI/boundary changes. If
    not, the zone is determined (``override_epsg`` if given, else
    ``estimate_utm_crs``), appended to the manifest, and returned.

    Args:
        agglomeration_id: stable AfricaPolis id (the manifest key).
        aoi_gdf: the city's AOI (used only when computing a new zone).
        override_epsg: force a specific zone (e.g. read from a morphology grid via
            :func:`epsg_from_grid_file`); recorded with source ``morphology_grid``.
        append: write the resolved zone back to the manifest (default True).
    """
    manifest = load_zone_manifest(manifest_path)
    if agglomeration_id in manifest:
        return int(manifest[agglomeration_id]["utm_epsg"])

    epsg = int(override_epsg if override_epsg is not None
               else aoi_gdf.estimate_utm_crs().to_epsg())
    if append:
        _append_row(Path(manifest_path), {
            "Agglomeration_ID": agglomeration_id,
            "ISO3": iso3,
            "name": name,
            "utm_epsg": epsg,
            "side_length": side_length,
            "origin_x": origin[0],
            "origin_y": origin[1],
            "source": "morphology_grid" if override_epsg is not None else "estimate_utm_crs",
        })
    return epsg
