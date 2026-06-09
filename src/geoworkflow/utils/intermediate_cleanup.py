"""
Safe pruning of per-city raster intermediates after a cube is built.

The point of the cube workflow is that the NetCDF cube (a few MB) is the durable
product, while the per-city GeoTIFF/netCDF clips it was computed from (hundreds of
files per city) are disposable. This module deletes those clips -- but only after
verifying the cube is real and complete, and it is a **dry run by default** so it
never removes anything unless explicitly asked to.

Only ever pass *per-city* clip directories (e.g. data/city/<ISO3>/mod11a1_lst).
Never pass shared/global source directories (data/global/...): those feed every
city and must be preserved.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict, Any
import logging

logger = logging.getLogger("geoworkflow")

_RASTER_SUFFIXES = {".tif", ".tiff", ".nc"}


def prune_raster_intermediates(
    cube_file: Union[str, Path],
    tif_dirs: Iterable[Union[str, Path]],
    *,
    grid_file: Optional[Union[str, Path]] = None,
    min_vars: int = 1,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Delete per-city raster clips, but only after validating the cube.

    The cube is opened and checked first; if anything looks wrong (missing file,
    too few variables, cell count disagreeing with the grid) the function raises and
    deletes nothing. With ``dry_run=True`` (the default) it reports what *would* be
    removed without touching the filesystem.

    Args:
        cube_file: NetCDF cube that must exist and open before any deletion.
        tif_dirs: directories of per-city raster clips to prune. Must NOT be shared
            global source directories.
        grid_file: optional grid; if given, the cube's ``cell`` count must equal the
            grid's feature count (a guard that the cube really covers this city).
        min_vars: the cube must contain at least this many data variables.
        dry_run: if True (default) nothing is deleted; only the plan is returned.

    Returns:
        Summary dict: ``verified``, ``files`` (found), ``bytes``, ``deleted``,
        ``dry_run``, ``cube_vars``.
    """
    import xarray as xr

    cube_file = Path(cube_file)
    if not cube_file.exists():
        raise FileNotFoundError(f"Cube not found; refusing to prune intermediates: {cube_file}")

    # 1) Validate the cube before deleting anything.
    with xr.open_dataset(cube_file) as ds:
        n_vars = len(ds.data_vars)
        n_cell = int(ds.sizes.get("cell", 0))
    if n_vars < min_vars:
        raise ValueError(
            f"Cube {cube_file.name} has {n_vars} variable(s) (< {min_vars}); refusing to prune."
        )
    if grid_file is not None:
        import geopandas as gpd
        n_grid = len(gpd.read_file(grid_file))
        if n_cell != n_grid:
            raise ValueError(
                f"Cube cells ({n_cell}) != grid features ({n_grid}) for "
                f"{cube_file.name}; refusing to prune."
            )

    # 2) Gather raster clips under the given (per-city) directories.
    files: List[Path] = []
    for d in tif_dirs:
        d = Path(d)
        if not d.exists():
            continue
        files += [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _RASTER_SUFFIXES]
    total_bytes = sum(p.stat().st_size for p in files)

    deleted = 0
    if not dry_run:
        for p in files:
            try:
                p.unlink()
                deleted += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", p, exc)

    logger.info(
        "%s %d intermediate file(s) (%.1f MB) [cube ok: %d vars, %d cells]",
        "Would delete" if dry_run else "Deleted", len(files), total_bytes / 1e6, n_vars, n_cell,
    )
    return {
        "verified": True,
        "files": len(files),
        "bytes": total_bytes,
        "deleted": deleted,
        "dry_run": dry_run,
        "cube_vars": n_vars,
    }
