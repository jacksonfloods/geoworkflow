"""Unit tests for safe pruning of per-city raster intermediates."""

import numpy as np
import pytest

try:
    import xarray as xr
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

pytestmark = pytest.mark.skipif(not HAS_LIBS, reason="xarray not available")

if HAS_LIBS:
    from geoworkflow.utils.intermediate_cleanup import prune_raster_intermediates


def _make_cube(path, n_cell=3, n_vars=2):
    ds = xr.Dataset(
        {f"v{i}": ("cell", np.arange(n_cell, dtype=float)) for i in range(n_vars)},
        coords={"cell": [f"h{i}" for i in range(n_cell)]},
    )
    ds.to_netcdf(path)
    return path


def _make_tifs(d, n=5):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"clip_{i}.tif").write_bytes(b"x" * 100)
    return d


class TestPrune:
    def test_dry_run_reports_but_keeps(self, temp_dir):
        cube = _make_cube(temp_dir / "c.nc")
        d = _make_tifs(temp_dir / "clips")
        res = prune_raster_intermediates(cube, [d])  # dry_run defaults True
        assert res["files"] == 5 and res["deleted"] == 0 and res["dry_run"] is True
        assert len(list(d.glob("*.tif"))) == 5  # nothing removed

    def test_delete_removes_files(self, temp_dir):
        cube = _make_cube(temp_dir / "c.nc")
        d = _make_tifs(temp_dir / "clips")
        res = prune_raster_intermediates(cube, [d], dry_run=False)
        assert res["deleted"] == 5
        assert len(list(d.glob("*.tif"))) == 0

    def test_missing_cube_refuses(self, temp_dir):
        d = _make_tifs(temp_dir / "clips")
        with pytest.raises(FileNotFoundError):
            prune_raster_intermediates(temp_dir / "nope.nc", [d], dry_run=False)
        assert len(list(d.glob("*.tif"))) == 5  # untouched

    def test_too_few_vars_refuses(self, temp_dir):
        cube = _make_cube(temp_dir / "c.nc", n_vars=1)
        d = _make_tifs(temp_dir / "clips")
        with pytest.raises(ValueError):
            prune_raster_intermediates(cube, [d], min_vars=2, dry_run=False)
        assert len(list(d.glob("*.tif"))) == 5
