"""
Grid-to-NetCDF processor (Pipeline 2).

Takes the tidy Parquet table from Pipeline 1 plus the hex grid geometry and
builds a per-agglomeration NetCDF "data cube" with dimensions ``(cell, time)``:

    dims:   cell(N hexes), time(M months)
    coords: cell == GridID; lat, lon, q, r along cell; time (datetime64)
    vars:   one per raster variable, e.g. PM25(cell, time), odiac_co2(cell, time)

The cube is xarray-native, so the usual queries work::

    ds["PM25"].sel(time="2021-06")                       # one month, per hex
    ds["PM25"].sel(time=slice("2021-01", "2021-12")).mean("time")   # 2021 mean
    ds["PM25"].sel(time=ds.time.dt.month == 6).mean("time")         # all Junes
    ds["PM25"].mean("cell")                              # city-wide per month

Helpers for these (and joining values back to hex geometry for maps) live in
:mod:`geoworkflow.utils.hexcube`.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.base import ProcessingResult
from geoworkflow.core.exceptions import ProcessingError
from geoworkflow.schemas.config_models import GridNetCDFConfig
from geoworkflow.utils.resource_utils import ensure_directory


class GridToNetCDFProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """Build a hexagon (cell, time) NetCDF cube from a tidy statistics table."""

    def __init__(
        self,
        config: Union[GridNetCDFConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, GridNetCDFConfig):
            config_dict = config.model_dump(mode="json")
            self.netcdf_config = config
        else:
            config_dict = config
            self.netcdf_config = GridNetCDFConfig(**config_dict)

        super().__init__(config_dict, logger)
        self.table: Optional["pd.DataFrame"] = None
        self.grid_gdf: Optional["gpd.GeoDataFrame"] = None

    # ------------------------------------------------------------------
    # Template method hooks
    # ------------------------------------------------------------------

    def _get_path_config_keys(self) -> List[str]:
        return ["input_file", "grid_file"]

    def _validate_custom_inputs(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "info": {}}
        cfg = self.netcdf_config

        if not HAS_LIBS:
            result["valid"] = False
            result["errors"].append("geopandas, pandas, xarray and netCDF4 are required.")
            return result

        if not cfg.input_file.exists():
            result["valid"] = False
            result["errors"].append(f"Input table does not exist: {cfg.input_file}")
        if not cfg.grid_file.exists():
            result["valid"] = False
            result["errors"].append(f"Grid file does not exist: {cfg.grid_file}")

        try:
            ensure_directory(cfg.output_file.parent)
        except Exception as exc:
            result["valid"] = False
            result["errors"].append(f"Cannot create output directory: {exc}")

        return result

    def _setup_custom_processing(self) -> Dict[str, Any]:
        cfg = self.netcdf_config
        self.log_processing_step("Loading tidy table and grid")
        self.table = pd.read_parquet(cfg.input_file)
        expected = {cfg.grid_id_column, "variable", "time", "statistic", "value"}
        missing = expected - set(self.table.columns)
        if missing:
            raise ProcessingError(f"Tidy table missing columns: {sorted(missing)}")

        self.grid_gdf = gpd.read_file(cfg.grid_file)
        if cfg.grid_id_column not in self.grid_gdf.columns:
            raise ProcessingError(
                f"Grid id column '{cfg.grid_id_column}' not in grid file."
            )
        if self.grid_gdf.crs is None:
            self.logger.warning("Grid has no CRS; assuming EPSG:4326")
            self.grid_gdf.set_crs("EPSG:4326", inplace=True)
        return {"table_rows": len(self.table), "grid_features": len(self.grid_gdf)}

    def process_data(self) -> ProcessingResult:
        result = ProcessingResult(success=True)
        cfg = self.netcdf_config

        if cfg.skip_existing and cfg.output_file.exists():
            result.skipped_count = 1
            result.message = f"Skipping - output exists: {cfg.output_file}"
            result.add_output_path(cfg.output_file)
            return result

        ds = self._build_dataset()

        self.log_processing_step(f"Writing NetCDF -> {cfg.output_file}")
        ds.to_netcdf(cfg.output_file)

        result.add_output_path(cfg.output_file)
        result.processed_count = len(ds.data_vars)
        n_time = ds.sizes.get("time", 0)
        result.message = (
            f"Wrote cube with {len(ds.data_vars)} variable(s), "
            f"{ds.sizes['cell']} cells, {n_time} time step(s) -> {cfg.output_file}"
        )
        result.metadata = {
            "output_file": str(cfg.output_file),
            "data_vars": list(ds.data_vars),
            "n_cells": int(ds.sizes["cell"]),
            "n_time": int(n_time),
            "statistics": list(cfg.statistics),
        }
        return result

    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for attr in ("table", "grid_gdf"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
                info[f"{attr}_cleared"] = True
        return info

    # ------------------------------------------------------------------
    # Cube construction
    # ------------------------------------------------------------------

    def _build_dataset(self) -> "xr.Dataset":
        cfg = self.netcdf_config
        gid = cfg.grid_id_column
        df = self.table
        grid = self.grid_gdf

        # Cell order and per-cell coordinates come from the grid (authoritative
        # geometry), so the cube aligns with the hexagons even if the table is
        # missing rows for some cells.
        cell_ids = grid[gid].astype(str).tolist()

        # Compute centroids in a projected CRS (UTM) then reproject to lon/lat,
        # avoiding the inaccurate-geographic-centroid warning.
        projected = grid.to_crs(grid.estimate_utm_crs())
        cent = gpd.GeoSeries(projected.geometry.centroid, crs=projected.crs).to_crs("EPSG:4326")
        coords: Dict[str, Any] = {
            "cell": cell_ids,
            "GridID": ("cell", cell_ids),
            "lat": ("cell", cent.y.to_numpy()),
            "lon": ("cell", cent.x.to_numpy()),
        }
        for axial in ("q", "r"):
            if axial in grid.columns:
                coords[axial] = ("cell", grid[axial].to_numpy())

        # Time axis from the non-null times in the table.
        times = pd.to_datetime(df["time"]).dropna()
        has_time = not times.empty
        time_index = np.sort(times.unique()) if has_time else None
        if has_time:
            coords["time"] = time_index

        stats = list(cfg.statistics)
        multi_stat = len(stats) > 1
        available = set(zip(df["variable"], df["statistic"]))

        data_vars: Dict[str, Any] = {}
        units_by_var: Dict[str, Any] = {}

        for variable in sorted(df["variable"].unique()):
            for statistic in stats:
                if (variable, statistic) not in available:
                    continue
                sub = df[(df["variable"] == variable) & (df["statistic"] == statistic)]
                name = f"{variable}_{statistic}" if multi_stat else str(variable)
                if has_time:
                    pivot = sub.pivot_table(
                        index=gid, columns="time", values="value", aggfunc="first"
                    ).reindex(index=cell_ids, columns=time_index)
                    data_vars[name] = (("cell", "time"), pivot.to_numpy())
                else:
                    series = sub.set_index(gid)["value"].reindex(cell_ids)
                    data_vars[name] = (("cell",), series.to_numpy())

                units = sub["units"].dropna().unique() if "units" in sub.columns else []
                units_by_var[name] = units[0] if len(units) else None

        if not data_vars:
            raise ProcessingError(
                f"No (variable, statistic) combinations found for {stats}. "
                f"Table has: {sorted(available)}"
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        for name, units in units_by_var.items():
            if units is not None:
                ds[name].attrs["units"] = str(units)
            ds[name].attrs["statistic"] = (
                name.split("_")[-1] if multi_stat else stats[0]
            )

        ds.attrs["title"] = cfg.title or Path(cfg.grid_file).stem
        ds.attrs["source"] = "geoworkflow GridToNetCDFProcessor"
        ds.attrs["grid_file"] = str(cfg.grid_file)
        ds.attrs["created"] = pd.Timestamp.now().isoformat()
        ds["cell"].attrs["long_name"] = "hexagon GridID"
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lon"].attrs["units"] = "degrees_east"
        return ds


def build_grid_netcdf(
    input_file: Union[str, Path],
    grid_file: Union[str, Path],
    output_file: Union[str, Path],
    statistics: Optional[List[str]] = None,
    **kwargs: Any,
) -> ProcessingResult:
    """Convenience wrapper to build a hex cube in one call."""
    config = GridNetCDFConfig(
        input_file=Path(input_file),
        grid_file=Path(grid_file),
        output_file=Path(output_file),
        statistics=statistics or ["weighted_mean"],
        **kwargs,
    )
    return GridToNetCDFProcessor(config).process()
