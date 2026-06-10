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

        # Compress data variables (and store continuous values as float32).
        # Cubes exist to be smaller than the rasters they summarize; uncompressed
        # float64 was ~6-10x larger than needed (Nairobi: 98 MB -> ~10 MB).
        encoding = None
        if cfg.compress:
            encoding = {}
            for name, da in ds.data_vars.items():
                enc: Dict[str, Any] = {"zlib": True, "complevel": 4}
                if da.dtype == np.float64:
                    enc["dtype"] = "float32"
                encoding[name] = enc

        self.log_processing_step(f"Writing NetCDF -> {cfg.output_file}")
        ds.to_netcdf(cfg.output_file, encoding=encoding)

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
            # The statistics actually included (cfg.statistics=None means all).
            "statistics": sorted({
                str(ds[v].attrs.get("statistic")) for v in ds.data_vars
            }),
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

    @staticmethod
    def _is_annual(times: "pd.Series") -> bool:
        """Auto-detect annual cadence: >=2 distinct years, all on Jan 1, one per year.

        A multi-year Jan-1 series (e.g. land cover 2020 & 2021) is unambiguously
        yearly. A *single* Jan timestamp is ambiguous (it could be January of a
        monthly series), so single-year annual products must be declared via
        ``annual_variables`` rather than inferred.
        """
        t = pd.to_datetime(times).dropna()
        if t.empty:
            return False
        all_jan1 = bool(((t.dt.month == 1) & (t.dt.day == 1)).all())
        one_per_year = t.nunique() == t.dt.year.nunique()
        return all_jan1 and one_per_year and t.dt.year.nunique() >= 2

    def _build_dataset(self) -> "xr.Dataset":
        from collections import defaultdict
        from geoworkflow.core.legends import DEFAULT_LEGENDS

        cfg = self.netcdf_config
        gid = cfg.grid_id_column
        df = self.table
        grid = self.grid_gdf

        # Cell order and per-cell coordinates come from the grid (authoritative
        # geometry), so the cube aligns with the hexagons even if the table is
        # missing rows for some cells.
        cell_ids = grid[gid].astype(str).tolist()
        # Centroids in a projected CRS (UTM) then reproject to lon/lat, avoiding the
        # inaccurate-geographic-centroid warning.
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

        # Present (variable, statistic) pairs, filtered to requested statistics,
        # grouped per variable.
        stats_filter = set(cfg.statistics) if cfg.statistics else None
        all_pairs = sorted(set(zip(df["variable"], df["statistic"])))
        var_stats: Dict[str, List[str]] = defaultdict(list)
        for v, s in all_pairs:
            if stats_filter is None or s in stats_filter:
                var_stats[v].append(s)
        if not var_stats:
            raise ProcessingError(
                f"No (variable, statistic) combinations found for {cfg.statistics}. "
                f"Table has: {all_pairs}"
            )

        # Classify each variable's cadence and build the (up to two) time axes:
        # monthly variables share `time`; annual variables share `year`.
        annual_override = set(cfg.annual_variables or [])
        cadence: Dict[str, str] = {}
        monthly_times: set = set()
        annual_years: set = set()
        for v in var_stats:
            vtimes = df.loc[df["variable"] == v, "time"]
            t = pd.to_datetime(vtimes).dropna()
            if t.empty:
                cadence[v] = "static"          # no time at all -> (cell,)
            elif (v in annual_override) or self._is_annual(vtimes):
                cadence[v] = "annual"
                annual_years.update(int(y) for y in t.dt.year.unique())
            else:
                cadence[v] = "monthly"
                monthly_times.update(t.unique())

        time_index = np.sort(np.array(list(monthly_times))) if monthly_times else None
        year_index = np.sort(np.array(list(annual_years), dtype=int)) if annual_years else None
        if time_index is not None:
            coords["time"] = time_index
        if year_index is not None:
            coords["year"] = year_index

        # One data var per (variable, statistic). A variable with a single statistic
        # keeps its bare name (PM25); with several it is suffixed (landcover_variety).
        data_vars: Dict[str, Any] = {}
        var_meta: Dict[str, Dict[str, Any]] = {}
        for v, slist in var_stats.items():
            multi = len(slist) > 1
            for s in slist:
                sub = df[(df["variable"] == v) & (df["statistic"] == s)]
                name = f"{v}_{s}" if multi else str(v)
                if cadence[v] == "static":
                    series = sub.set_index(gid)["value"].reindex(cell_ids)
                    data_vars[name] = (("cell",), series.to_numpy())
                elif cadence[v] == "annual":
                    sub = sub.assign(_year=pd.to_datetime(sub["time"]).dt.year)
                    pivot = sub.pivot_table(index=gid, columns="_year", values="value",
                                            aggfunc="first").reindex(index=cell_ids, columns=year_index)
                    data_vars[name] = (("cell", "year"), pivot.to_numpy())
                else:
                    pivot = sub.pivot_table(index=gid, columns="time", values="value",
                                            aggfunc="first").reindex(index=cell_ids, columns=time_index)
                    data_vars[name] = (("cell", "time"), pivot.to_numpy())
                units = sub["units"].dropna().unique() if "units" in sub.columns else []
                var_meta[name] = {"variable": v, "statistic": s,
                                  "units": units[0] if len(units) else None}

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Per-variable CF attributes (units, statistic, long_name, categorical flags).
        legends = {**DEFAULT_LEGENDS, **(cfg.legends or {})}
        for name, meta in var_meta.items():
            v, s = meta["variable"], meta["statistic"]
            if meta["units"] is not None:
                ds[name].attrs["units"] = str(meta["units"])
            ds[name].attrs["statistic"] = s
            if v in cfg.long_names:
                ds[name].attrs["long_name"] = cfg.long_names[v]
            # Class codes -> CF flag_values/flag_meanings, on the class-returning stat.
            if s in ("majority", "minority", "mode") and v in legends:
                leg = {int(k): str(val) for k, val in legends[v].items()}
                codes = sorted(leg)
                ds[name].attrs["flag_values"] = np.array(codes, dtype="int32")
                ds[name].attrs["flag_meanings"] = " ".join(leg[c] for c in codes)

        ds.attrs["title"] = cfg.title or Path(cfg.grid_file).stem
        ds.attrs["source"] = "geoworkflow GridToNetCDFProcessor"
        ds.attrs["grid_file"] = str(cfg.grid_file)
        ds.attrs["created"] = pd.Timestamp.now().isoformat()
        ds["cell"].attrs["long_name"] = "hexagon GridID"
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lon"].attrs["units"] = "degrees_east"
        if year_index is not None:
            ds["year"].attrs["long_name"] = "year"
        return ds


def build_grid_netcdf(
    input_file: Union[str, Path],
    grid_file: Union[str, Path],
    output_file: Union[str, Path],
    statistics: Optional[List[str]] = None,
    **kwargs: Any,
) -> ProcessingResult:
    """Convenience wrapper to build a hex cube in one call.

    ``statistics=None`` (the default) includes every statistic present in the
    table — nothing is dropped silently. Pass a list to keep only those.
    """
    config = GridNetCDFConfig(
        input_file=Path(input_file),
        grid_file=Path(grid_file),
        output_file=Path(output_file),
        statistics=statistics,
        **kwargs,
    )
    return GridToNetCDFProcessor(config).process()
