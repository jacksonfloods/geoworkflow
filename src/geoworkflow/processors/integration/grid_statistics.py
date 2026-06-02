"""
Grid statistics processor (Pipeline 1).

Summarizes any number of rasters (GeoTIFF and netCDF) into per-hexagon zonal
statistics and writes a tidy long-format Parquet table::

    GridID                  variable   time        statistic      value   units
    SSA_HQ+014562_R+005142  PM25       2019-01-01  weighted_mean   18.4    ug/m3
    SSA_HQ+014562_R+005142  odiac_co2  2021-01-01  weighted_mean    3.1    gC/m2/day

It composes the three foundations:
  * the raster source layer (AOI-clipped single-band slices),
  * the dataset registry (variable / time / CRS per file),
  * the exactextract zonal engine (coverage-weighted statistics).

The tidy output is the canonical intermediate consumed by Pipeline 2
(the NetCDF builder).
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

try:
    import geopandas as gpd
    import pandas as pd
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.base import ProcessingResult
from geoworkflow.core.exceptions import ProcessingError
from geoworkflow.core.statistics import resolve_statistics
from geoworkflow.schemas.config_models import GridStatisticsConfig
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory

# Tidy output column order.
TIDY_COLUMNS = ["variable", "time", "statistic", "value", "units"]


class GridStatisticsProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """Compute per-hex zonal statistics for a set of rasters into a tidy table."""

    def __init__(
        self,
        config: Union[GridStatisticsConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, GridStatisticsConfig):
            config_dict = config.model_dump(mode="json")
            self.grid_stats_config = config
        else:
            config_dict = config
            self.grid_stats_config = GridStatisticsConfig(**config_dict)

        super().__init__(config_dict, logger)

        self.grid_gdf: Optional["gpd.GeoDataFrame"] = None
        self.registry = None

    # ------------------------------------------------------------------
    # Template method hooks
    # ------------------------------------------------------------------

    def _get_path_config_keys(self) -> List[str]:
        return ["grid_file"]

    def _validate_custom_inputs(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "info": {}}
        cfg = self.grid_stats_config

        if not HAS_GEOSPATIAL_LIBS:
            result["valid"] = False
            result["errors"].append("geopandas and pandas are required.")
            return result

        if not cfg.grid_file.exists():
            result["valid"] = False
            result["errors"].append(f"Grid file does not exist: {cfg.grid_file}")

        if not cfg.raster_inputs:
            result["valid"] = False
            result["errors"].append("No raster_inputs provided.")
        else:
            missing = [str(p) for p in cfg.raster_inputs if not Path(p).exists()]
            if missing:
                result["valid"] = False
                result["errors"].append(f"Raster inputs do not exist: {missing}")

        # Fail fast on bad statistic names.
        try:
            resolve_statistics(cfg.statistics)
        except KeyError as exc:
            result["valid"] = False
            result["errors"].append(f"Unknown statistic: {exc}")

        if cfg.dataset_registry is not None and not Path(cfg.dataset_registry).exists():
            result["valid"] = False
            result["errors"].append(
                f"dataset_registry does not exist: {cfg.dataset_registry}"
            )

        try:
            ensure_directory(cfg.output_file.parent)
            result["info"]["output_dir"] = str(cfg.output_file.parent)
        except Exception as exc:
            result["valid"] = False
            result["errors"].append(f"Cannot create output directory: {exc}")

        return result

    def _setup_custom_processing(self) -> Dict[str, Any]:
        setup_info: Dict[str, Any] = {}
        cfg = self.grid_stats_config

        # Defer heavy imports to setup so a missing optional stack fails cleanly.
        from geoworkflow.core.dataset_registry import load_dataset_registry

        self.log_processing_step("Loading hex grid")
        self.grid_gdf = gpd.read_file(cfg.grid_file)
        if self.grid_gdf.crs is None:
            self.logger.warning("Grid has no CRS; assuming EPSG:4326")
            self.grid_gdf.set_crs("EPSG:4326", inplace=True)
        if cfg.grid_id_column not in self.grid_gdf.columns:
            raise ProcessingError(
                f"Grid id column '{cfg.grid_id_column}' not in grid "
                f"({list(self.grid_gdf.columns)})"
            )
        setup_info["grid_features"] = len(self.grid_gdf)
        self.add_metric("grid_features", len(self.grid_gdf))

        self.registry = load_dataset_registry(cfg.dataset_registry)
        setup_info["registry_datasets"] = self.registry.names()
        return setup_info

    def process_data(self) -> ProcessingResult:
        result = ProcessingResult(success=True)
        cfg = self.grid_stats_config

        # Import here so the module imports even without the raster stack.
        from geoworkflow.utils.raster_source import open_raster_slices
        from geoworkflow.utils.zonal_utils import compute_zonal_statistics

        if cfg.skip_existing and cfg.output_file.exists():
            result.skipped_count = 1
            result.message = f"Skipping - output exists: {cfg.output_file}"
            result.add_output_path(cfg.output_file)
            return result

        gid = cfg.grid_id_column
        frames: List["pd.DataFrame"] = []
        slices_done = 0
        variables_seen = set()

        slices = open_raster_slices(
            [Path(p) for p in cfg.raster_inputs],
            aoi=self.grid_gdf,
            registry=self.registry,
            dataset=cfg.dataset,
            default_crs=cfg.default_crs,
            recursive=cfg.recursive,
        )

        for sl in track_progress(slices, description="Summarizing rasters", quiet=False):
            if sl.is_empty:
                self.logger.warning("Empty slice for %s (%s); skipping.",
                                    sl.variable, sl.source_path)
                continue
            with sl.open() as ds:
                wide = compute_zonal_statistics(
                    ds, self.grid_gdf, cfg.statistics, include_cols=[gid]
                )
            tidy = wide.melt(
                id_vars=[gid], value_vars=cfg.statistics,
                var_name="statistic", value_name="value",
            )
            tidy["variable"] = sl.variable
            # Build time/units as explicit-dtype columns (datetime64 + object) so
            # that slices with no time/units don't produce ambiguous all-NA
            # columns that trip pandas' concat dtype-inference deprecation.
            tidy["time"] = pd.to_datetime(pd.Series([sl.time] * len(tidy), index=tidy.index))
            tidy["units"] = pd.array([sl.units] * len(tidy), dtype="object")
            frames.append(tidy)

            slices_done += 1
            variables_seen.add(sl.variable)
            result.processed_count += 1
            self.update_progress(1, f"{sl.variable} {sl.time}")

        if not frames:
            raise ProcessingError(
                "No raster slices produced any statistics. Check raster_inputs, "
                "the AOI overlap, and the dataset registry."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined[[gid, *TIDY_COLUMNS]]
        combined = combined.sort_values([gid, "variable", "time", "statistic"]) \
                           .reset_index(drop=True)

        self.log_processing_step(f"Writing tidy table -> {cfg.output_file}")
        combined.to_parquet(cfg.output_file, index=False)

        result.add_output_path(cfg.output_file)
        result.message = (
            f"Wrote {len(combined)} rows for {len(variables_seen)} variable(s) "
            f"from {slices_done} slice(s) -> {cfg.output_file}"
        )
        result.metadata = {
            "grid_file": str(cfg.grid_file),
            "output_file": str(cfg.output_file),
            "rows": len(combined),
            "slices": slices_done,
            "variables": sorted(variables_seen),
            "statistics": list(cfg.statistics),
            "grid_features": len(self.grid_gdf),
        }
        self.add_metric("tidy_rows", len(combined))
        return result

    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        cleanup_info: Dict[str, Any] = {}
        if self.grid_gdf is not None:
            del self.grid_gdf
            self.grid_gdf = None
            cleanup_info["grid_gdf_cleared"] = True
        return cleanup_info


def compute_grid_statistics(
    grid_file: Union[str, Path],
    raster_inputs: Union[str, Path, List[Union[str, Path]]],
    output_file: Union[str, Path],
    statistics: Optional[List[str]] = None,
    **kwargs: Any,
) -> ProcessingResult:
    """Convenience wrapper to run the grid-statistics pipeline in one call."""
    if isinstance(raster_inputs, (str, Path)):
        raster_inputs = [raster_inputs]
    config = GridStatisticsConfig(
        grid_file=Path(grid_file),
        raster_inputs=[Path(p) for p in raster_inputs],
        output_file=Path(output_file),
        statistics=statistics or ["weighted_mean"],
        **kwargs,
    )
    return GridStatisticsProcessor(config).process()
