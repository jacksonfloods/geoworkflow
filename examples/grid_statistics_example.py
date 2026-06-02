"""
Pipeline 1 example: summarize rasters into a hex grid as a tidy table.

Computes per-hexagon zonal statistics (coverage-weighted) for one agglomeration
from a mix of raster types -- the monthly PM2.5 netCDF and monthly ODIAC CO2
GeoTIFFs -- and writes a single tidy Parquet table:

    GridID                  variable   time        statistic      value   units
    SSA_HQ+014562_R+005142  PM25       2019-06-01  weighted_mean   12.0    ug/m3
    SSA_HQ+014562_R+005142  odiac_co2  2021-08-01  weighted_mean    3.5    gC/m2/day

Variable / time / CRS / units for each file are resolved from the dataset
registry (geoworkflow/config/raster_datasets.json plus any data/raster_datasets.json
you add). Outputs go to ../data/ per the project convention.

Run from inside the repo:
    python examples/grid_statistics_example.py
"""

from pathlib import Path

from geoworkflow.schemas.config_models import GridStatisticsConfig
from geoworkflow.processors.integration.grid_statistics import GridStatisticsProcessor

# --- Inputs -----------------------------------------------------------------
# The hex grid for one agglomeration (one file == one "shared space").
GRID_FILE = Path("../data/grids/dar_es_salaam_hex.geojson")

# Raster inputs: files and/or directories, GeoTIFF and/or netCDF. Directories
# are searched recursively. Here, one year of monthly PM2.5 and one year of
# monthly ODIAC -- the registry parses the month out of each filename.
RASTER_INPUTS = [
    Path("../data/global/PM25/2019"),     # 12 monthly netCDF files
    Path("../data/global/odiac/2021"),    # 12 monthly GeoTIFFs
]

# Tidy Parquet output (the canonical input to Pipeline 2, the NetCDF builder).
OUTPUT_FILE = Path("../data/grid_stats/dar_es_salaam_grid_stats.parquet")


def main() -> None:
    config = GridStatisticsConfig(
        grid_file=GRID_FILE,
        raster_inputs=RASTER_INPUTS,
        output_file=OUTPUT_FILE,
        # Add any statistic from geoworkflow.core.statistics, or percentile_<n>.
        statistics=["weighted_mean", "max"],
        # dataset_registry=Path("../data/raster_datasets.json"),  # optional user entries
        skip_existing=False,
    )

    processor = GridStatisticsProcessor(config)
    result = processor.process()

    print(f"success: {result.success}")
    print(f"message: {result.message}")
    if result.metadata:
        print(f"variables: {result.metadata.get('variables')}")
        print(f"rows:      {result.metadata.get('rows')}")
        print(f"output:    {result.metadata.get('output_file')}")


if __name__ == "__main__":
    main()
