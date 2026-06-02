"""
End-to-end example: AOI -> hex grid -> per-hex statistics -> NetCDF cube.

Runs all three steps in tandem for one agglomeration (Nairobi). Each step is an
independent processor with its own config, so you can also run them separately
(see grid_statistics_example.py and grid_to_netcdf_example.py); this script just
sequences them and wires each output to the next input.

    1. HexGridProcessor        AOI polygon        -> nairobi_hex.geojson
    2. GridStatisticsProcessor grid + rasters     -> nairobi_grid_stats.parquet
    3. GridToNetCDFProcessor   table + grid       -> nairobi.nc  (cell, time)

Run from inside the repo:
    python examples/full_pipeline_example.py

Outputs go to ../data/ per the project convention. The rasters are resolved
through the dataset registry (geoworkflow/config/raster_datasets.json plus any
data/raster_datasets.json). Verify a new file's parsing first with:
    geoworkflow datasets test <filename>
"""

from pathlib import Path

from geoworkflow.schemas.config_models import (
    HexGridConfig, GridStatisticsConfig, GridNetCDFConfig,
)
from geoworkflow.processors.spatial.hexgrid import HexGridProcessor
from geoworkflow.processors.integration.grid_statistics import GridStatisticsProcessor
from geoworkflow.processors.integration.netcdf_builder import GridToNetCDFProcessor
from geoworkflow.utils import hexcube

# --- Inputs ---------------------------------------------------------------
AOI_FILE = Path("../data/boundaries/nairobi.gpkg")     # one agglomeration polygon
RASTER_INPUTS = [
    Path("../data/global/PM25/2019"),                  # monthly PM2.5 netCDF
    Path("../data/global/odiac/2019"),                 # monthly ODIAC GeoTIFF
]

# --- Outputs (all under ../data/) -----------------------------------------
GRID_FILE = Path("../data/grids/nairobi_hex.geojson")
TIDY_TABLE = Path("../data/grid_stats/nairobi_grid_stats.parquet")
OUTPUT_NC = Path("../data/cubes/nairobi.nc")


def main() -> None:
    # 1. Generate the hex grid clipped to the AOI.
    grid_result = HexGridProcessor(HexGridConfig(
        aoi_file=AOI_FILE,
        output_file=GRID_FILE,
        side_length=150.0,
        skip_existing=True,
    )).process()
    print(f"[1/3] hex grid: {grid_result.success} | {grid_result.message}")

    # 2. Summarize the rasters into the grid (coverage-weighted) -> tidy table.
    stats_result = GridStatisticsProcessor(GridStatisticsConfig(
        grid_file=GRID_FILE,
        raster_inputs=RASTER_INPUTS,
        output_file=TIDY_TABLE,
        statistics=["weighted_mean"],
        skip_existing=True,
    )).process()
    print(f"[2/3] grid stats: {stats_result.success} | {stats_result.message}")

    # 3. Assemble the (cell, time) NetCDF cube.
    nc_result = GridToNetCDFProcessor(GridNetCDFConfig(
        input_file=TIDY_TABLE,
        grid_file=GRID_FILE,
        output_file=OUTPUT_NC,
        title="Nairobi",
        skip_existing=True,
    )).process()
    print(f"[3/3] netcdf: {nc_result.success} | {nc_result.message}")

    if OUTPUT_NC.exists():
        ds = hexcube.open_hex_cube(OUTPUT_NC)
        print("cube dims:", dict(ds.sizes), "| variables:", list(ds.data_vars))
        ds.close()


if __name__ == "__main__":
    main()
