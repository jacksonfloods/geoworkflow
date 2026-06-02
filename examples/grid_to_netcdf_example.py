"""
Pipeline 2 example: build a hexagon (cell, time) NetCDF cube and query it.

Takes the tidy Parquet produced by Pipeline 1 (examples/grid_statistics_example.py)
plus the hex grid geometry, and writes a per-agglomeration cube where:

    dims:   cell (one hexagon), time (months)
    coords: cell == GridID; lat, lon, q, r along cell; time
    vars:   PM25(cell, time), odiac_co2(cell, time)

Run from inside the repo (after running grid_statistics_example.py):
    python examples/grid_to_netcdf_example.py
"""

from pathlib import Path

from geoworkflow.schemas.config_models import GridNetCDFConfig
from geoworkflow.processors.integration.netcdf_builder import GridToNetCDFProcessor
from geoworkflow.utils import hexcube

TIDY_TABLE = Path("../data/grid_stats/dar_es_salaam_grid_stats.parquet")
GRID_FILE = Path("../data/grids/dar_es_salaam_hex.geojson")
OUTPUT_NC = Path("../data/cubes/dar_es_salaam.nc")


def build() -> None:
    config = GridNetCDFConfig(
        input_file=TIDY_TABLE,
        grid_file=GRID_FILE,
        output_file=OUTPUT_NC,
        statistics=["weighted_mean"],   # single stat -> data vars named by variable
        title="Dar es Salaam",
    )
    result = GridToNetCDFProcessor(config).process()
    print(f"success: {result.success} | {result.message}")


def query_examples() -> None:
    ds = hexcube.open_hex_cube(OUTPUT_NC)

    hexcube.select_month(ds, "PM25", 2019, 6)
    # PM2.5 value per hexagon for June 2019, shape (cell,) -- a map over hexes

    hexcube.select_month(ds, "odiac_co2", 2019, 6)
    # ODIAC CO2 for the same month, shape (cell,)

    hexcube.annual_mean(ds, "PM25", 2019)
    # per-hex mean of all months in 2019, shape (cell,)

    hexcube.month_across_years(ds, "PM25", 6)
    # per-hex mean across every June in the record, shape (cell,)

    # The same things with raw xarray (note the .squeeze for a single month):
    ds["PM25"].sel(time="2019-06").squeeze("time", drop=True)
    # one-month map; without .squeeze you keep a length-1 time dim

    ds["PM25"].sel(time=slice("2019-01", "2019-12")).mean("time")
    # per-hex 2019 mean, shape (cell,)

    ds["PM25"].sel(time=ds.time.dt.month == 6).mean("time")
    # per-hex mean across all Junes, shape (cell,)

    hexcube.to_geodataframe(ds, GRID_FILE, variable="PM25", time="2019-06")
    # GeoDataFrame of hexes + the June 2019 PM2.5 column, ready for .plot(column=...)

    ds.close()


if __name__ == "__main__":
    build()
    if OUTPUT_NC.exists():
        query_examples()
