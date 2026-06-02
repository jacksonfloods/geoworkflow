# Claude Code Instructions for geoworkflow

## Project Structure

```
AfricaProject/
├── geoworkflow/          # This repo - Python package for geospatial workflows
├── data/                 # All outputs go here (NOT inside geoworkflow/)
│   ├── 00_source/        # Source data files
│   ├── 01_extracted/     # Extracted/processed outputs
│   ├── emissions/        # Emissions data (ODIAC, etc.)
│   └── satellite/        # Satellite imagery outputs
└── ...
```

## Important: Output Directory Convention

**All processor outputs should go to `../data/` (parent directory), not inside geoworkflow.**

Example:
```python
from pathlib import Path

# Correct - outputs to parent/data/
output_dir = Path("../data/satellite")

# Wrong - don't put outputs inside geoworkflow/
# output_dir = Path("./outputs")
```

## Hexagonal raster pipelines (grid statistics → NetCDF)

Two composable pipelines summarize rasters onto hexagonal grids and pack the
result into a per-agglomeration NetCDF cube. They run separately or in tandem
(see `examples/full_pipeline_example.py`).

```
AOI ─HexGridProcessor─▶ hex grid ─GridStatisticsProcessor─▶ tidy Parquet ─GridToNetCDFProcessor─▶ (cell, time) .nc
```

**Pipeline 1 — `GridStatisticsProcessor`** (`processors/integration/grid_statistics.py`):
hex grid + any mix of GeoTIFF/netCDF rasters → a tidy long table
`GridID, variable, time, statistic, value, units`. Statistics are
**coverage-weighted via exactextract** (a hex smaller than a pixel still gets
that pixel's value — `rasterstats` returns null there, so do not use it here).

```python
from geoworkflow.processors.integration.grid_statistics import compute_grid_statistics
compute_grid_statistics(
    grid_file="../data/grids/dar_es_salaam_hex.geojson",
    raster_inputs=["../data/global/PM25/2019", "../data/global/odiac/2019"],
    output_file="../data/grid_stats/dar.parquet",
    statistics=["weighted_mean", "max"],   # or percentile_90, median, stdev, ...
)
```

Add a statistic with `@register_statistic` in `core/statistics.py` (op-backed or
a reducer over per-cell values+coverage).

**Pipeline 2 — `GridToNetCDFProcessor`** (`processors/integration/netcdf_builder.py`):
tidy Parquet + grid geometry → NetCDF with dims `(cell, time)`, `cell` indexed by
`GridID`, `lat`/`lon`/`q`/`r` per-cell coords, one data var per variable.

```python
from geoworkflow.processors.integration.netcdf_builder import build_grid_netcdf
build_grid_netcdf("../data/grid_stats/dar.parquet",
                  "../data/grids/dar_es_salaam_hex.geojson",
                  "../data/cubes/dar.nc", title="Dar es Salaam")
```

Query it with `geoworkflow.utils.hexcube` (`select_month`, `annual_mean`,
`month_across_years`, `monthly_climatology`, `city_mean`, `nearest_cell`,
`to_geodataframe`).

**Dataset registry.** A file's variable / time / CRS / units are resolved from
`geoworkflow/config/raster_datasets.json` plus the user's
`data/raster_datasets.json` (glob `match`, regex/static/coord `time`). Adding a
dataset needs **no code change** — edit the JSON, then verify before a big run:

```bash
geoworkflow datasets list
geoworkflow datasets test odiac2024_1km_excl_intl_2101.tif   # -> odiac, 2021-01
```

## Earth Engine Configuration

This project uses Google Earth Engine. The GCP project ID is:
```
project_id = "africa-cities-jdf277"
```

Before using Earth Engine features, ensure authentication:
```bash
earthengine authenticate
```

## Satellite Imagery Downloader

Download Sentinel-2 RGB imagery for polygons:

```python
from pathlib import Path
from geoworkflow.schemas import SatelliteImageryConfig
from geoworkflow.processors.extraction import SatelliteImageryProcessor

config = SatelliteImageryConfig(
    aoi_file=Path("path/to/area.geojson"),
    output_dir=Path("../data/satellite"),  # Use parent/data/
    start_date="2024-01-01",
    end_date="2024-06-30",
    project_id="africa-cities-jdf277"
)

processor = SatelliteImageryProcessor(config)
result = processor.process()
```

### Batch Mode (AfricaPolis)

```python
config = SatelliteImageryConfig(
    aoi_file="africapolis",
    country=["KEN", "TZA"],  # ISO3 codes
    output_dir=Path("../data/satellite"),
    start_date="2024-01-01",
    end_date="2024-06-30",
    project_id="africa-cities-jdf277"
)
```

## Environment & dependencies

This project runs in a **conda** environment (`environment.yml`, env name `geoworkflow`).
Conda manages the compiled geospatial stack — GDAL, GEOS, PROJ, HDF5, and the
libraries that link them (rasterio, geopandas, shapely, fiona, xarray). **Never
`pip install` those, or anything that links them**, into this env.

### Mixing pip and conda — the rule

`pip install` *into* a conda env is safe only when the package is:

- **pure Python** (e.g. `click`, `pydantic`, `pyyaml`, `s2sphere`, `gcsfs`,
  `earthengine-api`), or
- a **manylinux wheel that vendors its own native libs** (self-contained, with
  hash-mangled SONAMEs, so it can't clash with conda's copies).

It is **risky** when the package is compiled to **dynamically link conda's**
GDAL/GEOS/PROJ/HDF5 (e.g. `pip install rasterio` / `fiona` / `gdal`, or any
`--no-binary` source build). Two copies of the same native library in one process
cause intermittent segfaults. Keep those **conda-managed**: add them to
`environment.yml` and recreate/update the env instead of using pip.

### Audit any pip install into the env

After `pip install <pkg>`, check what its compiled extensions link against:

```bash
ldd "$(python -c 'import <pkg>, os; print(os.path.dirname(<pkg>.__file__))')"/*.so \
    | grep -iE "geos|gdal|proj|hdf5"
```

- Hits resolve to a vendored `*.libs/` dir (or only system libs) → **safe**.
- Hits resolve to `…/envs/geoworkflow/lib/libgdal…` etc. → the package is
  borrowing conda's native libs → **mixing risk; install it from conda-forge
  instead**.

Record every pip-installed dependency in the `pip:` section of `environment.yml`
so the env stays reproducible. Full rationale: docs → Installation → "Adding new
dependencies (pip vs conda)".

**Worked example.** `exactextract` (coverage-weighted zonal stats) was added via
pip wheel because its conda-forge solve hung. `ldd` confirmed it vendors its own
GEOS (`exactextract.libs/libgeos-*.so`, RPATH `$ORIGIN/../exactextract.libs`) and
links no conda native lib — so it is safe and isolated from conda's GEOS.
