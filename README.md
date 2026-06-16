# geoworkflow

Geospatial workflows for African urban analysis. The primary interface is a
single **queryable hex database** over all ~9,000 African agglomerations, keyed
by hexagon — `geoworkflow.store`:

```python
from geoworkflow import open_hexdb
db = open_hexdb()
db["KEN"]["Nairobi"]["lst_day"].climatology(month=6).plot()    # avg of all Junes
db["KEN"]["Nairobi"]["lst_day"].sel("2020-06").to_geodataframe()
db.hexagon("UTM32737_HQ-008188_R+031618")                      # one hexagon, any city
```

Under the hood it's a DuckDB + Parquet warehouse (`data/hexdb/`), built
**reproducibly from a YAML recipe** and self-describing via a living
`_recipe.yaml` + `_provenance.jsonl`. Build/extend it with `geoworkflow hexdb`
(`build` / `query` / `plot` / `add-city` / `add-metric` / `provenance`). Time
selection follows pandas/xarray conventions (`.sel("2020-06")` selects, reducers
aggregate, `.groupby_month()`/`.climatology()` for climatologies).

Data gets *into* the database via the pipelines below: download rasters (Google
Earth Engine and elsewhere) and resample heterogeneous raster data onto
**hexagonal grids** as coverage-weighted statistics, written straight into the
warehouse (no intermediate cubes).

```
AOI ──HexGridProcessor──▶ hex grid ──GridStatisticsProcessor──▶ data/hexdb/ (DuckDB+Parquet)
       (250 m UTM hexes)      ▲      (exactextract, coverage-weighted)   ──▶ open_hexdb()
                              │
   GEERasterExportProcessor ──┘  (declarative GEE downloads, clipped per grid)
```

The pipelines tolerate **heterogeneous inputs** — mixed CRSs (including files
with *mislabeled* CRS tags), resolutions, GeoTIFF + netCDF, different temporal
cadences — because file naming/metadata conventions live in a JSON **dataset
registry**, not in code. Inputs that match no registry entry are an **error by
default**, never silent junk.

## Quickstart

```bash
conda env create -f environment.yml   # env name: geoworkflow
conda activate geoworkflow
pip install -e .
python -m pytest tests/ -q            # should be green (see tests/TESTING.md)
```

End-to-end on one city (see `examples/full_pipeline_example.py` for the same
flow with processor classes):

```python
from geoworkflow.processors.spatial.hexgrid import HexGridProcessor
from geoworkflow.processors.integration.grid_statistics import compute_grid_statistics
from geoworkflow.processors.integration.netcdf_builder import build_grid_netcdf

# 1. hex grid for an agglomeration (flat-top, 150 m, globally aligned GridIDs)
HexGridProcessor({"aoi_file": "../data/boundaries/nairobi.gpkg",
                  "output_file": "../data/grids/nairobi_hex.geojson",
                  "side_length": 150.0}).process()

# 2. rasters -> per-hex statistics (tidy long Parquet)
compute_grid_statistics(
    grid_file="../data/grids/nairobi_hex.geojson",
    raster_inputs=["../data/global/PM25", "../data/global/odiac"],
    output_file="../data/grid_stats/nairobi.parquet",
    statistics=["weighted_mean"],          # or median, p90, majority, ...
)

# 3. tidy table -> compressed NetCDF cube, dims (cell, time) [+ (cell, year)]
build_grid_netcdf("../data/grid_stats/nairobi.parquet",
                  "../data/grids/nairobi_hex.geojson",
                  "../data/cubes/nairobi.nc", title="Nairobi")
```

Downloading new GEE data is a config call, not a script — see
`GEERasterExportProcessor` in [CLAUDE.md](CLAUDE.md) (or
`notebooks/download_gee_to_hexgrids.ipynb` for the lab's standard datasets).

## Where things are

| | |
|---|---|
| Agent / contributor guide (conventions, pipelines, dependency rules) | [CLAUDE.md](CLAUDE.md) (= [AGENTS.md](AGENTS.md)) |
| Test guide + quarantined-legacy-tests list | [tests/TESTING.md](tests/TESTING.md) |
| Hex grid design & rationale (CRS modes, MAUP, change-of-support) | [docs/grid_design.md](docs/grid_design.md) |
| Data tree documentation (datasets, naming, layout) | `../data/DATA_CATALOG.md` |
| How to read the NetCDF cubes (for collaborators, no geoworkflow needed) | `../data/HOW_TO_USE_CUBES.md` |
| Sphinx docs | `docs/` |

## Key design points

- **exactextract** for zonal statistics: coverage-fraction weighting means a
  hexagon smaller than one raster pixel still gets that pixel's value
  (rasterstats returns null there — do not switch back).
- **Multi-band fast paths**: monthly stacks of one variable are reduced with a
  single coverage computation (`compute_zonal_statistics_coverage_reuse`,
  ~60× faster than per-file; equivalence-tested against exactextract).
- **Cubes** put each variable on the time axis matching its cadence — monthly
  `(cell, time)`, annual `(cell, year)` — with CF `flag_values`/`flag_meanings`
  for categorical layers, zlib + float32 storage.
- **Outputs live in `../data/`**, never inside this repo.
- **Conda owns the compiled geo stack** (GDAL/GEOS/PROJ/HDF5). Read the
  pip-vs-conda rules in CLAUDE.md before installing anything.
