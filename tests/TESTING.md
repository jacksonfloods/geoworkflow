# Testing guide

Run the suite from the repo root with the **geoworkflow** conda environment:

```bash
python -m pytest tests/ -q
```

The default run is expected to be **green** (passes + skips, no failures). If
you see failures, something you changed broke — do not commit.

## Test layout

- `tests/unit/test_core/` — statistics registry, dataset registry
- `tests/unit/test_utils/` — zonal engine (incl. equivalence-vs-exactextract
  gold-standard tests), raster source layer, intermediate cleanup
- `tests/unit/test_processors/spatial/` — hex grid generator
- `tests/unit/test_processors/integration/` — Pipeline 1 (grid statistics),
  Pipeline 2 (NetCDF cubes) and the hexcube query helpers
- `tests/unit/test_processors/extraction/` — extraction processors
  (GEE raster export tests are pure-logic: no network needed)

## Quarantined legacy tests (2026-06-10)

A number of tests for **pre-existing modules outside the hex pipelines** had
drifted from the code they test and failed for reasons unrelated to current
work. Rather than leave the suite permanently red (which trains people and AI
agents to ignore failures), they are explicitly skipped with reasons:

| Where | What | Why quarantined |
|---|---|---|
| `tests/test_stastitical_enrichment.py` (whole file) | StatisticalEnrichment processor | Tests predate a rewrite of `StatisticalEnrichmentConfig`; they construct fields that no longer exist. Needs a test rewrite against the current schema. (Note the filename typo: "stastitical".) |
| `tests/unit/.../test_gcs_integration.py` (whole file) | Open Buildings GCS extraction | Live integration tests: require network + GCS bucket access, and assertions have drifted. Run deliberately when touching `open_buildings_gcs`. |
| `test_s2_utils.py` (7 tests) | S2 cell utilities | Assertion drift vs current s2sphere behavior. |
| `test_osm_highways.py` (1 test) | OSM highways | `test_full_processing_flow` drifted. |
| `test_openbuildings_gcs.py` (2 tests) | Open Buildings GCS | One validation-message drift, one fixture error. |

**If you work on one of these modules:** un-skip its tests first, fix them
against current behavior, and delete the corresponding row here. Do not write
new code against a quarantined module without reviving its tests.

## Conventions for new tests

- Mirror the source layout (`src/geoworkflow/utils/zonal_utils.py` →
  `tests/unit/test_utils/test_zonal_utils.py`).
- Guard optional heavy imports and skip cleanly when absent (see any test
  file's `HAS_LIBS` pattern), so a minimal install can still run the suite.
- No network in unit tests. Network-dependent tests belong in clearly named
  integration files and must skip (not fail) without credentials.
- For numerical engines, prefer **equivalence tests against a reference
  implementation** over hand-computed constants (see
  `test_zonal_utils.py::TestCoverageReuse` — values must match exactextract).
