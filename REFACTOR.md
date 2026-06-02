# Refactor Todo

Issues identified that are not urgent but should be addressed in a dedicated refactor pass. Not blocking current work.

---

## 1. `from_file` factory functions are broken

**Symptom:** Calling any of the `create_*_processor(config_path)` factory functions (e.g. `create_aoi_processor`, `create_hex_grid_processor`) crashes with `AttributeError: type object 'XConfig' has no attribute 'from_file'`.

**Root cause:** There are two unrelated `BaseConfig` classes in the codebase:
- `core/base.py` — abstract class declaring `from_file` and `from_dict` as abstract methods
- `schemas/config_models.py` — Pydantic `BaseModel` with no such methods

All config classes inherit from the Pydantic one, so the factory functions reference methods that don't exist.

**Fix direction:** Implement `from_file` and `from_dict` on the Pydantic `BaseConfig` (load JSON/YAML, construct model). Either delete the abstract `BaseConfig` in `core/base.py` or make the Pydantic version inherit from it.

---

## 2. Test infrastructure is broken

**Symptom:** All 9 existing test files fail to collect under pytest with `ModuleNotFoundError: No module named 'geoworkflow.X'`. Smoke-importing the package in a Python REPL works fine.

**Root cause:** The repo directory is named `geoworkflow/`, same as the package. Pytest's auto-discovery walks up from test files, finds `tests/__init__.py`, treats `tests` as a subpackage, and imports the conftest as `geoworkflow.tests.conftest`. The repo root gets added to `sys.path` as if it were the `geoworkflow` package, shadowing the real one in `src/geoworkflow/`.

**Fix directions** (pick one):
- Remove `__init__.py` files from `tests/` and its subdirectories so pytest uses rootdir-based discovery
- Rename the repo directory to something other than `geoworkflow` (e.g. `geoworkflow-repo`)
- Switch pytest to `--import-mode=importlib` AND restructure tests to not be a package

---

## 3. Side effects in input validation

**Where:** `_validate_custom_inputs` in `AOIProcessor` and `HexGridProcessor` calls `mkdir(parents=True, exist_ok=True)` on the output directory.

**Why it's a problem:** Validation should be a pure check that returns a result. Creating directories is a setup concern. Mixing them means failed validation can still leave artifacts on disk.

**Fix direction:** Move directory creation into `_setup_custom_processing`. Validation just confirms the path is writable.

---

## 4. Exception flow in `process_data` is muddled

**Where:** `AOIProcessor.process_data`, `HexGridProcessor.process_data`, likely others.

**Pattern:**
```python
result = ProcessingResult(success=True)
try:
    ...
except Exception as exc:
    result.success = False
    result.message = f"... failed: {exc}"
    raise ProcessingError(result.message)
return result
```

**Problem:** The result object is built up in the except branch, then immediately discarded when the raise propagates. The outer `process()` in `TemplateMethodProcessor` builds its own result from the caught exception.

**Fix direction:** Either don't raise (return the failed result, let the caller check `result.success`), or don't build the local result in the except (just raise). Pick one convention and apply consistently.

---

## 5. Hardcoded `driver="GeoJSON"` in `to_file` calls

**Where:** Multiple processors hardcode the GeoJSON driver even when the `output_file` could in principle have a different extension (e.g. `.gpkg`, `.shp`).

**Fix direction:** Either auto-detect from extension (let `gpd.to_file` infer) or validate the extension in the config schema and document GeoJSON-only support clearly.

---

## 6. `_WORKING_CRS` in `HexGridProcessor` is hardcoded

**Where:** [`hexgrid.py`](src/geoworkflow/processors/spatial/hexgrid.py) — `_WORKING_CRS = "ESRI:102022"` is the metric CRS used internally for hex geometry math. It is coupled to the default `grid_origin_x/y` values (`-3_000_000`, `-2_000_000` in ESRI:102022 meters).

**Why it's a problem:** A user who overrides `grid_origin_x/y` for a non-Africa region cannot also override the working CRS. The grid would be generated in ESRI:102022 space using their custom origin, producing wrong results.

**Fix direction:** Either expose `working_crs` as a config field that defaults to ESRI:102022, or document that the three values (`grid_origin_x`, `grid_origin_y`, working CRS) are a coupled triplet that should change together. The cleanest solution is to make all three configurable with a single "preset" abstraction.
