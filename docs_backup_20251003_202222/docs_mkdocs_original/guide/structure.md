# Project Structure

This page documents the organization of the GeoWorkflow codebase.

## Source Code Layout

```
src/geoworkflow/
│   ├── __init__.py
│   ├── __version__.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── raster/
│   │   │   ├── __init__.py
│   │   │   ├── processor.py
│   │   ├── vector/
│   │   │   ├── __init__.py
│   │   ├── reports/
│   │   │   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── enhanced_base.py
│   │   ├── exceptions.py
│   │   ├── logging_setup.py
│   │   ├── pipeline.py
│   │   ├── pipeline_enhancements.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── earth_engine_error_handler.py
│   │   ├── earth_engine_utils.py
│   │   ├── file_utils.py
│   │   ├── mask_utils.py
│   │   ├── progress_utils.py
│   │   ├── raster_utils.py
│   │   ├── resource_utils.py
│   │   ├── validation.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── cli_structure.py
│   │   ├── main.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── aoi.py
│   │   │   ├── extract.py
│   │   │   ├── pipeline.py
│   │   │   ├── process.py
│   │   │   ├── visualize.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── config_models.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── integration/
│   │   │   ├── __init__.py
│   │   │   ├── enrichment.py
│   │   ├── spatial/
│   │   │   ├── __init__.py
│   │   │   ├── aligner.py
│   │   │   ├── clipper.py
│   │   │   ├── masker.py
│   │   ├── extraction/
│   │   │   ├── __init__.py
│   │   │   ├── archive.py
│   │   │   ├── open_buildings.py
│   │   ├── aoi/
│   │   │   ├── __init__.py
│   │   │   ├── processor.py
```

## Directory Descriptions

### `core/`

**Core modules** - Foundation classes, base processors, configuration, and constants

### `processors/`

**Data processors** - Specialized processors for each workflow stage

### `processors/aoi/`

Area of Interest (AOI) creation and management

### `processors/spatial/`

Spatial operations (clipping, alignment, reprojection)

### `processors/extraction/`

Data extraction from archives and downloads

### `processors/integration/`

Statistical enrichment and data integration

### `schemas/`

**Configuration schemas** - Pydantic models for validation

### `utils/`

**Utility modules** - Helper functions and common operations

### `cli/`

**Command-line interface** - Entry points for CLI commands

