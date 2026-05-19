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

## Dependencies

If imports fail, install missing packages:
```bash
pip install s2sphere gcsfs earthengine-api
```
