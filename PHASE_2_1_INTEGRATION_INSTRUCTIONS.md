# Phase 2.1 Integration Instructions

## Summary
A new configuration class `OpenBuildingsGCSConfig` has been created for the GCS-based 
Open Buildings extraction processor.

## Files Created
- `src/geoworkflow/schemas/open_buildings_gcs_config.py` - New config class

## Next Steps

### Option 1: Keep as Separate Module (Recommended)
The configuration is currently in its own file. This is fine and follows good practices.

To use it in your code:
```python
from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig
```

### Option 2: Integrate into config_models.py
If you prefer to have all configs in one file, add the following to 
`src/geoworkflow/schemas/config_models.py`:

1. Add to imports at the top:
```python
from typing import Optional, Literal  # Add Literal if not already imported
```

2. Add the entire `OpenBuildingsGCSConfig` class definition to the file

3. Add to `__all__` export list:
```python
__all__ = [
    # ... existing exports ...
    'OpenBuildingsGCSConfig',
    'OpenBuildingsGCSPointsConfig',
]
```

### Update schemas/__init__.py
Add to `src/geoworkflow/schemas/__init__.py`:

```python
from .open_buildings_gcs_config import (
    OpenBuildingsGCSConfig,
    OpenBuildingsGCSPointsConfig,
)

__all__ = [
    # ... existing exports ...
    'OpenBuildingsGCSConfig',
    'OpenBuildingsGCSPointsConfig',
]
```

## Testing the Configuration

Test the new configuration:

```python
from pathlib import Path
from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig

# Create a test config
config = OpenBuildingsGCSConfig(
    aoi_file=Path("test_area.geojson"),  # Replace with actual file
    output_dir=Path("./test_output/"),
    confidence_threshold=0.75,
    num_workers=4
)

# Print summary
print(config.summary())

# Get output path
print(f"Output will be saved to: {config.get_output_file_path()}")
```

## Configuration Features

The `OpenBuildingsGCSConfig` includes:

✓ **Validation**: Automatically validates file paths, value ranges, and GCS paths
✓ **Defaults**: Sensible defaults for all optional parameters
✓ **Helper Methods**: 
  - `get_output_file_path()` - Generate output filename
  - `summary()` - Get configuration overview
  - `estimate_memory_usage()` - Estimate memory requirements
✓ **Type Safety**: Full Pydantic validation with type hints
✓ **Documentation**: Detailed field descriptions and examples

## Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `confidence_threshold` | 0.75 | Minimum building confidence (0.5-1.0) |
| `num_workers` | 4 | Parallel workers for download |
| `s2_level` | 6 | S2 cell level (matches GCS structure) |
| `export_format` | "geojson" | Output format (geojson/shapefile/csv) |
| `min_area_m2` | 10.0 | Minimum building area filter |

## What's Next?

Once you've reviewed and are happy with the configuration:

1. Choose integration option (separate module or merge)
2. Update `__init__.py` files as needed
3. Test the configuration with a sample AOI file
4. Proceed to **Phase 2.2**: Main GCS Processor implementation

## Backup

A backup of your original config_models.py was created:
- Location: `src/geoworkflow/schemas/config_models.py.backup_phase2_1_*`
