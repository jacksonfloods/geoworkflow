# Quick Start

This guide will get you up and running with GeoWorkflow in minutes.

## Basic Workflow Example

```python
from geoworkflow.core.pipeline import ProcessingPipeline
from pathlib import Path

# Define your workflow configuration
config = {
    "stages": ["extract", "clip", "align"],
    "source_dir": "data/00_source",
    "output_dir": "data/processed",
    "countries": ["Ghana", "Togo"]
}

# Create and run pipeline
pipeline = ProcessingPipeline(config)
results = pipeline.run()

# Check results
print(f"Processed {results.processed_count} files")
```

## Using Individual Processors

```python
from geoworkflow.processors.spatial.clipper import ClippingProcessor
from geoworkflow.schemas.config_models import ClippingConfig

# Configure clipping
config = ClippingConfig(
    input_dir=Path("data/01_extracted"),
    output_dir=Path("data/02_clipped"),
    boundary_file=Path("data/aoi/ghana_boundary.geojson")
)

# Run processor
processor = ClippingProcessor(config)
result = processor.process()
```

## Next Steps

- Learn about [Configuration](configuration.md)
- Explore [Directory Structure](../guide/structure.md)
- Try a [Complete Workflow Tutorial](../tutorials/basic-workflow.md)
