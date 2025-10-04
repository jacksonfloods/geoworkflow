#!/bin/bash
# setup_mkdocs.sh - Create all missing MkDocs documentation files

set -e  # Exit on error

echo "Setting up MkDocs documentation structure..."

# First, update mkdocs.yaml to point to the correct docs directory
echo "Updating mkdocs.yaml configuration..."
cat > mkdocs.yaml << 'YAMLEOF'
site_name: GeoWorkflow Project Guide
site_description: Comprehensive guide to the AfricaPolis GeoWorkflow codebase
site_author: AfricaPolis Team
repo_url: https://github.com/jacksonfloods/geoworkflow
repo_name: geoworkflow

# Point to the docs_mkdocs directory
docs_dir: docs_mkdocs

theme:
  name: material
  palette:
    # Light mode
    - scheme: default
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - scheme: slate
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - toc.integrate

plugins:
  - search
  - gen-files:
      scripts:
        - docs_mkdocs/gen_ref_pages.py
  - literate-nav:
      nav_file: SUMMARY.md
  - section-index

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
    - Configuration: getting-started/configuration.md
  - Project Guide:
    - Directory Structure: guide/structure.md
    - Core Concepts: guide/concepts.md
    - Workflow Stages: guide/workflow.md
  - Tutorials:
    - Basic Workflow: tutorials/basic-workflow.md
    - Custom Processors: tutorials/custom-processors.md
  - Reference:
    - API Documentation: https://your-sphinx-docs-url/
  - Development:
    - Contributing: development/contributing.md
    - Testing: development/testing.md
YAMLEOF

echo "✓ Updated mkdocs.yaml"
echo ""

# Create quickstart.md
cat > docs_mkdocs/getting-started/quickstart.md << 'EOF'
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
EOF

echo "✓ Created quickstart.md"

# Create configuration.md
cat > docs_mkdocs/getting-started/configuration.md << 'EOF'
# Configuration Guide

GeoWorkflow uses YAML configuration files to define workflows and processor settings.

## Configuration File Structure

```yaml
# workflow_config.yaml
workflow:
  name: "AfricaPolis Processing"
  stages:
    - extract
    - clip
    - align
    - enrich

paths:
  source_dir: "data/00_source"
  extracted_dir: "data/01_extracted"
  clipped_dir: "data/02_clipped"
  processed_dir: "data/03_processed"
  output_dir: "data/04_analysis_ready"

aoi:
  countries: ["Ghana", "Togo", "Kenya", "Tanzania"]
  buffer_km: 50

processing:
  target_crs: "EPSG:4326"
  resolution_m: 100
  resampling_method: "bilinear"
```

## Loading Configuration

```python
from geoworkflow.core.config import ConfigManager

# Load from file
config = ConfigManager.load("config/workflows/my_workflow.yaml")

# Or create programmatically
config = {
    "stages": ["clip", "align"],
    "source_dir": "data/raw"
}
```

## Environment Variables

Set these environment variables for Earth Engine integration:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export GEE_PROJECT_ID="your-project-id"
```

## See Also

- [Processor-specific configs](../guide/concepts.md#configuration-models)
- [Workflow examples](../tutorials/basic-workflow.md)
EOF

echo "✓ Created configuration.md"

# Create workflow.md
cat > docs_mkdocs/guide/workflow.md << 'EOF'
# Workflow Stages

GeoWorkflow processes geospatial data through a series of well-defined stages.

## Processing Pipeline

```
Source Data → Extract → Clip → Align → Enrich → Visualize → Analysis-Ready
```

## Stage Descriptions

### 1. Extract
**Purpose**: Extract data from archives and downloads

**Input**: Compressed archives (ZIP, TAR.GZ)  
**Output**: Uncompressed geospatial files  
**Processor**: `ArchiveExtractionProcessor`

```python
from geoworkflow.processors.extraction.archive import ArchiveExtractionProcessor

processor = ArchiveExtractionProcessor(config)
result = processor.process()
```

### 2. Clip
**Purpose**: Spatially subset data to area of interest

**Input**: Large-extent rasters or vectors  
**Output**: Clipped data for specific regions  
**Processor**: `ClippingProcessor`

**Use cases**:
- Extract country-specific data from global datasets
- Focus processing on urban areas
- Reduce file sizes for analysis

### 3. Align
**Purpose**: Ensure consistent spatial properties

**Input**: Misaligned rasters (different CRS, resolution, extent)  
**Output**: Aligned rasters ready for analysis  
**Processor**: `AlignmentProcessor`

**Operations**:
- Reproject to common CRS
- Resample to common resolution
- Match extents and pixel grids

### 4. Enrich
**Purpose**: Calculate zonal statistics and integrate datasets

**Input**: Vector boundaries + raster data  
**Output**: Enriched vector with statistical attributes  
**Processor**: `StatisticalEnrichmentProcessor`

**Calculated metrics**:
- Mean, min, max, std
- Percentiles
- Counts and sums

### 5. Visualize
**Purpose**: Generate maps and charts

**Input**: Analysis-ready data  
**Output**: PNG/PDF maps, interactive HTML  
**Processor**: `VisualizationProcessor`

## Running Multi-Stage Workflows

```python
from geoworkflow.core.pipeline import ProcessingPipeline

config = {
    "stages": ["extract", "clip", "align", "enrich"],
    "source_dir": "data/00_source",
    # ... other config
}

pipeline = ProcessingPipeline(config)
results = pipeline.run()

# Or run stages individually
pipeline.run_stage("clip")
pipeline.run_stage("align")
```

## Stage Dependencies

Each stage expects specific inputs:

| Stage | Requires | Produces |
|-------|----------|----------|
| Extract | Archives | Raw files |
| Clip | Raw files + boundaries | Clipped files |
| Align | Clipped rasters | Aligned rasters |
| Enrich | Aligned rasters + vectors | Enriched vectors |
| Visualize | Enriched data | Maps/charts |
EOF

echo "✓ Created workflow.md"

# Create basic-workflow.md
cat > docs_mkdocs/tutorials/basic-workflow.md << 'EOF'
# Basic Workflow Tutorial

This tutorial walks through a complete data processing workflow using GeoWorkflow.

## Scenario

Process PM2.5 air quality data for Ghana and Togo, enriching urban boundary polygons with pollution statistics.

## Prerequisites

- GeoWorkflow installed
- Sample data in `data/00_source/`
- AFRICAPOLIS boundaries

## Step 1: Create Area of Interest

```python
from geoworkflow.processors.aoi.processor import AOIProcessor
from geoworkflow.schemas.config_models import AOIConfig
from pathlib import Path

# Configure AOI creation
aoi_config = AOIConfig(
    source_file=Path("data/00_source/boundaries/africa_boundaries.geojson"),
    output_dir=Path("data/aoi"),
    countries=["Ghana", "Togo"],
    buffer_km=50,
    output_format="geojson"
)

# Create AOI
aoi_processor = AOIProcessor(aoi_config)
result = aoi_processor.process()

print(f"Created AOI: {result.output_paths[0]}")
```

## Step 2: Extract Archives

```python
from geoworkflow.processors.extraction.archive import ArchiveExtractionProcessor
from geoworkflow.schemas.config_models import ExtractionConfig

extract_config = ExtractionConfig(
    source_dir=Path("data/00_source/archives/pm25"),
    output_dir=Path("data/01_extracted/pm25"),
    archive_format="zip"
)

extractor = ArchiveExtractionProcessor(extract_config)
result = extractor.process()
```

## Step 3: Clip to AOI

```python
from geoworkflow.processors.spatial.clipper import ClippingProcessor
from geoworkflow.schemas.config_models import ClippingConfig

clip_config = ClippingConfig(
    input_dir=Path("data/01_extracted/pm25"),
    output_dir=Path("data/02_clipped/pm25"),
    boundary_file=Path("data/aoi/ghana_togo_aoi.geojson"),
    maintain_extent=False
)

clipper = ClippingProcessor(clip_config)
result = clipper.process()
```

## Step 4: Align Rasters

```python
from geoworkflow.processors.spatial.aligner import AlignmentProcessor
from geoworkflow.schemas.config_models import AlignmentConfig

align_config = AlignmentConfig(
    input_dir=Path("data/02_clipped/pm25"),
    output_dir=Path("data/03_processed/pm25"),
    target_crs="EPSG:4326",
    resolution=(0.001, 0.001),  # ~100m at equator
    resampling_method="bilinear"
)

aligner = AlignmentProcessor(align_config)
result = aligner.process()
```

## Step 5: Enrich Urban Boundaries

```python
from geoworkflow.processors.integration.enrichment import StatisticalEnrichmentProcessor
from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig

enrich_config = StatisticalEnrichmentConfig(
    vector_file=Path("data/01_extracted/AFRICAPOLIS2020.geojson"),
    raster_dir=Path("data/03_processed/pm25"),
    output_dir=Path("data/04_analysis_ready"),
    statistics=["mean", "std", "min", "max"],
    prefix="pm25"
)

enricher = StatisticalEnrichmentProcessor(enrich_config)
result = enricher.process()

print(f"Enriched {result.processed_count} urban areas")
```

## Step 6: Examine Results

```python
import geopandas as gpd

# Load enriched data
gdf = gpd.read_file("data/04_analysis_ready/africapolis_pm25_stats.geojson")

# View statistics
print(gdf[["AgglomName", "pm25_mean", "pm25_std"]].head())

# Simple visualization
gdf.plot(column="pm25_mean", legend=True, figsize=(12, 8))
```

## Complete Pipeline Version

Or run everything at once:

```python
from geoworkflow.core.pipeline import ProcessingPipeline

pipeline_config = {
    "stages": ["extract", "clip", "align", "enrich"],
    "source_dir": Path("data/00_source"),
    "countries": ["Ghana", "Togo"],
    "target_crs": "EPSG:4326",
    "resolution_m": 100
}

pipeline = ProcessingPipeline(pipeline_config)
results = pipeline.run()

for stage, result in results.items():
    print(f"{stage}: {result.processed_count} files processed")
```

## Troubleshooting

**Issue**: Files not found  
**Solution**: Check that paths are correct and files exist

**Issue**: CRS mismatch warnings  
**Solution**: This is expected - the alignment stage handles it

**Issue**: Memory errors with large rasters  
**Solution**: Process smaller regions or use windowed reading
EOF

echo "✓ Created basic-workflow.md"

# Create custom-processors.md
cat > docs_mkdocs/tutorials/custom-processors.md << 'EOF'
# Creating Custom Processors

Learn how to extend GeoWorkflow with custom processors.

## Basic Processor Structure

All processors inherit from `BaseProcessor`:

```python
from geoworkflow.core.base import BaseProcessor, ProcessingResult
from pathlib import Path
from typing import Dict, Any

class MyCustomProcessor(BaseProcessor):
    """Custom processor for specific transformation."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        # Custom initialization
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processor configuration."""
        required_keys = ["input_dir", "output_dir"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config: {key}")
        return config
    
    def process(self) -> ProcessingResult:
        """Execute the processing logic."""
        self._start_processing()
        
        # Your processing logic here
        processed_count = 0
        
        try:
            # Process files
            processed_count = self._do_processing()
            
            result = ProcessingResult(
                success=True,
                processed_count=processed_count,
                elapsed_time=self._elapsed_time()
            )
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            result = ProcessingResult(success=False, message=str(e))
        
        return result
    
    def _do_processing(self) -> int:
        """Implement your actual processing logic."""
        # Example: process all GeoJSON files
        count = 0
        input_dir = Path(self.config["input_dir"])
        
        for geojson_file in input_dir.glob("*.geojson"):
            # Process each file
            self._process_file(geojson_file)
            count += 1
        
        return count
```

## Example: Custom Filter Processor

```python
import geopandas as gpd
from geoworkflow.core.base import BaseProcessor, ProcessingResult

class AttributeFilterProcessor(BaseProcessor):
    """Filter vector features by attribute values."""
    
    def _validate_config(self, config):
        required = ["input_file", "output_file", "filter_column", "filter_values"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing: {key}")
        return config
    
    def process(self) -> ProcessingResult:
        self._start_processing()
        
        try:
            # Load data
            gdf = gpd.read_file(self.config["input_file"])
            
            # Apply filter
            column = self.config["filter_column"]
            values = self.config["filter_values"]
            filtered = gdf[gdf[column].isin(values)]
            
            # Save result
            filtered.to_file(self.config["output_file"])
            
            return ProcessingResult(
                success=True,
                processed_count=len(filtered),
                elapsed_time=self._elapsed_time(),
                message=f"Filtered to {len(filtered)} features"
            )
        except Exception as e:
            self.logger.error(f"Filter failed: {e}")
            return ProcessingResult(success=False, message=str(e))

# Usage
config = {
    "input_file": "data/cities.geojson",
    "output_file": "data/large_cities.geojson",
    "filter_column": "population",
    "filter_values": [100000, 500000, 1000000]
}

processor = AttributeFilterProcessor(config)
result = processor.process()
```

## Integrating with Pipeline

Register your processor:

```python
from geoworkflow.core.pipeline import ProcessingPipeline

# Register custom processor
pipeline = ProcessingPipeline(config)
pipeline.register_processor("filter", AttributeFilterProcessor)

# Use in workflow
workflow_config = {
    "stages": ["filter", "clip", "align"],
    # ...
}
pipeline.run()
```

## Best Practices

1. **Always validate configuration** in `_validate_config()`
2. **Use ProcessingResult** for consistent return values
3. **Log important events** using `self.logger`
4. **Handle errors gracefully** with try/except
5. **Document your processor** with clear docstrings
6. **Test thoroughly** with various input types
EOF

echo "✓ Created custom-processors.md"

# Create contributing.md
cat > docs_mkdocs/development/contributing.md << 'EOF'
# Contributing to GeoWorkflow

We welcome contributions! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/jacksonfloods/geoworkflow.git
cd geoworkflow

# Create development environment
conda env create -f environment.yml
conda activate geoworkflow

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use Google-style docstrings
- Type hints for function signatures

### Running Formatters

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check types
mypy src/geoworkflow

# Lint
flake8 src/ tests/
```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/ -v
   pytest --cov=geoworkflow --cov-report=html
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-new-feature
   ```

## Testing Guidelines

- Write tests for all new processors
- Aim for >80% code coverage
- Use pytest fixtures for common test data
- Test both success and failure cases

Example test:

```python
import pytest
from geoworkflow.processors.spatial.clipper import ClippingProcessor

def test_clipper_basic(tmp_path):
    """Test basic clipping functionality."""
    config = {
        "input_dir": "tests/fixtures/rasters",
        "output_dir": tmp_path,
        "boundary_file": "tests/fixtures/boundaries/test_aoi.geojson"
    }
    
    processor = ClippingProcessor(config)
    result = processor.process()
    
    assert result.success
    assert result.processed_count > 0
```

## Documentation

- Update docstrings for any changed functions
- Add examples to documentation files
- Update README if adding major features
- Build docs locally to check formatting:
  ```bash
  cd docs
  make html
  mkdocs serve
  ```

## Pull Request Process

1. Ensure all tests pass
2. Update CHANGELOG.md
3. Request review from maintainers
4. Address review feedback
5. Squash commits if requested

## Questions?

Open an issue or reach out to the maintainers.
EOF

echo "✓ Created contributing.md"

# Create testing.md
cat > docs_mkdocs/development/testing.md << 'EOF'
# Testing Guide

GeoWorkflow uses pytest for testing.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=geoworkflow --cov-report=html

# Run specific test file
pytest tests/test_aoi_processor.py

# Run specific test
pytest tests/test_aoi_processor.py::test_aoi_creation

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/                # Test data
│   ├── rasters/
│   ├── vectors/
│   └── configs/
├── unit/                    # Unit tests
│   ├── test_processors/
│   ├── test_utils/
│   └── test_schemas/
└── integration/             # Integration tests
    └── test_pipelines/
```

## Writing Tests

### Basic Test Example

```python
import pytest
from pathlib import Path
from geoworkflow.processors.aoi.processor import AOIProcessor

def test_aoi_processor_creates_output(tmp_path):
    """Test that AOI processor creates output file."""
    config = {
        "source_file": "tests/fixtures/boundaries.geojson",
        "output_dir": tmp_path,
        "countries": ["Ghana"],
        "output_format": "geojson"
    }
    
    processor = AOIProcessor(config)
    result = processor.process()
    
    assert result.success
    assert len(result.output_paths) > 0
    assert result.output_paths[0].exists()
```

### Using Fixtures

```python
# conftest.py
@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample raster for testing."""
    import rasterio
    import numpy as np
    
    raster_path = tmp_path / "test_raster.tif"
    data = np.random.rand(100, 100)
    
    with rasterio.open(
        raster_path, 'w',
        driver='GTiff',
        height=100, width=100,
        count=1, dtype=data.dtype,
        crs='EPSG:4326',
        transform=rasterio.transform.from_bounds(0, 0, 1, 1, 100, 100)
    ) as dst:
        dst.write(data, 1)
    
    return raster_path

# test_file.py
def test_with_fixture(sample_raster):
    """Test using the fixture."""
    assert sample_raster.exists()
```

### Testing Errors

```python
def test_processor_invalid_config():
    """Test that invalid config raises error."""
    with pytest.raises(ValueError, match="Missing required"):
        processor = MyProcessor({})
```

### Parametrized Tests

```python
@pytest.mark.parametrize("crs,expected", [
    ("EPSG:4326", "WGS84"),
    ("EPSG:3857", "Web Mercator"),
])
def test_crs_names(crs, expected):
    """Test CRS name conversion."""
    result = get_crs_name(crs)
    assert expected in result
```

## Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_large_dataset():
    pass

# Mark integration tests
@pytest.mark.integration
def test_full_pipeline():
    pass

# Run only unit tests
pytest -m "not integration"
```

## Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=geoworkflow --cov-report=html

# Open report
open htmlcov/index.html
```

## Continuous Integration

Tests run automatically on every PR via GitHub Actions.
EOF

echo "✓ Created testing.md"

# Generate structure documentation
echo ""
echo "Generating structure documentation..."
python docs_mkdocs/gen_ref_pages.py

echo "✓ Generated structure.md"
echo ""

# Test the build
echo "Testing MkDocs build..."
if mkdocs build --strict; then
    echo ""
    echo "✓ MkDocs build successful!"
    echo ""
    echo "Documentation setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Preview docs: mkdocs serve"
    echo "  2. Visit: http://localhost:8000"
    echo "  3. Build for production: mkdocs build"
    echo ""
else
    echo ""
    echo "✗ MkDocs build failed"
    echo "Check the error messages above"
    exit 1
fi