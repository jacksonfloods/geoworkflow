# GeoWorkflow

**Comprehensive geospatial data processing workflow for African geospatial analysis**

GeoWorkflow is a unified Python toolkit designed specifically for processing, analyzing, and visualizing geospatial data with a focus on African datasets. The package provides a streamlined workflow from raw data archives to analysis-ready datasets, supporting major geospatial data sources including Copernicus Land Cover, ODIAC emissions, PM2.5 concentrations, and AFRICAPOLIS urban data.

Built with a modular architecture, GeoWorkflow transforms complex geospatial processing tasks into simple, reproducible workflows. Whether you're a researcher studying urban development patterns, an analyst working with environmental data, or a developer building geospatial applications, GeoWorkflow provides the tools you need to efficiently process large-scale African geospatial datasets.

The package emphasizes ease of use without sacrificing power, offering both a command-line interface for quick operations and a Python API for complex workflows. All processing stages are designed to handle the unique challenges of African geospatial data, including coordinate system management, large file processing, and cross-dataset integration.

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Processing Workflow](#processing-workflow)
6. [Command Line Interface](#command-line-interface)
7. [Python API Usage](#python-api-usage)
8. [Configuration](#configuration)
9. [Data Sources](#data-sources)
10. [Examples](#examples)
11. [Advanced Usage](#advanced-usage)
12. [API Reference](#api-reference)
13. [Contributing](#contributing)
14. [License](#license)

## Key Features

### 🌍 **African-Focused Geospatial Processing**
- Optimized for common African geospatial datasets and coordinate systems
- Built-in support for Africa Albers Equal Area projection and UTM zones
- Handles administrative boundary data for all African countries

### 📦 **Complete Data Pipeline**
- **Area of Interest (AOI) Creation**: Define study areas from country boundaries with buffering
- **Archive Extraction**: Automated extraction from ZIP archives with format detection
- **Spatial Clipping**: Clip raster and vector data to study areas
- **Raster Alignment**: Align multiple datasets to common grids and projections
- **Statistical Enrichment**: Extract zonal statistics for urban areas and administrative units
- **Visualization**: Generate publication-quality maps and visualizations

### 🛠 **Modern Architecture**
- Type-safe configuration with Pydantic models
- Comprehensive error handling and logging
- Progress tracking with rich console output
- Resource management with automatic cleanup
- Extensible processor architecture

### 📊 **Supported Data Types**
- **Raster Data**: GeoTIFF, NetCDF, HDF5
- **Vector Data**: Shapefile, GeoJSON, GeoPackage, KML
- **Archives**: ZIP, TAR, compressed formats
- **Coordinate Systems**: Geographic and projected CRS with automatic reprojection

### 🎯 **Built-in Data Source Support**
- Copernicus Land Cover classification
- ODIAC fossil fuel emissions
- PM2.5 air quality concentrations
- AFRICAPOLIS urban settlement data
- Landsat and MODIS satellite imagery
- Custom dataset integration

## Installation

### Environment Setup (Recommended)

GeoWorkflow works best with conda for managing geospatial dependencies:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate geoworkflow

# Install the package
pip install -e .
```

### Alternative Installation

For pip-only installation (requires system GDAL):

```bash
pip install geoworkflow[geospatial]
```

### Development Installation

```bash
git clone https://github.com/jacksonfloods/geoworkflow.git
cd geoworkflow
conda env create -f environment.yml
conda activate geoworkflow
pip install -e ".[dev]"
```

### System Requirements

- Python 3.8+
- GDAL 3.5+
- Sufficient disk space for geospatial datasets (typically 10GB+ per workflow)

## Quick Start

### Command Line Usage

```bash
# Create an Area of Interest for Southern Africa
geoworkflow aoi create \
  --input-file data/africa_boundaries.geojson \
  --countries "Angola,Namibia,Botswana" \
  --buffer 100 \
  --output data/aoi/southern_africa_aoi.geojson

# Extract and clip data from archives
geoworkflow extract archives \
  --source data/source_archives/ \
  --aoi data/aoi/southern_africa_aoi.geojson \
  --output data/clipped/

# Align rasters to common grid
geoworkflow process align \
  --input data/clipped/ \
  --reference data/clipped/copernicus/reference.tif \
  --output data/aligned/

# Enrich cities with raster statistics
geoworkflow process enrich \
  --coi-dir data/clipped/ \
  --raster-dir data/aligned/ \
  --output data/enriched_cities.geojson

# Create visualizations
geoworkflow visualize rasters \
  --input data/aligned/ \
  --output outputs/visualizations/
```

### Python API Usage

```python
from geoworkflow import AOIProcessor, AOIConfig
from pathlib import Path

# Create AOI configuration
aoi_config = AOIConfig(
    input_file=Path("data/africa_boundaries.geojson"),
    country_name_column="NAME_0",
    countries=["Angola", "Namibia", "Botswana"],
    buffer_km=100,
    output_file=Path("data/aoi/southern_africa_aoi.geojson")
)

# Process AOI
processor = AOIProcessor(aoi_config)
result = processor.process()

if result.success:
    print(f"AOI created successfully: {result.output_paths[0]}")
else:
    print(f"Processing failed: {result.message}")
```

## Core Concepts

### Processing Stages

GeoWorkflow organizes data processing into logical stages:

1. **Source Data** (`00_source`): Raw archives and boundary files
2. **Extracted Data** (`01_extracted`): Files extracted from archives
3. **Clipped Data** (`02_clipped`): Data clipped to Areas of Interest
4. **Processed Data** (`03_processed`): Aligned and standardized datasets
5. **Analysis-Ready Data** (`04_analysis_ready`): Final enriched datasets

### Processors

Each processing operation is handled by a specialized processor:

- **AOIProcessor**: Creates Areas of Interest from administrative boundaries
- **ArchiveExtractionProcessor**: Extracts and clips data from archives
- **ClippingProcessor**: Clips raster and vector data to AOI boundaries
- **AlignmentProcessor**: Aligns rasters to reference grids
- **StatisticalEnrichmentProcessor**: Enriches vector data with raster statistics
- **VisualizationProcessor**: Creates maps and visualizations

### Configuration System

GeoWorkflow uses Pydantic models for type-safe configuration:

```python
from geoworkflow.schemas.config_models import ClippingConfig

config = ClippingConfig(
    input_directory=Path("data/extracted/"),
    aoi_file=Path("data/aoi/study_area.geojson"),
    output_dir=Path("data/clipped/"),
    all_touched=True,
    create_visualizations=True
)
```

## Processing Workflow

### Standard Workflow Pipeline

```python
from geoworkflow.core.pipeline import ProcessingPipeline
from geoworkflow.schemas.config_models import WorkflowConfig

# Load workflow configuration
config = WorkflowConfig.from_yaml("config/southern_africa_workflow.yml")

# Create and run pipeline
pipeline = ProcessingPipeline(config)
result = pipeline.run()

# Check results
for stage, stage_result in result.stage_results.items():
    print(f"{stage}: {'✓' if stage_result.success else '✗'}")
```

### Individual Stage Processing

Each stage can be run independently:

```python
# Extract data from archives
from geoworkflow.processors.extraction.archive import ArchiveExtractionProcessor

processor = ArchiveExtractionProcessor(extraction_config)
result = processor.process()

# Clip extracted data
from geoworkflow.processors.spatial.clipper import ClippingProcessor

clipper = ClippingProcessor(clipping_config)
clipped_result = clipper.process()
```

## Command Line Interface

### AOI Operations

```bash
# Create AOI from countries
geoworkflow aoi create \
  --input-file data/africa_boundaries.geojson \
  --countries "Kenya,Tanzania,Uganda" \
  --buffer 50 \
  --output data/aoi/east_africa.geojson

# List available countries
geoworkflow aoi list-countries \
  --boundaries-file data/africa_boundaries.geojson \
  --prefix "South"

# Validate AOI file
geoworkflow aoi validate data/aoi/study_area.geojson
```

### Data Extraction

```bash
# Extract from ZIP archives
geoworkflow extract archives \
  --source data/archives/ \
  --output data/extracted/ \
  --pattern "*.tif"

# Convert NetCDF to GeoTIFF
geoworkflow extract netcdf \
  input_file.nc \
  --variable PM25 \
  --output output.tiff \
  --crs EPSG:4326
```

### Processing Operations

```bash
# Clip data to AOI
geoworkflow process clip \
  --input data/extracted/ \
  --aoi data/aoi/study_area.geojson \
  --output data/clipped/ \
  --all-touched

# Align rasters
geoworkflow process align \
  --input data/clipped/ \
  --reference data/clipped/copernicus_lc.tif \
  --output data/aligned/ \
  --method cubic

# Statistical enrichment
geoworkflow process enrich \
  --coi-dir data/clipped/ \
  --coi-pattern "*AFRICAPOLIS*" \
  --raster-dir data/aligned/ \
  --output data/enriched_cities.geojson \
  --statistics "mean,max,min,std"
```

### Visualization

```bash
# Create raster visualizations
geoworkflow visualize rasters \
  --input data/aligned/ \
  --output outputs/maps/ \
  --colormap plasma \
  --dpi 300

# Use custom configuration
geoworkflow visualize rasters \
  --input data/aligned/ \
  --output outputs/maps/ \
  --config config/visualization.yml
```

### Pipeline Operations

```bash
# Run complete workflow
geoworkflow pipeline run \
  --workflow config/southern_africa_workflow.yml

# Run from specific stage
geoworkflow pipeline run \
  --workflow config/workflow.yml \
  --from-stage clip

# Check pipeline status
geoworkflow pipeline status \
  --workflow config/workflow.yml

# Resume pipeline
geoworkflow pipeline resume \
  --workflow config/workflow.yml \
  --from-stage align
```

## Python API Usage

### Basic Processing

```python
from geoworkflow.processors.aoi.processor import AOIProcessor
from geoworkflow.schemas.config_models import AOIConfig
from pathlib import Path

# Configure AOI creation
config = AOIConfig(
    input_file=Path("data/africa_boundaries.geojson"),
    country_name_column="NAME_0",
    countries=["Morocco", "Algeria", "Tunisia"],
    dissolve_boundaries=True,
    buffer_km=50,
    output_file=Path("data/aoi/north_africa.geojson")
)

# Create processor and run
processor = AOIProcessor(config)
result = processor.process()

# Check results
if result.success:
    print(f"Created AOI with {result.metadata['feature_count']} features")
    print(f"Processing time: {result.elapsed_time:.2f} seconds")
else:
    print(f"Failed: {result.message}")
```

### Statistical Enrichment

```python
from geoworkflow.processors.integration.enrichment import StatisticalEnrichmentProcessor
from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig

# Configure enrichment
config = StatisticalEnrichmentConfig(
    coi_directory=Path("data/clipped/"),
    coi_pattern="*AFRICAPOLIS*",
    raster_directory=Path("data/aligned/"),
    raster_pattern="*.tif",
    output_file=Path("data/enriched_cities.geojson"),
    statistics=["mean", "max", "min", "std"],
    add_area_column=True,
    area_units="km2"
)

# Run enrichment
processor = StatisticalEnrichmentProcessor(config)
result = processor.process()

print(f"Enriched {result.metadata['original_features']} cities")
print(f"Added {result.metadata['new_columns_added']} statistical columns")
```

### Batch Processing

```python
from geoworkflow.processors.spatial.clipper import ClippingProcessor
from geoworkflow.schemas.config_models import ClippingConfig

# Process multiple AOIs
aoi_files = [
    "data/aoi/west_africa.geojson",
    "data/aoi/east_africa.geojson", 
    "data/aoi/southern_africa.geojson"
]

for aoi_file in aoi_files:
    region_name = Path(aoi_file).stem
    
    config = ClippingConfig(
        input_directory=Path("data/extracted/"),
        aoi_file=Path(aoi_file),
        output_dir=Path(f"data/clipped/{region_name}/"),
        create_visualizations=True
    )
    
    processor = ClippingProcessor(config)
    result = processor.process()
    
    print(f"{region_name}: {result.processed_count} files processed")
```

## Configuration

### Configuration Files

GeoWorkflow supports YAML and JSON configuration files:

```yaml
# config/southern_africa_workflow.yml
name: "Southern Africa Analysis"
description: "Land cover and emissions analysis for Southern Africa"

stages: ["extract", "clip", "align", "enrich", "visualize"]

aoi:
  input_file: "data/africa_boundaries.geojson"
  country_name_column: "NAME_0"
  countries: ["Angola", "Namibia", "Botswana"]
  buffer_km: 100
  output_file: "data/aoi/southern_africa_aoi.geojson"

extraction:
  zip_folder: "data/source_archives/"
  aoi_file: "data/aoi/southern_africa_aoi.geojson"
  output_dir: "data/extracted/"
  create_visualizations: true

clipping:
  input_directory: "data/extracted/"
  aoi_file: "data/aoi/southern_africa_aoi.geojson"
  output_dir: "data/clipped/"
  all_touched: true

alignment:
  input_directory: "data/clipped/"
  output_dir: "data/aligned/"
  resampling_method: "cubic"
  recursive: true

enrichment:
  coi_directory: "data/clipped/"
  coi_pattern: "*AFRICAPOLIS*"
  raster_directory: "data/aligned/"
  output_file: "data/enriched_cities.geojson"
  statistics: ["mean", "max", "min", "std"]
  add_area_column: true

visualization:
  input_directory: "data/aligned/"
  output_dir: "outputs/visualizations/"
  colormap: "viridis"
  dpi: 300
```

### Configuration Templates

Generate configuration templates:

```bash
# Create AOI configuration template
geoworkflow config --template aoi --output config/aoi_template.yml

# Create complete workflow template
geoworkflow config --template pipeline --output config/workflow_template.yml
```

### Environment Variables

```bash
export GEOWORKFLOW_DATA_DIR="/path/to/data"
export GEOWORKFLOW_LOG_LEVEL="INFO"
export GEOWORKFLOW_TEMP_DIR="/tmp/geoworkflow"
```

## Data Sources

### Supported Datasets

#### Land Cover Data
- **Copernicus Global Land Cover**: Annual land cover classifications
- **ESA WorldCover**: 10m resolution global land cover
- **MODIS Land Cover**: 500m annual land cover products

#### Atmospheric Data
- **ODIAC**: Fossil fuel CO₂ emissions (1km resolution)
- **PM2.5 Concentrations**: Ground-level particulate matter
- **NO₂ Concentrations**: Nitrogen dioxide levels

#### Urban Data
- **AFRICAPOLIS**: Urban agglomeration boundaries and population
- **Global Human Settlement Layer**: Built-up area and population density
- **OpenStreetMap**: Road networks and infrastructure

#### Administrative Data
- **GADM**: Administrative boundaries (levels 0-3)
- **Natural Earth**: Physical and cultural vector data
- **African Union**: Official country boundaries

### Data Organization

```
data/
├── 00_source/
│   ├── archives/           # ZIP files containing datasets
│   └── boundaries/         # Administrative boundary files
├── 01_extracted/          # Files extracted from archives
├── 02_clipped/           # Data clipped to AOIs
├── 03_processed/         # Aligned and standardized data
└── 04_analysis_ready/    # Final enriched datasets
```

### Custom Data Integration

```python
from geoworkflow.core.constants import DataSource

# Register custom data source
class CustomProcessor(BaseProcessor):
    def _detect_data_source(self, file_path):
        if "my_dataset" in file_path.name.lower():
            return DataSource.OTHER
        return super()._detect_data_source(file_path)
```

## Examples

### Example 1: Urban Air Quality Analysis

```python
from geoworkflow.core.pipeline import ProcessingPipeline
from geoworkflow.schemas.config_models import WorkflowConfig

# Configure workflow for air quality analysis
config = WorkflowConfig(
    name="Urban Air Quality Analysis",
    stages=["extract", "clip", "align", "enrich"],
    aoi=AOIConfig(
        countries=["Nigeria", "Ghana", "Senegal"],
        buffer_km=25
    ),
    enrichment=StatisticalEnrichmentConfig(
        coi_pattern="*AFRICAPOLIS*",
        statistics=["mean", "max", "min", "std"],
        add_area_column=True
    )
)

# Run analysis
pipeline = ProcessingPipeline(config)
result = pipeline.run()

# Results available in data/04_analysis_ready/
```

### Example 2: Land Cover Change Analysis

```bash
# Extract land cover data for multiple years
geoworkflow extract archives \
  --source data/copernicus_lc_2016_2020/ \
  --aoi data/aoi/study_area.geojson \
  --output data/land_cover/

# Align all years to common grid
geoworkflow process align \
  --input data/land_cover/ \
  --reference data/land_cover/2020/copernicus_lc_2020.tif \
  --output data/land_cover/aligned/

# Create change detection visualizations
geoworkflow visualize rasters \
  --input data/land_cover/aligned/ \
  --output outputs/land_cover_maps/ \
  --config config/land_cover_viz.yml
```

### Example 3: Multi-Region Comparison

```python
import pandas as pd
from geoworkflow.processors.integration.enrichment import enrich_cities_with_statistics

regions = {
    "west_africa": ["Nigeria", "Ghana", "Burkina Faso"],
    "east_africa": ["Kenya", "Tanzania", "Uganda"],
    "southern_africa": ["South Africa", "Botswana", "Zambia"]
}

results = {}

for region_name, countries in regions.items():
    # Create region-specific AOI
    aoi_config = AOIConfig(
        countries=countries,
        buffer_km=50,
        output_file=f"data/aoi/{region_name}.geojson"
    )
    
    # Process region
    success = enrich_cities_with_statistics(
        coi_directory=f"data/clipped/{region_name}/",
        raster_directory=f"data/aligned/{region_name}/",
        output_file=f"data/enriched_{region_name}.geojson"
    )
    
    if success:
        # Load results for comparison
        gdf = gpd.read_file(f"data/enriched_{region_name}.geojson")
        results[region_name] = gdf

# Combine results for comparative analysis
combined_df = pd.concat(results.values(), keys=results.keys())
```

## Advanced Usage

### Custom Processors

```python
from geoworkflow.core.enhanced_base import TemplateMethodProcessor

class CustomDataProcessor(TemplateMethodProcessor):
    def _validate_custom_inputs(self):
        # Custom validation logic
        return {"valid": True, "errors": [], "warnings": []}
    
    def process_data(self):
        # Main processing logic
        result = ProcessingResult(success=True)
        # ... processing code ...
        return result
    
    def _cleanup_custom_processing(self):
        # Cleanup logic
        return {"cleanup_completed": True}
```

### Pipeline Customization

```python
from geoworkflow.core.pipeline import ProcessingPipeline

class CustomPipeline(ProcessingPipeline):
    def _create_stage_config(self, stage_name, **kwargs):
        config = super()._create_stage_config(stage_name, **kwargs)
        
        # Add custom configuration
        if stage_name == "custom_analysis":
            config.update({
                "analysis_parameters": self.custom_params,
                "output_format": "netcdf"
            })
        
        return config
```

### Error Handling and Logging

```python
from geoworkflow.core.logging_setup import setup_logging
from geoworkflow.core.exceptions import ProcessingError

# Configure detailed logging
setup_logging(level="DEBUG", log_file="logs/processing.log")

try:
    processor = ClippingProcessor(config)
    result = processor.process()
    
    if not result.success:
        print(f"Processing failed: {result.message}")
        for failed_file in result.failed_files:
            print(f"  Failed: {failed_file}")
            
except ProcessingError as e:
    print(f"Processing error: {e.message}")
    if e.details:
        print(f"Details: {e.details}")
```

### Performance Monitoring

```python
from geoworkflow.utils.progress_utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Process data
for file in large_file_list:
    process_file(file)
    monitor.checkpoint(f"processed_{file.name}")

# Get performance summary
summary = monitor.get_summary()
print(f"Total time: {summary['total_time']:.2f}s")
print(f"Peak memory: {summary.get('peak_memory_mb', 'N/A')} MB")
```

## API Reference

For complete API documentation, see [API_REFERENCE.md](docs/API_REFERENCE.md).

### Core Classes

- **BaseProcessor**: Abstract base for all processors
- **ProcessingResult**: Standard result object with metrics
- **ProcessingPipeline**: Orchestrates multi-stage workflows
- **ConfigManager**: Handles configuration loading and validation

### Processors

- **AOIProcessor**: Creates Areas of Interest
- **ArchiveExtractionProcessor**: Extracts data from archives
- **ClippingProcessor**: Spatial clipping operations
- **AlignmentProcessor**: Raster alignment and resampling
- **StatisticalEnrichmentProcessor**: Zonal statistics computation
- **VisualizationProcessor**: Map and chart generation

### Configuration Models

- **AOIConfig**: AOI creation parameters
- **ExtractionConfig**: Archive extraction settings
- **ClippingConfig**: Spatial clipping options
- **AlignmentConfig**: Raster alignment parameters
- **StatisticalEnrichmentConfig**: Statistical analysis settings
- **VisualizationConfig**: Visualization parameters
- **WorkflowConfig**: Complete workflow definition

### Utilities

- **ProgressTracker**: Progress monitoring with rich output
- **ResourceManager**: Temporary file and directory management
- **ProcessingMetrics**: Performance and resource tracking

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/jacksonfloods/geoworkflow.git
cd geoworkflow
conda env create -f environment.yml
conda activate geoworkflow
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_aoi.py -v
pytest tests/test_clipping.py -k "test_raster_clipping"

# Run with coverage
pytest --cov=geoworkflow --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check types
mypy src/geoworkflow

# Lint code
flake8 src/ tests/

## API Documentation

GeoWorkflow uses `pydoc-markdown` to automatically generate comprehensive API documentation from Python docstrings. This ensures that the documentation stays synchronized with the codebase and provides detailed information about all classes, methods, and functions.

### Generating API Documentation

#### Prerequisites

Install `pydoc-markdown` if not already available:

```bash
pip install pydoc-markdown
```

#### Using the Configuration File

The project includes a pre-configured `pydoc-markdown.yaml` file that defines which modules to document and how to format the output:

```bash
# Generate API documentation using the project configuration
pydoc-markdown

# Or specify the config file explicitly
pydoc-markdown --config pydoc-markdown.yaml
```

This will generate the complete API reference at `docs/API_REFERENCE.md`.

#### Configuration Overview

The `pydoc-markdown.yaml` configuration includes:

- **Modules to document**: Core modules, processors, utilities, and configuration models
- **Filtering**: Excludes private methods and internal implementation details
- **Cross-references**: Links between related classes and methods
- **Output format**: Clean Markdown suitable for GitHub and documentation sites

#### Manual Generation Commands

For specific modules or custom output:

```bash
# Document specific modules
pydoc-markdown -m geoworkflow.core.base -m geoworkflow.processors.aoi.processor

# Generate with custom output file
pydoc-markdown --config pydoc-markdown.yaml --render-toc --output docs/custom_api.md

# Generate HTML documentation
pydoc-markdown --config pydoc-markdown.yaml --output docs/api.html --renderer html
```

### Customizing Documentation

#### Modifying the Configuration

Edit `pydoc-markdown.yaml` to customize documentation generation:

```yaml
# Add new modules to document
loaders:
  - type: python
    search_path: [src]
    modules: 
      - geoworkflow.core.base
      - geoworkflow.processors.custom.processor  # Add your module here
    ignore_when_discovered: ['__pycache__']

# Customize filtering
processors:
  - type: filter
    expression: not name.startswith('_') and default()  # Exclude private methods
  - type: smart
  - type: crossref

# Change output format or location
renderer:
  type: markdown
  filename: docs/API_REFERENCE.md  # Customize output path
```

#### Documentation Standards

For best results, follow these docstring conventions:

```python
class ExampleProcessor:
    """
    Brief description of the processor.
    
    Longer description explaining the purpose, key features,
    and usage patterns. This appears in the generated docs.
    
    Example:
        ```python
        processor = ExampleProcessor(config)
        result = processor.process()
        ```
    """
    
    def process(self, input_data: Path) -> ProcessingResult:
        """
        Execute the main processing logic.
        
        Args:
            input_data: Path to input data file
            
        Returns:
            ProcessingResult with operation outcomes
            
        Raises:
            ProcessingError: If processing fails
            ValidationError: If input validation fails
            
        Example:
            ```python
            result = processor.process(Path("data/input.tif"))
            if result.success:
                print(f"Processed {result.processed_count} files")
            ```
        """
        pass
```

### Integration with Development Workflow

#### Pre-commit Hook

Add documentation generation to your pre-commit workflow:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: generate-docs
        name: Generate API Documentation
        entry: pydoc-markdown
        language: system
        files: ^src/.*\.py$
        pass_filenames: false
```

#### GitHub Actions

Automate documentation updates with GitHub Actions:

```yaml
# .github/workflows/docs.yml
name: Update Documentation
on:
  push:
    branches: [main]
    paths: ['src/**/*.py']

jobs:
  update-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install pydoc-markdown
          
      - name: Generate API documentation
        run: pydoc-markdown
        
      - name: Commit updated docs
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/API_REFERENCE.md
          git diff --staged --quiet || git commit -m "Update API documentation"
          git push
```

#### Development Script

Create a development script for easy documentation updates:

```bash
#!/bin/bash
# scripts/update_docs.sh

echo "🔄 Updating API documentation..."

# Generate API docs
pydoc-markdown

# Validate the output
if [ -f "docs/API_REFERENCE.md" ]; then
    echo "✅ API documentation generated successfully"
    
    # Count documented modules
    module_count=$(grep -c "^# geoworkflow\." docs/API_REFERENCE.md)
    echo "📚 Documented $module_count modules"
    
    # Show file size
    size=$(stat -f%z docs/API_REFERENCE.md 2>/dev/null || stat -c%s docs/API_REFERENCE.md)
    echo "📄 Documentation size: $((size / 1024)) KB"
else
    echo "❌ Failed to generate API documentation"
    exit 1
fi

echo "✨ Documentation update complete!"
```

Make it executable and run:

```bash
chmod +x scripts/update_docs.sh
./scripts/update_docs.sh
```

### Advanced Documentation Features

#### Custom Renderers

For specialized output formats, create custom renderers:

```python
# custom_renderer.py
from pydoc_markdown.interfaces import Renderer

class CustomRenderer(Renderer):
    def render(self, modules):
        # Custom rendering logic
        for module in modules:
            # Process module documentation
            pass
```

Use in configuration:

```yaml
renderer:
  type: custom_renderer.CustomRenderer
  output_file: docs/custom_output.md
```

#### Documentation Validation

Validate documentation completeness:

```bash
# Check for undocumented public methods
python -c "
import ast
import glob

for file in glob.glob('src/**/*.py', recursive=True):
    with open(file) as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            if not ast.get_docstring(node):
                print(f'{file}:{node.lineno}: Missing docstring for {node.name}')
"
```

#### Multi-format Output

Generate documentation in multiple formats:

```bash
# Generate both Markdown and HTML
pydoc-markdown --renderer markdown --output docs/api.md
pydoc-markdown --renderer html --output docs/api.html

# Generate JSON for programmatic use
pydoc-markdown --renderer json --output docs/api.json
```

### Troubleshooting

#### Common Issues

**Import Errors:**
```bash
# Ensure modules can be imported
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pydoc-markdown
```

**Missing Dependencies:**
```bash
# Install optional dependencies
pip install pydoc-markdown[all]
```

**Large Documentation:**
```yaml
# Limit documentation scope
processors:
  - type: filter
    expression: |
      (default() and not name.startswith('_') 
       and 'internal' not in name.lower())
```

#### Debugging Configuration

Test configuration without generating output:

```bash
# Validate configuration
pydoc-markdown --config pydoc-markdown.yaml --dry-run

# Verbose output for debugging
pydoc-markdown --config pydoc-markdown.yaml --verbose
```

### Documentation Maintenance

#### Regular Updates

Set up automated documentation updates:

```bash
# Weekly documentation refresh
# Add to crontab: 0 0 * * 0 /path/to/update_docs.sh

# Check for documentation drift
git diff --name-only HEAD~1 HEAD | grep "\.py$" | xargs -I {} echo "Module {} changed, update docs"
```

#### Quality Checks

Implement documentation quality metrics:

```python
# check_docs.py
import re
from pathlib import Path

def check_documentation_completeness():
    api_doc = Path("docs/API_REFERENCE.md").read_text()
    
    # Count documented classes
    classes = len(re.findall(r"^## \w+ Objects", api_doc, re.MULTILINE))
    
    # Count documented methods
    methods = len(re.findall(r"^#### \w+", api_doc, re.MULTILINE))
    
    print(f"📊 Documentation stats:")
    print(f"   Classes documented: {classes}")
    print(f"   Methods documented: {methods}")
    print(f"   Total lines: {len(api_doc.splitlines())}")

if __name__ == "__main__":
    check_documentation_completeness()
```

The automated API documentation system ensures that GeoWorkflow's documentation remains comprehensive, accurate, and synchronized with the codebase, making it easier for users and developers to understand and use the package effectively.```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
