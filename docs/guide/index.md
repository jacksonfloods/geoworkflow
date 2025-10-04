# GeoWorkflow Documentation

Welcome to the **GeoWorkflow** project documentation! This guide helps you understand, use, and contribute to the AfricaPolis geospatial data processing pipeline.

## What is GeoWorkflow?

GeoWorkflow is a Python-based pipeline for processing geospatial data, specifically designed for the AfricaPolis project. It handles the complete workflow from raw data extraction to analysis-ready enriched datasets.

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation, configuration, and quick start guides
- **[Project Guide](guide/structure.md)** - Understand the codebase structure and architecture
- **[Tutorials](tutorials/basic-workflow.md)** - Step-by-step examples and how-tos
- **[API Reference](https://your-sphinx-url/)** - Detailed API documentation (via Sphinx)
- **[Development](development/contributing.md)** - Contributing guidelines and testing

## Quick Links

- 📦 [GitHub Repository](https://github.com/jacksonfloods/geoworkflow)
- 📚 [API Documentation](https://your-sphinx-url/) (Sphinx)
- 🐛 [Issue Tracker](https://github.com/jacksonfloods/geoworkflow/issues)

## Key Features

- **Multi-stage Processing Pipeline** - Extract, clip, align, enrich, visualize
- **Modular Processors** - Extensible architecture for custom processors
- **Configuration-driven** - YAML-based workflow definitions
- **Progress Tracking** - Rich console output with detailed metrics
- **Error Handling** - Comprehensive error reporting and recovery

## At a Glance
```python
from geoworkflow.core.pipeline import ProcessingPipeline

# Define workflow configuration
config = {
    "stages": ["clip", "align", "enrich"],
    "source_dir": "data/raw",
    "output_dir": "data/processed"
}

# Run pipeline
pipeline = ProcessingPipeline(config)
results = pipeline.run()

## Documentation Structure

- **[API Reference](https://your-sphinx-url/)** - Complete API docs (Sphinx)