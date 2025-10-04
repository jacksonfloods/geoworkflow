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
