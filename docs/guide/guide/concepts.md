# Core Concepts

Understanding these key concepts will help you work effectively with GeoWorkflow.

## Processors

Processors are the building blocks of the workflow. Each processor handles a specific transformation:

- **BaseProcessor** - Abstract base class all processors inherit from
- **AOIProcessor** - Creates Areas of Interest from various sources
- **ClippingProcessor** - Spatially clips datasets to boundaries
- **AlignmentProcessor** - Ensures consistent CRS and resolution
- **StatisticalEnrichmentProcessor** - Calculates zonal statistics

## Processing Results

Every processor returns a standardized `ProcessingResult` object:
```python
@dataclass
class ProcessingResult:
    success: bool
    processed_count: int
    failed_count: int
    elapsed_time: float
    output_paths: List[Path]
    metadata: Dict[str, Any]