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
