# GCS Buildings Batch Processing Implementation Guide

## Project Context

The geoworkflow project has two building extraction processors:
1. **OSMHighwaysProcessor** - Extracts highways from OpenStreetMap (ALREADY HAS BATCH PROCESSING)
2. **OpenBuildingsGCSProcessor** - Extracts buildings from Google Open Buildings via GCS (NEEDS BATCH PROCESSING)

## Goal

Add batch processing capability to `OpenBuildingsGCSProcessor` that mirrors the working implementation in `OSMHighwaysProcessor`.

## What is Batch Processing?

Instead of processing one city at a time by providing a single AOI file, batch mode:
- Uses `aoi_file="africapolis"` as a trigger
- Loads multiple city boundaries from the AFRICAPOLIS dataset
- Filters by `country` (ISO3 codes like `["GHA", "TGO"]`) and optionally by `city` names
- Processes each city individually and exports separate files
- Returns `BatchProcessResult` with succeeded/failed city lists

## Working Reference: OSM Highways

The OSM highways processor already has this working. Key files:
- **Config**: `src/geoworkflow/schemas/osm_highways_config.py`
- **Processor**: `src/geoworkflow/processors/extraction/osm_highways.py`

### Example Usage (OSM - Already Working)
```python
config = OSMHighwaysConfig(
    aoi_file="africapolis",  # Batch mode trigger
    output_dir=Path("./highways/"),
    country=["GHA", "TGO"],  # Filter countries
    city=["Accra", "Kumasi"],  # Optional: filter specific cities
    highway_types="all"
)

processor = OSMHighwaysProcessor(config)
result = processor.process()  # Returns BatchProcessResult
```

## Required Changes

### File 1: `open_buildings_gcs_config.py` (Config Schema)

#### Change 1.1: Update aoi_file Field Type
**Location**: Line ~27

**Current**:
```python
aoi_file: Path = Field(
    ...,
    description="Area of Interest boundary file (GeoJSON, Shapefile, etc.)"
)
```

**Change to**:
```python
aoi_file: Union[Path, str] = Field(
    ...,
    description="Area of Interest boundary file (GeoJSON, Shapefile, etc.) or 'africapolis' for batch mode"
)
```

#### Change 1.2: Add Imports
**Location**: Top of file

**Current**:
```python
from typing import Optional, Literal
```

**Change to**:
```python
from typing import Optional, Literal, Union, List
```

#### Change 1.3: Add Batch Processing Fields
**Location**: After `aoi_file` field (around line 33)

**Insert**:
```python
# ==================== Batch Processing (AFRICAPOLIS mode) ====================
country: Optional[Union[List[str], str]] = Field(
    default=None,
    description="ISO3 country code(s) for batch processing. Use 'all' for all countries, "
                "list of codes like ['GHA', 'TGO'], or None for single AOI mode."
)

city: Optional[List[str]] = Field(
    default=None,
    description="Optional list of city names to filter when using AFRICAPOLIS batch mode. "
                "If None, processes all cities in specified countries."
)
```

#### Change 1.4: Replace aoi_file Validator
**Location**: Validators section (around line 143)

**Remove** existing `@field_validator('aoi_file')` validators (there may be 1 or 2)

**Replace with** (copy from OSM config):
```python
@field_validator('aoi_file')
@classmethod
def validate_aoi(cls, v):
    """Convert string to Path if not AfricaPolis keyword."""
    if isinstance(v, str) and v.lower() == "africapolis":
        return v.lower()  # Normalize to lowercase
    
    # Otherwise treat as path
    path = Path(v) if isinstance(v, str) else v
    if not path.exists():
        raise ValueError(f"AOI file not found: {path}")
    return path
```

### File 2: `open_buildings_gcs.py` (Processor)

#### Change 2.1: Update _validate_custom_inputs Method
**Location**: Find the `_validate_custom_inputs` method

**Current** (approximately):
```python
def _validate_custom_inputs(self) -> Dict[str, Any]:
    validation_result = {...}
    
    # Validates AOI file exists
    if not self.gcs_config.aoi_file.exists():
        validation_result["errors"].append(...)
        
    return validation_result
```

**Change to**:
```python
def _validate_custom_inputs(self) -> Dict[str, Any]:
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Skip AOI file validation if in africapolis mode
    if isinstance(self.gcs_config.aoi_file, str) and self.gcs_config.aoi_file == "africapolis":
        validation_result["info"] = {"mode": "africapolis_batch"}
        self.logger.info("Africapolis batch mode detected")
    else:
        # Validate AOI file exists for single-file mode only
        if not self.gcs_config.aoi_file.exists():
            validation_result["errors"].append(
                f"Path does not exist: aoi_file = {self.gcs_config.aoi_file}"
            )
            validation_result["valid"] = False
    
    # ... rest of validation ...
    
    return validation_result
```

#### Change 2.2: Add Batch Processing Methods
**Location**: Before `process_data` method

**Add these methods** (copy/adapt from OSM processor):

```python
def _is_africapolis_mode(self) -> bool:
    """Check if processor is in AFRICAPOLIS batch mode."""
    return (isinstance(self.gcs_config.aoi_file, str) and 
            self.gcs_config.aoi_file == "africapolis")

def _load_batch_geometries(self):
    """Load agglomerations from AFRICAPOLIS dataset."""
    from geoworkflow.utils.config_loader import ConfigLoader
    
    agglo_path = ConfigLoader.get_africapolis_path()
    columns = ConfigLoader.get_africapolis_columns()
    
    if not agglo_path.exists():
        raise FileNotFoundError(f"AFRICAPOLIS file not found: {agglo_path}")
    
    agglomerations = gpd.read_file(agglo_path)
    self.aoi_crs = agglomerations.crs
    
    filtered = self._filter_agglomerations(agglomerations, columns)
    
    # Return list of (geometry, name, iso3)
    return [
        (row.geometry, row[columns["name"]], row[columns["iso3"]]) 
        for _, row in filtered.iterrows()
    ]

def _load_single_geometry(self):
    """Load single AOI geometry from file."""
    aoi_gdf = gpd.read_file(self.gcs_config.aoi_file)
    self.aoi_crs = aoi_gdf.crs
    name = self.gcs_config.aoi_file.stem
    country_code = "UNKNOWN"
    return [(aoi_gdf.union_all(), name, country_code)]

def _filter_agglomerations(self, gdf: gpd.GeoDataFrame, columns: dict) -> gpd.GeoDataFrame:
    """Filter agglomerations by country and city."""
    filtered = gdf.copy()
    
    # Filter by country
    if self.gcs_config.country:
        if isinstance(self.gcs_config.country, str):
            if self.gcs_config.country.lower() == "all":
                pass
            else:
                filtered = filtered[filtered[columns["iso3"]] == self.gcs_config.country]
        elif isinstance(self.gcs_config.country, list):
            filtered = filtered[filtered[columns["iso3"]].isin(self.gcs_config.country)]
    
    # Filter by city
    if self.gcs_config.city:
        filtered = filtered[filtered[columns["name"]].isin(self.gcs_config.city)]
    
    if len(filtered) == 0:
        raise ValueError(
            f"No agglomerations match filters: "
            f"country={self.gcs_config.country}, city={self.gcs_config.city}"
        )
    
    return filtered

def _process_batch(self, geometries):
    """Process multiple geometries in batch mode."""
    from geoworkflow.schemas.processing_result import BatchProcessResult
    
    succeeded = []
    failed = {}
    all_output_files = []
    
    for geom, name, country_code in geometries:
        try:
            # Store original config
            original_aoi = self.gcs_config.aoi_file
            original_output_dir = self.gcs_config.output_dir
            
            # Create temp AOI for this city
            temp_aoi = self.resource_manager.create_temp_file(suffix=".geojson")
            temp_gdf = gpd.GeoDataFrame([{"name": name}], geometry=[geom], crs=self.aoi_crs)
            temp_gdf.to_file(temp_aoi, driver="GeoJSON")
            
            # Update config
            self.gcs_config.aoi_file = temp_aoi
            city_output_dir = original_output_dir / f"{name.replace(' ', '_').lower()}_buildings"
            city_output_dir.mkdir(parents=True, exist_ok=True)
            self.gcs_config.output_dir = city_output_dir
            
            # Process using original single-city logic
            result = self._process_single_aoi()
            
            # Restore config
            self.gcs_config.aoi_file = original_aoi
            self.gcs_config.output_dir = original_output_dir
            
            if result.success:
                succeeded.append(name)
                if result.output_paths:
                    all_output_files.extend(result.output_paths)
            else:
                failed[name] = result.message
                
        except Exception as e:
            failed[name] = str(e)
    
    return BatchProcessResult(
        success=len(succeeded) > 0,
        total_count=len(geometries),
        succeeded_count=len(succeeded),
        failed_count=len(failed),
        succeeded=succeeded,
        failed=failed,
        output_files=all_output_files
    )
```

#### Change 2.3: Update process_data Method
**Location**: The existing `process_data` method

**Strategy**: 
1. Rename current `process_data` to `_process_single_aoi`
2. Create new `process_data` that routes to batch or single mode

**New process_data**:
```python
def process_data(self):
    """
    Execute building extraction.
    
    Handles both single-city and batch processing modes.
    """
    try:
        if self._is_africapolis_mode():
            self.logger.info("Running in AFRICAPOLIS batch mode")
            geometries = self._load_batch_geometries()
            return self._process_batch(geometries)
        else:
            self.logger.info("Running in single-city mode")
            return self._process_single_aoi()
            
    except Exception as e:
        self.logger.error(f"Processing failed: {e}", exc_info=True)
        return ProcessingResult(
            success=False,
            message=f"Processing failed: {str(e)}"
        )

def _process_single_aoi(self):
    """Process single AOI - original process_data logic."""
    # [MOVE ALL ORIGINAL process_data CODE HERE]
```

## Testing

### Test 1: Verify Config Accepts Africapolis Mode
```python
from pathlib import Path
from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig

# Should NOT raise validation error
config = OpenBuildingsGCSConfig(
    aoi_file="africapolis",
    output_dir=Path("./test/"),
    country=["GHA"]
)
print("✓ Config validation passed")
```

### Test 2: Verify Processor Initializes
```python
from geoworkflow.processors.extraction.open_buildings_gcs import OpenBuildingsGCSProcessor

processor = OpenBuildingsGCSProcessor(config)
print("✓ Processor initialized")
```

### Test 3: Run Small Batch
```python
# Use notebook: notebooks/agglomerations_buildings.ipynb
config = OpenBuildingsGCSConfig(
    aoi_file="africapolis",
    output_dir=Path("./buildings/"),
    country=["GHA"],  # Just Ghana first
    city=["Accra"],   # Just one city
    confidence_threshold=0.75,
    num_workers=4
)

processor = OpenBuildingsGCSProcessor(config)
result = processor.process()

print(f"Success: {result.success}")
print(f"Succeeded: {result.succeeded}")
print(f"Failed: {result.failed}")
```

## Common Issues

### Issue 1: "AOI file not found: africapolis"
**Cause**: Config validator is trying to check `.exists()` on the string "africapolis"
**Fix**: Make sure aoi_file field type is `Union[Path, str]` and validator checks for "africapolis" string first

### Issue 2: "'str' object has no attribute 'exists'"
**Cause**: Processor's `_validate_custom_inputs()` is calling `.exists()` on string
**Fix**: Add check for africapolis mode before calling `.exists()`

### Issue 3: "No module named 'config_loader'"
**Cause**: Missing import
**Fix**: Methods should import: `from geoworkflow.utils.config_loader import ConfigLoader`

## Key Differences from OSM Implementation

1. **Data Source**: OSM uses PBF files per country, GCS uses S2 cells
2. **Processing Logic**: GCS computes S2 covering for each city's geometry
3. **Output**: Both create separate files per city, but GCS uses S2 cell downloads

## Files to Reference

Look at these working files for guidance:
- `src/geoworkflow/schemas/osm_highways_config.py` - Config pattern
- `src/geoworkflow/processors/extraction/osm_highways.py` - Processor pattern
- `notebooks/agglomerations_highways.ipynb` - Usage example

## Success Criteria

✅ Config accepts `aoi_file="africapolis"` without validation error
✅ Processor validates without checking `.exists()` on "africapolis"
✅ Batch mode loads geometries from AFRICAPOLIS dataset
✅ Each city processed individually with separate output file
✅ Returns `BatchProcessResult` with succeeded/failed lists
✅ Single-file mode still works (backward compatibility)

## Implementation Approach

**RECOMMENDED**: Make changes manually, one at a time, testing after each:
1. Fix config schema → test config creation
2. Fix processor validation → test processor initialization
3. Add batch methods → test with dry-run
4. Update process_data → test full batch run

**AVOID**: Automated scripts that make multiple changes at once - harder to debug when things break.
