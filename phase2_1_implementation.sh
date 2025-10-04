#!/bin/bash
# Phase 2.1 Implementation: Configuration Schema for GCS-Based Open Buildings Extraction
# This script creates the configuration model for the GCS processor

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Phase 2.1: Configuration Schema Implementation ===${NC}"

# Check if config_models.py exists
CONFIG_FILE="src/geoworkflow/schemas/config_models.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Error: $CONFIG_FILE not found!${NC}"
    echo "  Make sure you're running this from the project root directory."
    exit 1
fi

echo -e "${BLUE}Found existing config file: $CONFIG_FILE${NC}"

# Create backup
BACKUP_FILE="${CONFIG_FILE}.backup_phase2_1_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo -e "${GREEN}✓ Created backup: $BACKUP_FILE${NC}"

# Create the new configuration class as a separate file first
# This allows review before integration
TEMP_CONFIG="src/geoworkflow/schemas/open_buildings_gcs_config.py"

echo -e "\n${YELLOW}Creating OpenBuildingsGCSConfig class...${NC}"

cat > "$TEMP_CONFIG" << 'EOF'
"""
Configuration model for GCS-based Open Buildings extraction.

This configuration supports direct Google Cloud Storage access for
building footprint extraction from the Open Buildings v3 dataset.
"""
from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class OpenBuildingsGCSConfig(BaseModel):
    """
    Configuration for GCS-based Open Buildings extraction.
    
    This is the primary extraction method for Open Buildings data,
    significantly faster than the Earth Engine approach.
    
    Example:
        >>> config = OpenBuildingsGCSConfig(
        ...     aoi_file=Path("study_area.geojson"),
        ...     output_dir=Path("./buildings/"),
        ...     confidence_threshold=0.75,
        ...     num_workers=4
        ... )
    """
    
    # ==================== Required Inputs ====================
    aoi_file: Path = Field(
        ...,
        description="Area of Interest boundary file (GeoJSON, Shapefile, etc.)"
    )
    
    output_dir: Path = Field(
        ...,
        description="Output directory for extracted buildings"
    )
    
    # ==================== Data Source Settings ====================
    data_type: Literal["polygons", "points"] = Field(
        default="polygons",
        description="Type of building data to extract (polygons include geometry)"
    )
    
    s2_level: int = Field(
        default=6,
        ge=4,
        le=8,
        description="S2 cell level - must match GCS bucket structure (default: 6)"
    )
    
    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/polygons_s2_level_6_gzip_no_header",
        description="GCS path to Open Buildings data"
    )
    
    # ==================== Filtering Options ====================
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum building confidence score (0.5-1.0)"
    )
    
    min_area_m2: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum building area in square meters"
    )
    
    max_area_m2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum building area in square meters (None for no limit)"
    )
    
    # ==================== Export Settings ====================
    export_format: Literal["geojson", "shapefile", "csv", "geoparquet"] = Field(
        default="geojson",
        description="Output file format"
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores in output"
    )
    
    include_area: bool = Field(
        default=True,
        description="Include area calculations in output"
    )
    
    include_plus_codes: bool = Field(
        default=True,
        description="Include Plus Codes in output"
    )
    
    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )
    
    # ==================== Performance Settings ====================
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for downloading S2 cells"
    )
    
    chunk_size: int = Field(
        default=2_000_000,
        ge=100_000,
        description="Number of records to process per chunk"
    )
    
    # ==================== Authentication (optional for public data) ====================
    service_account_key: Optional[Path] = Field(
        default=None,
        description="Path to GCS service account key (not required for Open Buildings)"
    )
    
    use_anonymous_access: bool = Field(
        default=True,
        description="Use anonymous access for public data (recommended for Open Buildings)"
    )
    
    # ==================== Validators ====================
    @field_validator('aoi_file')
    @classmethod
    def validate_aoi_exists(cls, v: Path) -> Path:
        """Validate that AOI file exists."""
        if not v.exists():
            raise ValueError(f"AOI file not found: {v}")
        return v
    
    @field_validator('service_account_key')
    @classmethod
    def validate_service_account(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate service account key if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"Service account key file not found: {v}")
        return v
    
    @field_validator('max_area_m2')
    @classmethod
    def validate_area_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that max_area > min_area if both are set."""
        if v is not None:
            min_area = info.data.get('min_area_m2', 0.0)
            if v <= min_area:
                raise ValueError(
                    f"max_area_m2 ({v}) must be greater than min_area_m2 ({min_area})"
                )
        return v
    
    @field_validator('gcs_bucket_path')
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Validate GCS bucket path format."""
        if not v.startswith('gs://'):
            raise ValueError(f"GCS path must start with 'gs://': {v}")
        return v
    
    # ==================== Helper Methods ====================
    def get_output_file_path(self) -> Path:
        """
        Generate output file path based on format.
        
        Returns:
            Path object for the output file
        """
        extensions = {
            "geojson": ".geojson",
            "shapefile": ".shp",
            "csv": ".csv",
            "geoparquet": ".parquet"
        }
        extension = extensions.get(self.export_format, ".geojson")
        return self.output_dir / f"open_buildings{extension}"
    
    def get_gcs_file_pattern(self) -> str:
        """
        Get the GCS file pattern for S2 cells.
        
        Returns:
            String pattern for GCS files
        """
        return f"{self.gcs_bucket_path}/*.csv.gz"
    
    def estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB based on settings.
        
        Returns:
            Estimated memory usage in megabytes
        """
        # Rough estimate: chunk_size * 200 bytes per record
        memory_per_worker = (self.chunk_size * 200) / (1024 * 1024)
        total_memory = memory_per_worker * self.num_workers
        return total_memory
    
    def summary(self) -> dict:
        """
        Get a summary of key configuration settings.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "aoi_file": str(self.aoi_file),
            "output_dir": str(self.output_dir),
            "output_format": self.export_format,
            "confidence_threshold": self.confidence_threshold,
            "area_filter": f"{self.min_area_m2}-{self.max_area_m2 or 'unlimited'} m²",
            "parallel_workers": self.num_workers,
            "s2_level": self.s2_level,
            "data_source": self.gcs_bucket_path,
            "estimated_memory_mb": round(self.estimate_memory_usage(), 2)
        }
    
    class Config:
        """Pydantic model configuration."""
        # Allow Path objects
        arbitrary_types_allowed = True
        # Enable validation on assignment
        validate_assignment = True
        # JSON schema extras
        json_schema_extra = {
            "example": {
                "aoi_file": "study_area.geojson",
                "output_dir": "./buildings/",
                "confidence_threshold": 0.75,
                "min_area_m2": 10.0,
                "export_format": "geojson",
                "num_workers": 4
            }
        }


# Alternative configuration for points (lighter-weight)
class OpenBuildingsGCSPointsConfig(OpenBuildingsGCSConfig):
    """
    Specialized configuration for extracting building points only.
    
    This is faster and requires less storage than polygon extraction.
    """
    
    data_type: Literal["points"] = Field(
        default="points",
        description="Extract building centroids only"
    )
    
    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/points_s2_level_6_gzip",
        description="GCS path to Open Buildings points data"
    )
    
    # Points don't have geometry column, so this is always CSV
    export_format: Literal["csv", "geojson"] = Field(
        default="csv",
        description="Output format (CSV for tabular, GeoJSON to create points)"
    )
EOF

echo -e "${GREEN}✓ Created temporary config file: $TEMP_CONFIG${NC}"

# Create integration instructions
INSTRUCTIONS_FILE="PHASE_2_1_INTEGRATION_INSTRUCTIONS.md"

cat > "$INSTRUCTIONS_FILE" << 'EOF'
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
EOF

echo -e "${GREEN}✓ Created integration instructions: $INSTRUCTIONS_FILE${NC}"

# Create a simple test script
TEST_SCRIPT="test_phase2_1_config.py"

cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Test script for OpenBuildingsGCSConfig.

This script validates the configuration class works correctly.
"""
from pathlib import Path
import sys

try:
    # Try to import from the new location
    sys.path.insert(0, 'src')
    from geoworkflow.schemas.open_buildings_gcs_config import (
        OpenBuildingsGCSConfig,
        OpenBuildingsGCSPointsConfig
    )
    print("✓ Successfully imported OpenBuildingsGCSConfig")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_basic_config():
    """Test basic configuration creation."""
    print("\n=== Testing Basic Configuration ===")
    
    # Create a dummy AOI file for testing
    test_aoi = Path("test_aoi.geojson")
    if not test_aoi.exists():
        print(f"⚠ Warning: {test_aoi} doesn't exist, creating placeholder...")
        test_aoi.write_text('{"type":"FeatureCollection","features":[]}')
    
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            confidence_threshold=0.75,
            num_workers=2
        )
        print("✓ Basic configuration created successfully")
        print(f"  - AOI file: {config.aoi_file}")
        print(f"  - Output dir: {config.output_dir}")
        print(f"  - Confidence: {config.confidence_threshold}")
        print(f"  - Workers: {config.num_workers}")
        return True
    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        return False

def test_validation():
    """Test configuration validation."""
    print("\n=== Testing Validation ===")
    
    # Test invalid confidence threshold
    test_aoi = Path("test_aoi.geojson")
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            confidence_threshold=0.3  # Too low!
        )
        print("✗ Validation failed - accepted invalid confidence")
        return False
    except ValueError:
        print("✓ Correctly rejected invalid confidence threshold")
    
    # Test invalid area range
    try:
        config = OpenBuildingsGCSConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/"),
            min_area_m2=100.0,
            max_area_m2=50.0  # Max < Min!
        )
        print("✗ Validation failed - accepted invalid area range")
        return False
    except ValueError:
        print("✓ Correctly rejected invalid area range")
    
    return True

def test_helper_methods():
    """Test configuration helper methods."""
    print("\n=== Testing Helper Methods ===")
    
    test_aoi = Path("test_aoi.geojson")
    config = OpenBuildingsGCSConfig(
        aoi_file=test_aoi,
        output_dir=Path("./test_output/"),
        export_format="geojson"
    )
    
    # Test output path generation
    output_path = config.get_output_file_path()
    print(f"✓ Output path: {output_path}")
    
    # Test summary
    summary = config.summary()
    print("✓ Configuration summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    # Test memory estimation
    memory = config.estimate_memory_usage()
    print(f"✓ Estimated memory usage: {memory:.2f} MB")
    
    return True

def test_points_config():
    """Test points-specific configuration."""
    print("\n=== Testing Points Configuration ===")
    
    test_aoi = Path("test_aoi.geojson")
    try:
        config = OpenBuildingsGCSPointsConfig(
            aoi_file=test_aoi,
            output_dir=Path("./test_output/")
        )
        print("✓ Points configuration created successfully")
        print(f"  - Data type: {config.data_type}")
        print(f"  - GCS path: {config.gcs_bucket_path}")
        print(f"  - Export format: {config.export_format}")
        return True
    except Exception as e:
        print(f"✗ Failed to create points config: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OpenBuildingsGCSConfig Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_config,
        test_validation,
        test_helper_methods,
        test_points_config,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    # Cleanup test file
    test_aoi = Path("test_aoi.geojson")
    if test_aoi.exists():
        test_aoi.unlink()
        print("✓ Cleaned up test files")
    
    sys.exit(0 if all(results) else 1)
EOF

chmod +x "$TEST_SCRIPT"
echo -e "${GREEN}✓ Created test script: $TEST_SCRIPT${NC}"

# Summary
echo -e "\n${GREEN}=== Phase 2.1 Implementation Complete ===${NC}"
echo -e "\n${BLUE}Created files:${NC}"
echo "  ✓ src/geoworkflow/schemas/open_buildings_gcs_config.py"
echo "  ✓ $INSTRUCTIONS_FILE"
echo "  ✓ $TEST_SCRIPT"
echo "  ✓ Backup: $BACKUP_FILE"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Review the configuration class in:"
echo "     src/geoworkflow/schemas/open_buildings_gcs_config.py"
echo ""
echo "  2. Test the configuration:"
echo "     python $TEST_SCRIPT"
echo ""
echo "  3. Review integration options in:"
echo "     $INSTRUCTIONS_FILE"
echo ""
echo "  4. Update schemas/__init__.py to export the new config"
echo ""
echo "  5. Proceed to Phase 2.2: Main GCS Processor implementation"

echo -e "\n${BLUE}Quick test command:${NC}"
echo "  python $TEST_SCRIPT"
