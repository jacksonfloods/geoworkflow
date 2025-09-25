from pathlib import Path
from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig

def test_config():
    """Test the Open Buildings configuration."""
    
    # Create dummy AOI file FIRST
    aoi_file = Path("test_aoi.geojson")
    aoi_content = """{
  "type": "FeatureCollection", 
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-0.1, 5.5], [0.1, 5.5], [0.1, 5.7], [-0.1, 5.7], [-0.1, 5.5]
        ]]
      }
    }
  ]
}"""
    
    # Write the file BEFORE creating config
    with open(aoi_file, "w") as f:
        f.write(aoi_content)
    
    try:
        # NOW create the config object (after file exists)
        config = OpenBuildingsExtractionConfig(
            aoi_file=aoi_file,
            output_dir=Path("test_output")
        )
        
        print("✓ Basic configuration created successfully")
        print(f"  - Default confidence threshold: {config.confidence_threshold}")
        print(f"  - Default export format: {config.export_format.value}")  # .value for enum
        print(f"  - Output file path: {config.get_output_file_path()}")
        
        # Test academic guidance
        print("\n" + "="*50)
        print("ACADEMIC SETUP GUIDANCE:")
        print("="*50)
        print(config.get_academic_setup_guidance())
        
        print("\n" + "="*50)
        print("VALIDATION SUCCESS!")
        print("Phase 1.1 Configuration Schema is working correctly!")
        print("="*50)
        
    finally:
        # Clean up test file
        if aoi_file.exists():
            aoi_file.unlink()
            print(f"\n✓ Cleaned up test file: {aoi_file}")

if __name__ == "__main__":
    test_config()