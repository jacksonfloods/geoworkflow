#!/usr/bin/env python3
# File: examples/aoi_processor_example.py
"""
Example usage of the new AOI processor.

This demonstrates how to use the modernized AOI processor with the 
Phase 2.1 enhanced infrastructure.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from geoworkflow.core.logging_setup import setup_logging
from geoworkflow.schemas.config_models import AOIConfig
from geoworkflow.processors.aoi.processor import AOIProcessor


def example_1_southern_africa():
    """Example 1: Create Southern Africa AOI."""
    print("Example 1: Creating Southern Africa AOI")
    print("-" * 50)
    
    # Method 1: Create AOIConfig directly
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["Angola", "Namibia", "Botswana"],
        buffer_km=100,
        output_file=Path("data/aoi/southern_africa_aoi.geojson")
    )
    
    # Create and run processor
    processor = AOIProcessor(config)
    result = processor.process()
    
    print(f"✅ Success: {result.success}")
    print(f"📁 Output: {config.output_file}")
    print(f"⏱️  Time: {result.elapsed_time:.2f}s")
    print(f"📊 Features: {result.metadata.get('feature_count', 'N/A')}")


def example_2_from_yaml_config():
    """Example 2: Create AOI from YAML configuration file."""
    print("\nExample 2: Creating AOI from YAML configuration")
    print("-" * 50)
    
    # First, create a sample YAML config file
    yaml_config = """
input_file: "data/00_source/archives/africa_boundaries/africa_boundaries.geojson"
country_name_column: "NAME_0"
countries: ["Kenya", "Tanzania", "Uganda"]
buffer_km: 75
output_file: "data/aoi/east_africa_aoi.geojson"
"""
    
    config_file = Path("config/examples/east_africa_aoi.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        f.write(yaml_config)
    
    print(f"📝 Created config file: {config_file}")
    
    # Method 2: Load from YAML file
    config = AOIConfig.from_file(config_file)
    processor = AOIProcessor(config)
    result = processor.process()
    
    print(f"✅ Success: {result.success}")
    print(f"📁 Output: {config.output_file}")


def example_3_continental_aoi():
    """Example 3: Create continental AOI with dissolved boundaries."""
    print("\nExample 3: Creating continental AOI (all countries, dissolved)")
    print("-" * 50)
    
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        use_all_countries=True,
        dissolve_boundaries=True,
        buffer_km=50,
        output_file=Path("data/aoi/africa_continental_aoi.geojson")
    )
    
    processor = AOIProcessor(config)
    result = processor.process()
    
    # Get additional information
    bounds = processor.get_aoi_bounds()
    area_km2 = processor.get_aoi_area_km2()
    
    print(f"✅ Success: {result.success}")
    print(f"🌍 Bounds: {bounds}")
    print(f"📏 Area: {area_km2:,.0f} km²" if area_km2 else "Area: N/A")


def example_4_list_and_explore():
    """Example 4: Explore available countries."""
    print("\nExample 4: Exploring available countries")
    print("-" * 50)
    
    # Create minimal config for exploration
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["dummy"],  # Will be ignored for listing
        output_file=Path("data/aoi/dummy.geojson")  # Will be ignored for listing
    )
    
    processor = AOIProcessor(config)
    
    # List all available countries
    all_countries = processor.list_available_countries()
    print(f"📊 Total countries available: {len(all_countries)}")
    print(f"🔤 First 10 countries: {all_countries[:10]}")
    
    # List countries starting with specific letters
    for letter in ['S', 'K', 'G']:
        countries = processor.list_available_countries(prefix=letter)
        print(f"🔍 Countries starting with '{letter}': {countries}")


def example_5_error_handling():
    """Example 5: Demonstrate error handling."""
    print("\nExample 5: Error handling demonstration")
    print("-" * 50)
    
    # Example of handling validation errors
    try:
        config = AOIConfig(
            input_file=Path("nonexistent_file.geojson"),
            country_name_column="NAME_0",
            countries=["Angola"],
            output_file=Path("data/aoi/test.geojson")
        )
        processor = AOIProcessor(config)
        result = processor.process()
    except Exception as e:
        print(f"❌ Expected error caught: {type(e).__name__}")
        print(f"   Message: {str(e)}")
    
    # Example of handling invalid country names
    try:
        config = AOIConfig(
            input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
            country_name_column="NAME_0",
            countries=["NonexistentCountry"],
            output_file=Path("data/aoi/test.geojson")
        )
        processor = AOIProcessor(config)
        result = processor.process()
    except Exception as e:
        print(f"❌ Expected error caught: {type(e).__name__}")
        print(f"   Message: {str(e)}")


def example_6_enhanced_features():
    """Example 6: Demonstrate enhanced features from Phase 2.1."""
    print("\nExample 6: Enhanced features demonstration")
    print("-" * 50)
    
    config = AOIConfig(
        input_file=Path("data/00_source/archives/africa_boundaries/africa_boundaries.geojson"),
        country_name_column="NAME_0",
        countries=["South Africa", "Zimbabwe"],
        buffer_km=25,
        output_file=Path("data/aoi/enhanced_features_demo.geojson")
    )
    
    processor = AOIProcessor(config)
    result = processor.process()
    
    print(f"✅ Processing result: {result.success}")
    print(f"⏱️  Elapsed time: {result.elapsed_time:.2f}s")
    
    # Enhanced result information
    if hasattr(result, 'validation_results') and result.validation_results:
        print("\n🔍 Validation Results:")
        for component, validation in result.validation_results.items():
            print(f"   {component}: {'✅ Valid' if validation.get('valid', False) else '❌ Invalid'}")
    
    if hasattr(result, 'setup_info') and result.setup_info:
        print("\n⚙️  Setup Information:")
        for key, value in result.setup_info.items():
            print(f"   {key}: {value}")
    
    if hasattr(result, 'metrics') and result.metrics:
        print("\n📊 Processing Metrics:")
        metrics_dict = result.metrics.to_dict() if hasattr(result.metrics, 'to_dict') else result.metrics
        for key, value in metrics_dict.items():
            if key != 'custom_metrics':  # Skip nested dict for cleaner output
                print(f"   {key}: {value}")


def main():
    """Run all AOI processor examples."""
    print("🚀 AOI PROCESSOR USAGE EXAMPLES")
    print("=" * 60)
    print("Demonstrating the enhanced AOI processor with Phase 2.1 infrastructure")
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Ensure output directories exist
    Path("data/aoi").mkdir(parents=True, exist_ok=True)
    Path("config/examples").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    examples = [
        example_1_southern_africa,
        example_2_from_yaml_config,
        example_3_continental_aoi,
        example_4_list_and_explore,
        example_5_error_handling,
        example_6_enhanced_features,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nGenerated files:")
    aoi_files = list(Path("data/aoi").glob("*.geojson"))
    for aoi_file in aoi_files:
        file_size = aoi_file.stat().st_size
        print(f"   📁 {aoi_file.name} ({file_size:,} bytes)")


if __name__ == "__main__":
    main()