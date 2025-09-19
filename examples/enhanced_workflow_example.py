# File: examples/enhanced_workflow_example.py
"""
Complete example showing how to use the enhanced Phase 2.1 components.

This example demonstrates:
1. Enhanced BaseProcessor with template method pattern
2. Resource management and cleanup
3. Progress tracking and monitoring
4. Pipeline validation and data flow checking
5. Comprehensive error handling and logging
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import time
import json

# Enhanced geoworkflow imports
from geoworkflow.core.enhanced_base import TemplateMethodProcessor, EnhancedProcessingResult, GeospatialProcessorMixin
from geoworkflow.core.pipeline_enhancements import PipelineEnhancementMixin
from geoworkflow.core.pipeline import ProcessingPipeline
from geoworkflow.schemas.config_models import WorkflowConfig, ExtractionConfig
from geoworkflow.utils.progress_utils import track_progress, PerformanceMonitor
from geoworkflow.utils.resource_utils import temp_directory, ensure_directory
from geoworkflow.core.logging_setup import setup_logging


class EnhancedExtractionProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Example of an enhanced processor using the template method pattern.
    
    This processor demonstrates all the enhanced features:
    - Template method workflow (validate → setup → process → cleanup)
    - Resource management with automatic cleanup
    - Progress tracking with rich output
    - Comprehensive result tracking
    - Geospatial processing utilities
    """
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """Custom validation for extraction processor."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Add geospatial validation
        geo_validation = self.validate_geospatial_inputs()
        validation_result.update(geo_validation)
        
        # Check for source data
        if "source_dir" in self.config:
            source_dir = Path(self.config["source_dir"])
            if not source_dir.exists():
                validation_result["errors"].append(f"Source directory does not exist: {source_dir}")
                validation_result["valid"] = False
            else:
                # Count archive files
                archive_files = list(source_dir.glob("*.zip"))
                if not archive_files:
                    validation_result["warnings"].append("No ZIP files found in source directory")
                else:
                    validation_result["archive_count"] = len(archive_files)
        
        # Check AOI file
        if "aoi_file" in self.config:
            aoi_file = Path(self.config["aoi_file"])
            if not aoi_file.exists():
                validation_result["errors"].append(f"AOI file does not exist: {aoi_file}")
                validation_result["valid"] = False
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths."""
        return ["source_dir", "aoi_file", "output_dir"]
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        if "source_dir" in self.config:
            source_dir = Path(self.config["source_dir"])
            if source_dir.exists():
                archive_files = list(source_dir.glob("*.zip"))
                return len(archive_files)
        return 0
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup extraction-specific resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Create output directory
        if "output_dir" in self.config:
            output_dir = ensure_directory(self.config["output_dir"])
            setup_info["output_dir_created"] = str(output_dir)
        
        # Setup extraction workspace
        workspace = self.get_temp_directory()
        setup_info["workspace"] = str(workspace)
        
        return setup_info
    
    def process_data(self) -> EnhancedProcessingResult:
        """Main extraction processing logic."""
        result = EnhancedProcessingResult(success=True)
        
        try:
            # Get list of files to process
            source_dir = Path(self.config["source_dir"])
            archive_files = list(source_dir.glob("*.zip"))
            
            self.log_processing_step(f"Processing {len(archive_files)} archive files")
            
            # Process each archive with progress tracking
            for archive_file in track_progress(
                archive_files, 
                description="Extracting archives", 
                quiet=False
            ):
                self.log_processing_step(f"Processing {archive_file.name}")
                
                # Simulate extraction work
                with temp_directory(suffix="_extract") as extract_dir:
                    # Here would be actual extraction logic
                    time.sleep(0.1)  # Simulate work
                    
                    # Update metrics
                    self.metrics.files_processed += 1
                    self.add_metric("last_processed_file", str(archive_file))
                
                # Update our progress tracker
                self.update_progress(1, f"Processed {archive_file.name}")
            
            result.processed_count = len(archive_files)
            result.message = f"Successfully extracted {len(archive_files)} archives"
            
            # Add output paths
            output_dir = Path(self.config["output_dir"])
            result.add_output_path(output_dir)
            
        except Exception as e:
            result.success = False
            result.message = f"Extraction failed: {str(e)}"
            self.logger.error(f"Extraction processing failed: {e}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup extraction-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Any extraction-specific cleanup would go here
        cleanup_info["extraction_cleanup"] = "completed"
        
        return cleanup_info


class EnhancedProcessingPipeline(ProcessingPipeline, PipelineEnhancementMixin):
    """Enhanced pipeline with validation and tracking capabilities."""
    pass


def demonstrate_enhanced_processor():
    """Demonstrate the enhanced processor capabilities."""
    print("\n" + "="*60)
    print("ENHANCED PROCESSOR DEMONSTRATION")
    print("="*60)
    
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger("enhanced_example")
    
    # Create configuration
    config = {
        "source_dir": "data/00_source/archives",
        "aoi_file": "data/aoi/southern_africa_aoi.geojson", 
        "output_dir": "data/01_extracted"
    }
    
    # Create and run enhanced processor
    processor = EnhancedExtractionProcessor(config)
    
    logger.info("Running enhanced extraction processor...")
    result = processor.process()
    
    # Display results
    print(f"\nProcessor completed:")
    print(f"  Success: {result.success}")
    print(f"  Files processed: {result.processed_count}")
    print(f"  Elapsed time: {result.elapsed_time:.2f}s")
    
    if result.metrics:
        print(f"  Metrics: {json.dumps(result.metrics.to_dict(), indent=2, default=str)}")
    
    print(f"  Validation results: {json.dumps(result.validation_results, indent=2)}")
    print(f"  Setup info: {json.dumps(result.setup_info, indent=2)}")
    print(f"  Cleanup info: {json.dumps(result.cleanup_info, indent=2)}")


def demonstrate_enhanced_pipeline():
    """Demonstrate the enhanced pipeline capabilities."""
    print("\n" + "="*60)
    print("ENHANCED PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Create a sample workflow configuration
    workflow_config_dict = {
        "name": "Enhanced Southern Africa Workflow",
        "description": "Demonstrating enhanced pipeline capabilities",
        "aoi": {
            "input_file": "data/00_source/boundaries/africa_boundaries.geojson",
            "country_name_column": "NAME_0",
            "countries": ["Angola", "Namibia", "Botswana"],
            "buffer_km": 100,
            "output_file": "data/aoi/southern_africa_aoi.geojson"
        },
        "data_sources": {
            "copernicus": {"type": "raster", "priority": 1},
            "odiac": {"type": "raster", "priority": 2}
        },
        "stages": ["extract", "clip", "align"],
        "create_visualizations": True,
        "stop_on_error": True
    }
    
    # Create enhanced pipeline
    try:
        workflow_config = WorkflowConfig.from_dict(workflow_config_dict)
        pipeline = EnhancedProcessingPipeline(workflow_config)
        
        print("Running comprehensive pipeline validation...")
        
        # Run comprehensive validation
        validation_results = pipeline.run_comprehensive_validation()
        
        print(f"Validation results:")
        print(f"  Overall valid: {validation_results['overall_valid']}")
        print(f"  Errors: {len(validation_results['errors'])}")
        print(f"  Warnings: {len(validation_results['warnings'])}")
        
        if validation_results['errors']:
            print(f"  Error details: {validation_results['errors']}")
        
        if validation_results['warnings']:
            print(f"  Warning details: {validation_results['warnings']}")
        
        # Show stage status
        print("\nChecking stage status...")
        status = pipeline.get_stage_status()
        print(f"Overall status: {status['overall_status']}")
        print(f"Completion: {status['completion_percentage']:.1f}%")
        
        for stage, stage_info in status['stages'].items():
            print(f"  {stage}: {stage_info['status']} "
                  f"(completion: {stage_info['estimated_completion']:.1f}%)")
        
        # Demonstrate dependency validation
        print("\nValidating pipeline dependencies...")
        dep_validation = pipeline.validate_pipeline_dependencies()
        print(f"Dependencies valid: {dep_validation['valid']}")
        
        for stage, deps in dep_validation['stage_dependencies'].items():
            print(f"  {stage} depends on: {deps if deps else 'none'}")
        
        # Demonstrate data flow validation
        print("\nValidating data flow...")
        flow_validation = pipeline.validate_data_flow()
        print(f"Data flow valid: {flow_validation['valid']}")
        
        if flow_validation['errors']:
            print(f"  Flow errors: {flow_validation['errors']}")
        
    except Exception as e:
        print(f"Pipeline demonstration failed: {e}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Simulate various processing steps
    steps = [
        ("Loading configuration", 0.1),
        ("Validating inputs", 0.2),
        ("Setting up workspace", 0.1),
        ("Processing data", 0.5),
        ("Creating outputs", 0.3),
        ("Cleaning up", 0.1)
    ]
    
    for step_name, duration in steps:
        print(f"Executing: {step_name}")
        time.sleep(duration)  # Simulate work
        monitor.checkpoint(step_name)
    
    # Get performance summary
    summary = monitor.get_summary()
    
    print(f"\nPerformance Summary:")
    print(f"  Total time: {summary['total_time']:.2f}s")
    print(f"  Checkpoints: {summary['checkpoint_count']}")
    
    if 'peak_memory_mb' in summary:
        print(f"  Peak memory: {summary['peak_memory_mb']:.1f} MB")
    
    print(f"\nCheckpoint details:")
    for checkpoint in summary['checkpoints']:
        print(f"  {checkpoint['name']}: {checkpoint['elapsed']:.2f}s")


def main():
    """Run all demonstrations."""
    print("GEOWORKFLOW PHASE 2.1 ENHANCED COMPONENTS DEMONSTRATION")
    print("========================================================")
    
    try:
        # Demonstrate enhanced processor
        demonstrate_enhanced_processor()
        
        # Demonstrate enhanced pipeline
        demonstrate_enhanced_pipeline()
        
        # Demonstrate performance monitoring
        demonstrate_performance_monitoring()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey enhancements demonstrated:")
        print("✅ Template method pattern for structured processing")
        print("✅ Automatic resource management and cleanup")
        print("✅ Rich progress tracking and monitoring")
        print("✅ Comprehensive validation and error handling")
        print("✅ Pipeline dependency and data flow validation")
        print("✅ Performance monitoring and metrics collection")
        print("✅ Enhanced result objects with detailed information")
        print("✅ Geospatial processing mixins for reusable functionality")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# Additional utility functions for integration

def create_enhanced_processor_from_config(processor_type: str, config_path: str):
    """
    Factory function to create enhanced processors from configuration files.
    
    Args:
        processor_type: Type of processor ('extraction', 'clipping', etc.)
        config_path: Path to configuration file
        
    Returns:
        Configured enhanced processor instance
    """
    # This would be expanded to support different processor types
    processor_map = {
        'extraction': EnhancedExtractionProcessor,
        # Add other enhanced processors as they're implemented
    }
    
    if processor_type not in processor_map:
        raise ValueError(f"Unknown processor type: {processor_type}")
    
    # Load configuration
    config_dict = {}
    config_path = Path(config_path)
    
    if config_path.suffix.lower() == '.json':
        import json
        with open(config_path) as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    
    # Create processor
    processor_class = processor_map[processor_type]
    return processor_class(config_dict)


def run_enhanced_workflow_from_config(workflow_config_path: str):
    """
    Run a complete enhanced workflow from a configuration file.
    
    Args:
        workflow_config_path: Path to workflow configuration file
        
    Returns:
        Enhanced pipeline results
    """
    # Load workflow configuration
    workflow_config = WorkflowConfig.from_file(workflow_config_path)
    
    # Create enhanced pipeline
    pipeline = EnhancedProcessingPipeline(workflow_config)
    
    # Run with enhanced tracking
    return pipeline.run_with_enhanced_tracking()


# Example configuration templates for testing

EXAMPLE_EXTRACTION_CONFIG = {
    "source_dir": "data/00_source/archives",
    "aoi_file": "data/aoi/southern_africa_aoi.geojson",
    "output_dir": "data/01_extracted",
    "raster_pattern": "*.tif",
    "skip_existing": True,
    "create_visualizations": True
}

EXAMPLE_WORKFLOW_CONFIG = {
    "name": "Enhanced Test Workflow",
    "description": "Testing enhanced pipeline capabilities",
    "aoi": {
        "input_file": "data/00_source/boundaries/africa_boundaries.geojson",
        "country_name_column": "NAME_0",
        "countries": ["Angola", "Namibia", "Botswana"],
        "buffer_km": 100,
        "output_file": "data/aoi/southern_africa_aoi.geojson"
    },
    "data_sources": {
        "copernicus": {"type": "raster", "priority": 1},
        "odiac": {"type": "raster", "priority": 2}
    },
    "stages": ["extract", "clip", "align"],
    "create_visualizations": True,
    "stop_on_error": True,
    "log_level": "INFO"
}