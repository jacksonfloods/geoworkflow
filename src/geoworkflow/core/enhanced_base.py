# File: src/geoworkflow/core/enhanced_base.py
"""
Enhanced base classes for the geoworkflow package.

This module provides enhanced base classes with template method pattern,
resource management, and comprehensive progress tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Iterator
from pathlib import Path
import logging
from datetime import datetime
import time

from geoworkflow.core.base import BaseProcessor, ProcessingResult
from geoworkflow.core.exceptions import ProcessingError, ConfigurationError
from geoworkflow.utils.resource_utils import ResourceManager, ProcessingMetrics
from geoworkflow.utils.progress_utils import ProgressTracker


class EnhancedProcessingResult(ProcessingResult):
    """Enhanced result object with detailed metrics and tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics: Optional[ProcessingMetrics] = None
        self.resource_usage: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}
        self.setup_info: Dict[str, Any] = {}
        self.cleanup_info: Dict[str, Any] = {}
    
    def add_validation_result(self, component: str, result: Dict[str, Any]):
        """Add validation results for a component."""
        self.validation_results[component] = result
    
    def add_setup_info(self, component: str, info: Dict[str, Any]):
        """Add setup information for a component."""
        self.setup_info[component] = info
    
    def add_cleanup_info(self, component: str, info: Dict[str, Any]):
        """Add cleanup information for a component."""
        self.cleanup_info[component] = info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert enhanced result to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'resource_usage': self.resource_usage,
            'validation_results': self.validation_results,
            'setup_info': self.setup_info,
            'cleanup_info': self.cleanup_info
        })
        return base_dict


class TemplateMethodProcessor(BaseProcessor):
    """
    Enhanced BaseProcessor implementing the Template Method pattern.
    
    This class provides a structured workflow:
    1. validate_inputs() - Validate all inputs and configuration
    2. setup_processing() - Set up resources and prepare for processing  
    3. process_data() - Main processing logic (abstract)
    4. cleanup_resources() - Clean up resources and temporary files
    
    Subclasses only need to implement process_data() and can optionally
    override the other methods for custom behavior.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
        # Enhanced components
        self.resource_manager = ResourceManager(
            prefix=f"{self.__class__.__name__.lower()}_"
        )
        self.metrics = ProcessingMetrics()
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # State tracking
        self._setup_completed = False
        self._processing_started = False
        self._cleanup_completed = False

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for template method processor.
        
        This is the abstract method required by BaseProcessor.
        The actual validation is done in validate_inputs().
        """
        # Basic validation - just ensure config is a dict
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        return config
    
    def process(self) -> EnhancedProcessingResult:
        """
        Execute the complete processing workflow using template method pattern.
        
        Returns:
            EnhancedProcessingResult with comprehensive information
        """
        result = EnhancedProcessingResult(success=False)
        result.metrics = self.metrics
        try:
            # Start timing
            self.metrics.start()
            self.logger.info(f"Starting {self.__class__.__name__} processing")
            
            # Step 1: Validate inputs
            self.logger.debug("Step 1: Validating inputs")
            validation_result = self.validate_inputs()
            result.add_validation_result("inputs", validation_result)
            
            if not validation_result.get("valid", False):
                raise ConfigurationError(
                    f"Input validation failed: {validation_result.get('errors', [])}"
                )
            
            # Step 2: Setup processing
            self.logger.debug("Step 2: Setting up processing")
            setup_result = self.setup_processing()
            result.add_setup_info("processing", setup_result)
            self._setup_completed = True
            
            # Step 3: Main processing
            self.logger.debug("Step 3: Executing main processing")
            self._processing_started = True
            processing_result = self.process_data()
            
            # Merge processing result into our enhanced result
            if isinstance(processing_result, ProcessingResult):
                result.processed_count = processing_result.processed_count
                result.failed_count = processing_result.failed_count
                result.skipped_count = processing_result.skipped_count
                result.failed_files = processing_result.failed_files
                result.output_paths = processing_result.output_paths
                result.metadata = processing_result.metadata
                result.message = processing_result.message
            
            # Step 4: Cleanup
            self.logger.debug("Step 4: Cleaning up resources")
            cleanup_result = self.cleanup_resources()
            result.add_cleanup_info("resources", cleanup_result)
            self._cleanup_completed = True
            
            # Mark as successful
            result.success = True
            self.logger.info(f"Successfully completed {self.__class__.__name__} processing")
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            result.success = False
            result.message = str(e)
            
            # Attempt cleanup even on failure
            if self._setup_completed and not self._cleanup_completed:
                try:
                    cleanup_result = self.cleanup_resources()
                    result.add_cleanup_info("resources", cleanup_result)
                except Exception as cleanup_error:
                    self.logger.error(f"Cleanup also failed: {cleanup_error}")
        
        finally:
            # Finish timing
            self.metrics.finish()
            result.elapsed_time = self.metrics.elapsed_time
            
            # Final resource manager cleanup
            self.resource_manager.cleanup()
        
        return result
    
    def validate_inputs(self) -> Dict[str, Any]:
        """
        Validate inputs and configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        try:
            # Basic config validation (already done in __init__)
            validation_result["info"]["config_valid"] = True
            
            # Validate paths exist if specified
            path_keys = self._get_path_config_keys()
            for key in path_keys:
                if key in self.config and self.config[key] is not None:  # <- Add None check here
                    try:
                        path = Path(self.config[key])
                        if not path.exists():
                            validation_result["errors"].append(f"Path does not exist: {key} = {path}")
                            validation_result["valid"] = False
                    except (TypeError, ValueError) as e:
                        validation_result["errors"].append(f"Validation failed: {str(e)}")
                        validation_result["valid"] = False
            
            # Custom validation from subclass
            custom_validation = self._validate_custom_inputs()
            validation_result["info"]["custom_validation"] = custom_validation
            
            if not custom_validation.get("valid", True):
                validation_result["errors"].extend(custom_validation.get("errors", []))
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    def setup_processing(self) -> Dict[str, Any]:
        """
        Set up resources and prepare for processing.
        
        Returns:
            Dictionary with setup information
        """
        setup_info = {
            "temp_directories_created": [],
            "progress_tracker_initialized": False,
            "custom_setup_completed": False
        }
        
        try:
            # Create temporary directories as needed
            if self._needs_temp_directory():
                temp_dir = self.resource_manager.create_temp_directory()
                setup_info["temp_directories_created"].append(str(temp_dir))
                self.logger.debug(f"Created temporary directory: {temp_dir}")
            
            # Set up progress tracking
            total_items = self._estimate_total_items()
            if total_items > 0:
                self.progress_tracker = ProgressTracker(
                    total=total_items,
                    description=f"{self.__class__.__name__} Processing"
                )
                setup_info["progress_tracker_initialized"] = True
                setup_info["estimated_items"] = total_items
            
            # Custom setup from subclass
            custom_setup = self._setup_custom_processing()
            setup_info["custom_setup"] = custom_setup
            setup_info["custom_setup_completed"] = True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            setup_info["error"] = str(e)
            raise ProcessingError(f"Setup failed: {str(e)}")
        
        return setup_info
    
    @abstractmethod
    def process_data(self) -> ProcessingResult:
        """
        Execute the main processing logic.
        
        This method must be implemented by subclasses to perform the
        actual data processing work.
        
        Returns:
            ProcessingResult with processing outcomes
        """
        pass
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """
        Clean up resources and temporary files.
        
        Returns:
            Dictionary with cleanup information
        """
        cleanup_info = {
            "temp_files_cleaned": 0,
            "temp_directories_cleaned": 0,
            "custom_cleanup_completed": False
        }
        
        try:
            # Custom cleanup from subclass
            custom_cleanup = self._cleanup_custom_processing()
            cleanup_info["custom_cleanup"] = custom_cleanup
            cleanup_info["custom_cleanup_completed"] = True
            
            # Close progress tracker
            if self.progress_tracker:
                self.progress_tracker.close()
                cleanup_info["progress_tracker_closed"] = True
            
            # Resource manager will handle temp file cleanup automatically
            
        except Exception as e:
            self.logger.warning(f"Cleanup encountered errors: {str(e)}")
            cleanup_info["error"] = str(e)
        
        return cleanup_info
    
    # Protected methods for subclasses to override
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Custom input validation for subclasses.
        
        Returns:
            Validation result dictionary
        """
        return {"valid": True}
    
    def _get_path_config_keys(self) -> List[str]:
        """
        Get list of configuration keys that should contain paths.
        
        Returns:
            List of path configuration keys
        """
        return ["input_file", "input_directory", "output_dir", "aoi_file"]
    
    def _needs_temp_directory(self) -> bool:
        """
        Check if processing needs a temporary directory.
        
        Returns:
            True if temp directory needed
        """
        return True  # Default to creating temp directory
    
    def _estimate_total_items(self) -> int:
        """
        Estimate total number of items to process for progress tracking.
        
        Returns:
            Estimated number of items (0 = no progress tracking)
        """
        return 0  # Default to no progress tracking
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """
        Custom setup logic for subclasses.
        
        Returns:
            Setup information dictionary
        """
        return {}
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """
        Custom cleanup logic for subclasses.
        
        Returns:
            Cleanup information dictionary
        """
        return {}
    
    # Utility methods for subclasses
    
    def get_temp_directory(self) -> Path:
        """Get or create a temporary directory for processing."""
        if not hasattr(self, '_temp_dir'):
            self._temp_dir = self.resource_manager.create_temp_directory()
        return self._temp_dir
    
    def get_temp_file(self, suffix: str = "") -> Path:
        """Get a temporary file path."""
        return self.resource_manager.create_temp_file(suffix=suffix)
    
    def update_progress(self, increment: int = 1, description: Optional[str] = None):
        """Update progress tracking."""
        if self.progress_tracker:
            self.progress_tracker.update(increment, description)
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to tracking."""
        self.metrics.add_metric(name, value)
    
    def log_processing_step(self, step: str, details: Optional[str] = None):
        """Log a processing step with consistent formatting."""
        if details:
            self.logger.info(f"{step}: {details}")
        else:
            self.logger.info(step)


class GeospatialProcessorMixin:
    """
    Mixin class providing common geospatial processing utilities.
    
    This mixin can be used with TemplateMethodProcessor to add
    geospatial-specific functionality.
    """
    
    def validate_geospatial_inputs(self) -> Dict[str, Any]:
        """Validate geospatial-specific inputs."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "crs_info": {},
            "spatial_info": {}
        }
        
        # This would contain geospatial validation logic
        # Will be implemented when we create actual geospatial processors
        
        return validation_result
    
    def setup_geospatial_processing(self) -> Dict[str, Any]:
        """Set up geospatial processing resources."""
        setup_info = {
            "crs_validation_completed": False,
            "spatial_index_created": False
        }
        
        # Geospatial setup logic would go here
        
        return setup_info
    
    def cleanup_geospatial_resources(self) -> Dict[str, Any]:
        """Clean up geospatial processing resources."""
        cleanup_info = {
            "spatial_indexes_cleared": False,
            "memory_released": False
        }
        
        # Geospatial cleanup logic would go here
        
        return cleanup_info


# Example of how to use the enhanced base processor
class ExampleEnhancedProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """Example processor showing how to use the enhanced base class."""
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """Custom validation for this processor."""
        validation_result = super()._validate_custom_inputs()
        
        # Add geospatial validation
        geo_validation = self.validate_geospatial_inputs()
        validation_result.update(geo_validation)
        
        # Add processor-specific validation
        if "input_file" not in self.config:
            validation_result["valid"] = False
            validation_result["errors"].append("input_file is required")
        
        return validation_result
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Custom setup for this processor."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Add processor-specific setup
        setup_info["example_setup"] = "completed"
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """Main processing logic."""
        result = ProcessingResult(success=True)
        
        # Example processing
        self.log_processing_step("Starting example processing")
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        result.processed_count = 1
        result.message = "Example processing completed"
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Custom cleanup for this processor."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Add processor-specific cleanup
        cleanup_info["example_cleanup"] = "completed"
        
        return cleanup_info