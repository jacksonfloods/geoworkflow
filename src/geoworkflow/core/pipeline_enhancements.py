# File: src/geoworkflow/core/pipeline_enhancements.py
"""
Enhanced methods to add to the existing ProcessingPipeline class.

These methods provide dependency validation, data flow checking,
and comprehensive error reporting for pipeline operations.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import logging
from datetime import datetime

from geoworkflow.core.exceptions import PipelineError, ValidationError
from ..utils.progress_utils import BatchProgressTracker, PerformanceMonitor
try:
    from geoworkflow.utils.validation import validate_raster_file, validate_vector_file
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    def validate_raster_file(*args, **kwargs):
        return {"path": "unknown", "valid": True, "message": "Validation not available"}
    def validate_vector_file(*args, **kwargs):
        return {"path": "unknown", "valid": True, "message": "Validation not available"}


class PipelineEnhancementMixin:
    """
    Mixin class with enhanced pipeline methods.
    
    This should be mixed into the existing ProcessingPipeline class
    to add enhanced functionality without breaking existing code.
    """
    
    def validate_pipeline_dependencies(self) -> Dict[str, Any]:
        """
        Validate dependencies between pipeline stages.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stage_dependencies": {},
            "data_flow": {}
        }
        
        # Define stage dependencies
        dependency_map = {
            "extract": [],  # No dependencies
            "clip": ["extract"],  # Needs extracted data
            "align": ["clip"],  # Needs clipped data
            "integrate": ["align"],  # Needs aligned data
            "visualize": ["integrate"]  # Needs integrated data (or any processed data)
        }
        
        # Check that required stages are present
        stages_to_run = getattr(self.config, 'stages', [])
        
        for stage in stages_to_run:
            if stage in dependency_map:
                required_stages = dependency_map[stage]
                validation_result["stage_dependencies"][stage] = required_stages
                
                # Check if dependencies are satisfied
                for required_stage in required_stages:
                    if required_stage not in stages_to_run:
                        # Check if required data already exists
                        if not self._check_stage_data_exists(required_stage):
                            validation_result["errors"].append(
                                f"Stage '{stage}' requires '{required_stage}' but it's not in pipeline "
                                f"and required data doesn't exist"
                            )
                            validation_result["valid"] = False
                        else:
                            validation_result["warnings"].append(
                                f"Stage '{stage}' requires '{required_stage}' data - using existing data"
                            )
        
        return validation_result
    
    def validate_data_flow(self) -> Dict[str, Any]:
        """
        Validate that data flows correctly between stages.
        
        Returns:
            Dictionary with data flow validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "input_outputs": {},
            "missing_inputs": [],
            "orphaned_outputs": []
        }
        
        # Map stages to their input/output requirements
        stage_io_map = {
            "extract": {
                "inputs": ["source_archives"],
                "outputs": ["extracted_data"]
            },
            "clip": {
                "inputs": ["extracted_data", "aoi_file"],
                "outputs": ["clipped_data"]
            },
            "align": {
                "inputs": ["clipped_data", "reference_raster"],
                "outputs": ["aligned_data"]
            },
            "integrate": {
                "inputs": ["aligned_data"],
                "outputs": ["analysis_ready_data"]
            },
            "visualize": {
                "inputs": ["analysis_ready_data"],
                "outputs": ["visualizations"]
            }
        }
        
        stages_to_run = getattr(self.config, 'stages', [])
        
        # Check input/output flow
        available_data = set(["source_archives", "aoi_file"])  # Initial data
        
        for stage in stages_to_run:
            if stage in stage_io_map:
                stage_info = stage_io_map[stage]
                validation_result["input_outputs"][stage] = stage_info
                
                # Check inputs are available
                for required_input in stage_info["inputs"]:
                    if required_input not in available_data:
                        validation_result["missing_inputs"].append(
                            f"Stage '{stage}' requires '{required_input}' but it's not available"
                        )
                        validation_result["valid"] = False
                
                # Add outputs to available data
                available_data.update(stage_info["outputs"])
        
        return validation_result
    
    def validate_file_availability(self) -> Dict[str, Any]:
        """
        Validate that required files are available and accessible.
        
        Returns:
            Dictionary with file validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_checks": {}
        }
        
        # Check source directories
        if hasattr(self.config, 'source_dir'):
            source_dir = Path(self.config.source_dir)
            if not source_dir.exists():
                validation_result["errors"].append(f"Source directory does not exist: {source_dir}")
                validation_result["valid"] = False
            else:
                # Check for archives
                archive_files = list(source_dir.rglob("*.zip"))
                validation_result["file_checks"]["archive_count"] = len(archive_files)
                if len(archive_files) == 0:
                    validation_result["warnings"].append("No ZIP archives found in source directory")
        
        # Check AOI file
        if hasattr(self.config, 'aoi_file'):
            aoi_file = Path(self.config.aoi_file)
            if not aoi_file.exists():
                validation_result["errors"].append(f"AOI file does not exist: {aoi_file}")
                validation_result["valid"] = False
            else:
                try:
                    # Validate AOI file format
                    aoi_info = validate_vector_file(aoi_file)
                    validation_result["file_checks"]["aoi_info"] = aoi_info
                except Exception as e:
                    validation_result["errors"].append(f"AOI file validation failed: {e}")
                    validation_result["valid"] = False
        
        return validation_result
    
    def run_with_enhanced_tracking(self, start_stage: Optional[str] = None, 
                                 end_stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Run pipeline with enhanced progress tracking and monitoring.
        
        Args:
            start_stage: Stage to start from
            end_stage: Stage to end at
            
        Returns:
            Enhanced pipeline results with detailed tracking
        """
        # Initialize enhanced tracking
        stages_to_run = self._get_stages_to_run(start_stage, end_stage)
        
        batch_tracker = BatchProgressTracker(
            stages=stages_to_run,
            quiet=getattr(self, 'quiet', False)
        )
        
        performance_monitor = PerformanceMonitor()
        performance_monitor.start()
        
        try:
            # Pre-flight validation
            validation_results = self.run_comprehensive_validation()
            if not validation_results["overall_valid"]:
                raise PipelineError(
                    f"Pipeline validation failed: {validation_results['errors']}"
                )
            
            # Run pipeline with tracking
            performance_monitor.checkpoint("validation_complete")
            
            pipeline_result = self.run(start_stage, end_stage)
            
            performance_monitor.checkpoint("pipeline_complete")
            
            # Enhanced result
            enhanced_result = {
                "pipeline_result": pipeline_result,
                "validation_results": validation_results,
                "performance_metrics": performance_monitor.get_summary(),
                "tracking_info": {
                    "stages_tracked": stages_to_run,
                    "enhanced_tracking_used": True
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            performance_monitor.checkpoint("error_occurred")
            raise PipelineError(f"Enhanced pipeline execution failed: {e}")
        
        finally:
            batch_tracker.close()
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of the entire pipeline.
        
        Returns:
            Comprehensive validation results
        """
        validation_results = {
            "overall_valid": True,
            "errors": [],
            "warnings": [],
            "validation_components": {}
        }
        
        # 1. Validate dependencies
        dep_validation = self.validate_pipeline_dependencies()
        validation_results["validation_components"]["dependencies"] = dep_validation
        if not dep_validation["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(dep_validation["errors"])
        validation_results["warnings"].extend(dep_validation["warnings"])
        
        # 2. Validate data flow
        flow_validation = self.validate_data_flow()
        validation_results["validation_components"]["data_flow"] = flow_validation
        if not flow_validation["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(flow_validation["errors"])
        validation_results["warnings"].extend(flow_validation["warnings"])
        
        # 3. Validate file availability
        file_validation = self.validate_file_availability()
        validation_results["validation_components"]["files"] = file_validation
        if not file_validation["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(file_validation["errors"])
        validation_results["warnings"].extend(file_validation["warnings"])
        
        # 4. Validate configuration
        config_validation = self._validate_pipeline_config_enhanced()
        validation_results["validation_components"]["configuration"] = config_validation
        if not config_validation["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(config_validation["errors"])
        validation_results["warnings"].extend(config_validation["warnings"])
        
        return validation_results
    
    def get_stage_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all pipeline stages.
        
        Returns:
            Dictionary with stage status information
        """
        status = {
            "stages": {},
            "overall_status": "unknown",
            "completion_percentage": 0.0
        }
        
        stages_to_run = getattr(self.config, 'stages', [])
        completed_stages = 0
        
        for stage in stages_to_run:
            stage_status = {
                "data_exists": self._check_stage_data_exists(stage),
                "output_dir_exists": self._check_stage_output_dir_exists(stage),
                "estimated_completion": self._estimate_stage_completion(stage)
            }
            
            # Determine stage status
            if stage_status["estimated_completion"] >= 100:
                stage_status["status"] = "completed"
                completed_stages += 1
            elif stage_status["estimated_completion"] > 0:
                stage_status["status"] = "partial"
            else:
                stage_status["status"] = "not_started"
            
            status["stages"][stage] = stage_status
        
        # Calculate overall status
        if completed_stages == len(stages_to_run):
            status["overall_status"] = "completed"
        elif completed_stages > 0:
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "not_started"
        
        status["completion_percentage"] = (completed_stages / len(stages_to_run)) * 100 if stages_to_run else 0
        
        return status
    
    def cleanup_pipeline_artifacts(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clean up temporary artifacts from pipeline execution.
        
        Args:
            confirm: Whether to actually perform cleanup (safety check)
            
        Returns:
            Cleanup results
        """
        cleanup_results = {
            "files_removed": 0,
            "directories_removed": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        if not confirm:
            cleanup_results["message"] = "Cleanup not performed - set confirm=True to execute"
            return cleanup_results
        
        # Clean up temp directories, logs, cache files etc.
        # Implementation would go here
        
        return cleanup_results
    
    # Helper methods
    
    def _check_stage_data_exists(self, stage: str) -> bool:
        """Check if output data exists for a stage."""
        try:
            stage_output_dir = self.get_stage_output_dir(stage)
            return stage_output_dir.exists() and any(stage_output_dir.iterdir())
        except:
            return False
    
    def _check_stage_output_dir_exists(self, stage: str) -> bool:
        """Check if output directory exists for a stage."""
        try:
            stage_output_dir = self.get_stage_output_dir(stage)
            return stage_output_dir.exists()
        except:
            return False
    
    def _estimate_stage_completion(self, stage: str) -> float:
        """Estimate completion percentage for a stage."""
        try:
            stage_output_dir = self.get_stage_output_dir(stage)
            if not stage_output_dir.exists():
                return 0.0
            
            # Simple heuristic: if directory has files, assume complete
            files = list(stage_output_dir.glob("**/*"))
            return 100.0 if files else 0.0
        except:
            return 0.0
    
    def _validate_pipeline_config_enhanced(self) -> Dict[str, Any]:
        """Enhanced configuration validation."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required configuration attributes
        required_attrs = ['stages']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                validation_result["errors"].append(f"Missing required config attribute: {attr}")
                validation_result["valid"] = False
        
        # Check stages are valid
        if hasattr(self.config, 'stages'):
            valid_stages = {"extract", "clip", "align", "integrate", "visualize"}
            invalid_stages = set(self.config.stages) - valid_stages
            if invalid_stages:
                validation_result["errors"].append(f"Invalid stages: {invalid_stages}")
                validation_result["valid"] = False
        
        return validation_result


# Usage example - this would be mixed into the existing ProcessingPipeline class:
"""
from .pipeline_enhancements import PipelineEnhancementMixin

class EnhancedProcessingPipeline(ProcessingPipeline, PipelineEnhancementMixin):
    pass

# Or you could modify the existing ProcessingPipeline class by adding these methods directly
"""