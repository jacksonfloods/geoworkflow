# File: src/geoworkflow/core/pipeline.py
"""
Processing pipeline orchestration for the geoworkflow package.

This module provides the main pipeline class that coordinates the execution
of multiple processing stages in the correct order.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime

from geoworkflow.core.base import BaseProcessor, ProcessingResult
from geoworkflow.schemas.config_models import WorkflowConfig
from geoworkflow.core.exceptions import ProcessingError, ConfigurationError
from geoworkflow.core.logging_setup import get_logger
from geoworkflow.core.constants import ProcessingStage


@dataclass
class PipelineResult:
    """Result object for pipeline execution."""
    success: bool
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    stage_results: Dict[str, ProcessingResult] = field(default_factory=dict)
    total_elapsed_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def total_files_processed(self) -> int:
        """Total files processed across all stages."""
        return sum(result.processed_count for result in self.stage_results.values())
    
    @property
    def total_files_failed(self) -> int:
        """Total files that failed across all stages."""
        return sum(result.failed_count for result in self.stage_results.values())


class ProcessingPipeline:
    """
    Main processing pipeline for coordinating geospatial workflows.
    
    The pipeline manages the execution of multiple processing stages,
    handles dependencies between stages, and provides comprehensive
    logging and error handling.
    """
    
    def __init__(self, config: WorkflowConfig, quiet: bool = False):
        """
        Initialize the processing pipeline.
        
        Args:
            config: Workflow configuration
            quiet: Whether to suppress verbose output
        """
        self.config = config
        self.quiet = quiet
        self.logger = get_logger("Pipeline", quiet=quiet)
        
        # Registry of available processors (will be populated by imports)
        self._processor_registry: Dict[str, Type[BaseProcessor]] = {}
        
        # Pipeline state
        self._current_stage: Optional[str] = None
        self._pipeline_result: Optional[PipelineResult] = None
        
        # Register built-in processors
        self._register_processors()
    
    def _register_processors(self):
        """Register available processors."""
        # This will be populated as we implement processors
        # For now, we'll define the expected processor mapping
        self._stage_processor_map = {
            "extract": "ArchiveProcessor",
            "clip": "ClippingProcessor", 
            "align": "AlignmentProcessor",
            "integrate": "IntegrationProcessor",
            "visualize": "VisualizationProcessor",
        }
    
    def register_processor(self, stage_name: str, processor_class: Type[BaseProcessor]):
        """
        Register a processor for a specific stage.
        
        Args:
            stage_name: Name of the processing stage
            processor_class: Processor class to register
        """
        self._processor_registry[stage_name] = processor_class
        self.logger.debug(f"Registered processor {processor_class.__name__} for stage '{stage_name}'")
    
    def run(self, start_stage: Optional[str] = None, end_stage: Optional[str] = None) -> PipelineResult:
        """
        Run the complete processing pipeline.
        
        Args:
            start_stage: Stage to start from (None = start from beginning)
            end_stage: Stage to end at (None = run all stages)
            
        Returns:
            PipelineResult with comprehensive execution information
        """
        start_time = time.time()
        
        self._pipeline_result = PipelineResult(
            success=False,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Starting pipeline: {self.config.name}")
        if self.config.description:
            self.logger.info(f"Description: {self.config.description}")
        
        try:
            # Validate configuration
            self._validate_pipeline_config()
            
            # Determine stages to run
            stages_to_run = self._get_stages_to_run(start_stage, end_stage)
            self.logger.info(f"Pipeline will execute {len(stages_to_run)} stages: {', '.join(stages_to_run)}")
            
            # Execute stages
            for stage in stages_to_run:
                self.logger.info(f"Executing stage: {stage}")
                stage_result = self.run_stage(stage)
                
                self._pipeline_result.stage_results[stage] = stage_result
                
                if stage_result.success:
                    self._pipeline_result.stages_completed.append(stage)
                    self.logger.info(f"Stage '{stage}' completed successfully")
                else:
                    self._pipeline_result.stages_failed.append(stage)
                    error_msg = f"Stage '{stage}' failed: {stage_result.message}"
                    self.logger.error(error_msg)
                    
                    # Stop pipeline on failure (could be made configurable)
                    self._pipeline_result.error_message = error_msg
                    break
            
            # Pipeline succeeded if no stages failed
            self._pipeline_result.success = len(self._pipeline_result.stages_failed) == 0
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            self._pipeline_result.error_message = error_msg
            self._pipeline_result.success = False
        
        finally:
            # Calculate timing
            self._pipeline_result.total_elapsed_time = time.time() - start_time
            self._pipeline_result.end_time = datetime.now()
            
            # Log summary
            self._log_pipeline_summary()
        
        return self._pipeline_result
    
    def run_stage(self, stage_name: str, **kwargs) -> ProcessingResult:
        """
        Run a specific processing stage.
        
        Args:
            stage_name: Name of the stage to run
            **kwargs: Additional arguments to pass to the processor
            
        Returns:
            ProcessingResult for the stage
        """
        self._current_stage = stage_name
        
        try:
            # Get processor class for this stage
            if stage_name not in self._processor_registry:
                raise ProcessingError(
                    f"No processor registered for stage '{stage_name}'. "
                    f"Available stages: {list(self._processor_registry.keys())}"
                )
            
            processor_class = self._processor_registry[stage_name]
            
            # Create stage-specific configuration
            stage_config = self._create_stage_config(stage_name, **kwargs)
            
            # Initialize and run processor
            processor = processor_class(stage_config, quiet=self.quiet)
            result = processor.process()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stage '{stage_name}' failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=str(e),
                details={"stage": stage_name, "error_type": type(e).__name__}
            )
    
    def _validate_pipeline_config(self):
        """Validate pipeline configuration."""
        # Check that required directories exist or can be created
        directories_to_check = [
            self.config.source_dir,
            self.config.aoi_dir,
        ]
        
        directories_to_create = [
            self.config.extracted_dir,
            self.config.clipped_dir,
            self.config.processed_dir,
            self.config.analysis_ready_dir,
            self.config.output_dir,
        ]
        
        # Validate existing directories
        for directory in directories_to_check:
            if not directory.exists():
                raise ConfigurationError(f"Required directory does not exist: {directory}")
        
        # Create output directories
        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_stages_to_run(self, start_stage: Optional[str], end_stage: Optional[str]) -> List[str]:
        """Determine which stages to run based on start/end parameters."""
        all_stages = self.config.processing.stages
        
        if start_stage is None and end_stage is None:
            return all_stages
        
        start_idx = 0 if start_stage is None else all_stages.index(start_stage)
        end_idx = len(all_stages) if end_stage is None else all_stages.index(end_stage) + 1
        
        return all_stages[start_idx:end_idx]
    
    def _create_stage_config(self, stage_name: str, **kwargs) -> Dict[str, Any]:
        """Create configuration dictionary for a specific stage."""
        # Base configuration from workflow config
        stage_config = {
            "stage_name": stage_name,
            "workflow_name": self.config.name,
            "aoi_config": self.config.aoi.dict(),
            "processing_config": self.config.processing.dict(),
            "data_sources": {name: source.dict() for name, source in self.config.data_sources.items()},
            
            # Directory paths
            "source_dir": self.config.source_dir,
            "extracted_dir": self.config.extracted_dir,
            "clipped_dir": self.config.clipped_dir,
            "processed_dir": self.config.processed_dir,
            "analysis_ready_dir": self.config.analysis_ready_dir,
            "aoi_dir": self.config.aoi_dir,
            "output_dir": self.config.output_dir,
        }
        
        # Add any additional kwargs
        stage_config.update(kwargs)
        
        return stage_config
    
    def _log_pipeline_summary(self):
        """Log a summary of pipeline execution."""
        if not self._pipeline_result:
            return
        
        result = self._pipeline_result
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline: {self.config.name}")
        self.logger.info(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        self.logger.info(f"Total elapsed time: {result.total_elapsed_time:.2f} seconds")
        self.logger.info(f"Stages completed: {len(result.stages_completed)}")
        self.logger.info(f"Stages failed: {len(result.stages_failed)}")
        
        if result.stages_completed:
            self.logger.info(f"Completed stages: {', '.join(result.stages_completed)}")
        
        if result.stages_failed:
            self.logger.info(f"Failed stages: {', '.join(result.stages_failed)}")
        
        # File processing summary
        total_processed = result.total_files_processed
        total_failed = result.total_files_failed
        
        if total_processed > 0 or total_failed > 0:
            self.logger.info(f"Files processed: {total_processed}")
            self.logger.info(f"Files failed: {total_failed}")
        
        if result.error_message:
            self.logger.error(f"Error: {result.error_message}")
        
        self.logger.info("=" * 60)
    
    def get_stage_output_dir(self, stage_name: str) -> Path:
        """Get the output directory for a specific stage."""
        stage_dir_map = {
            "extract": self.config.extracted_dir,
            "clip": self.config.clipped_dir,
            "align": self.config.processed_dir / "aligned",
            "integrate": self.config.analysis_ready_dir,
            "visualize": self.config.output_dir / "visualizations",
        }
        
        return stage_dir_map.get(stage_name, self.config.output_dir)
    
    def get_stage_input_dir(self, stage_name: str) -> Path:
        """Get the input directory for a specific stage."""
        stage_input_map = {
            "extract": self.config.source_dir,
            "clip": self.config.extracted_dir,
            "align": self.config.clipped_dir,
            "integrate": self.config.processed_dir,
            "visualize": self.config.analysis_ready_dir,
        }
        
        return stage_input_map.get(stage_name, self.config.source_dir)

# File: Update to src/geoworkflow/core/pipeline.py
# Add these methods to the ProcessingPipeline class

def _register_processors(self):
    """Register available processors."""
    # Import processors when registering to avoid circular imports
    try:
        from ..processors.aoi.processor import AOIProcessor
        from ..processors.spatial.clipper import ClippingProcessor
        from ..processors.spatial.aligner import AlignmentProcessor
        from ..processors.integration.enrichment import StatisticalEnrichmentProcessor
        from ..processors.extraction.archive import ArchiveExtractionProcessor
        
        # Register processors for each stage
        self.register_processor("aoi", AOIProcessor)
        self.register_processor("extract", ArchiveExtractionProcessor) 
        self.register_processor("clip", ClippingProcessor)
        self.register_processor("align", AlignmentProcessor)
        self.register_processor("enrich", StatisticalEnrichmentProcessor)
        # self.register_processor("visualize", VisualizationProcessor)  # When implemented
        
        self.logger.debug("Successfully registered all processors")
        
    except ImportError as e:
        self.logger.warning(f"Could not import all processors: {e}")
        # Define the expected processor mapping as fallback
        self._stage_processor_map = {
            "extract": "ArchiveExtractionProcessor",
            "clip": "ClippingProcessor", 
            "align": "AlignmentProcessor",
            "enrich": "StatisticalEnrichmentProcessor",
            "integrate": "IntegrationProcessor",
            "visualize": "VisualizationProcessor",
        }

def _create_stage_config(self, stage_name: str, **kwargs) -> Dict[str, Any]:
    """Create configuration dictionary for a specific stage."""
    # Get stage-specific configuration from workflow config
    stage_config = {}
    
    if stage_name == "aoi":
        stage_config = self.config.aoi.to_dict()
    elif stage_name == "extract":
        stage_config = self.config.extraction or {}
        # Add required fields for extraction
        stage_config.update({
            "aoi_file": self.config.aoi.output_file,
        })
    elif stage_name == "clip":
        stage_config = self.config.clipping or {}
        stage_config.update({
            "aoi_file": self.config.aoi.output_file,
            "input_directory": self.get_stage_input_dir(stage_name),
            "output_dir": self.get_stage_output_dir(stage_name),
        })
    elif stage_name == "align":
        stage_config = self.config.alignment or {}
        stage_config.update({
            "input_directory": self.get_stage_input_dir(stage_name),
            "output_dir": self.get_stage_output_dir(stage_name),
        })
    elif stage_name == "enrich":
        if self.config.enrichment:
            stage_config = self.config.enrichment.to_dict()
        else:
            # Create default enrichment config
            stage_config = {
                "coi_directory": self.get_stage_input_dir(stage_name),
                "coi_pattern": "*AFRICAPOLIS*",
                "raster_directory": self.get_stage_input_dir(stage_name),
                "raster_pattern": "*.tif",
                "output_file": self.get_stage_output_dir(stage_name) / "enriched_cities.geojson",
                "statistics": ["mean", "max", "min"],
                "skip_existing": True,
                "add_area_column": True,
                "area_units": "km2"
            }
    elif stage_name == "visualize":
        if self.config.visualization:
            stage_config = self.config.visualization.to_dict()
        stage_config.update({
            "input_directory": self.get_stage_input_dir(stage_name),
            "output_dir": self.get_stage_output_dir(stage_name),
        })
    
    # Add common configuration
    stage_config.update({
        "stage_name": stage_name,
        "workflow_name": self.config.name,
        "data_sources": self.config.data_sources,
    })
    
    # Apply any additional kwargs
    stage_config.update(kwargs)
    
    return stage_config

def get_stage_output_dir(self, stage_name: str) -> Path:
    """Get the output directory for a specific stage."""
    # Base directories from workflow config or default structure
    base_data_dir = Path("data")
    base_output_dir = Path("outputs")
    
    stage_dir_map = {
        "aoi": base_data_dir / "aoi",
        "extract": base_data_dir / "01_extracted",
        "clip": base_data_dir / "02_clipped",
        "align": base_data_dir / "03_processed" / "aligned",
        "enrich": base_data_dir / "04_analysis_ready",
        "integrate": base_data_dir / "04_analysis_ready",
        "visualize": base_output_dir / "visualizations",
    }
    
    return stage_dir_map.get(stage_name, base_output_dir)

def get_stage_input_dir(self, stage_name: str) -> Path:
    """Get the input directory for a specific stage."""
    base_data_dir = Path("data")
    
    stage_input_map = {
        "aoi": base_data_dir / "00_source" / "boundaries",
        "extract": base_data_dir / "00_source" / "archives",
        "clip": base_data_dir / "01_extracted",
        "align": base_data_dir / "02_clipped", 
        "enrich": base_data_dir / "03_processed" / "aligned",  # Prefer aligned data
        "integrate": base_data_dir / "03_processed",
        "visualize": base_data_dir / "04_analysis_ready",
    }
    
    input_dir = stage_input_map.get(stage_name, base_data_dir)
    
    # For enrichment, fallback to clipped data if aligned doesn't exist
    if stage_name == "enrich" and not input_dir.exists():
        fallback_dir = base_data_dir / "02_clipped"
        if fallback_dir.exists():
            self.logger.info(f"Using clipped data for enrichment: {fallback_dir}")
            return fallback_dir
    
    return input_dir

# Also add this method to handle enrichment stage validation
def _validate_enrichment_stage(self) -> Dict[str, Any]:
    """Validate enrichment stage requirements."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    if "enrich" not in self.config.stages:
        return validation_result
    
    # Check that either clipped or aligned data exists
    clipped_dir = Path("data/02_clipped")
    aligned_dir = Path("data/03_processed/aligned")
    
    if not clipped_dir.exists() and not aligned_dir.exists():
        validation_result["errors"].append(
            "Enrichment stage requires either clipped or aligned raster data"
        )
        validation_result["valid"] = False
    
    # Check for COI data
    input_dir = self.get_stage_input_dir("enrich")
    if input_dir.exists():
        # Look for potential COI files
        coi_pattern = "*AFRICAPOLIS*"
        if self.config.enrichment and self.config.enrichment.coi_pattern:
            coi_pattern = self.config.enrichment.coi_pattern
        
        import fnmatch
        coi_files = []
        for file_path in input_dir.rglob("*"):
            if file_path.is_file() and fnmatch.fnmatch(file_path.name, coi_pattern):
                if file_path.suffix.lower() in {'.shp', '.geojson', '.gpkg', '.gml', '.kml'}:
                    coi_files.append(file_path)
        
        if not coi_files:
            validation_result["warnings"].append(
                f"No COI files found matching pattern '{coi_pattern}' in {input_dir}"
            )
        elif len(coi_files) > 1:
            validation_result["warnings"].append(
                f"Multiple COI files found matching pattern '{coi_pattern}'. "
                f"Processor will require more specific pattern."
            )
    
    return validation_result