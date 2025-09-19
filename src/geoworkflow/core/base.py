"""
Base classes for the geoworkflow package.

This module provides abstract base classes that define the interface
for processors, visualizers, and other components in the geoworkflow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

from ..core.exceptions import ProcessingError, ConfigurationError
from ..core.constants import ProcessingStage


@dataclass
class ProcessingResult:
    """Standard result object for all processing operations."""
    
    success: bool
    processed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failed_files: Optional[List[str]] = None
    elapsed_time: float = 0.0
    message: Optional[str] = None
    stage: Optional[ProcessingStage] = None
    output_paths: Optional[List[Path]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []
        if self.output_paths is None:
            self.output_paths = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_files(self) -> int:
        """Total number of files processed."""
        return self.processed_count + self.failed_count + self.skipped_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        # Convert Path objects to strings for serialization
        if self.output_paths:
            result_dict['output_paths'] = [str(p) for p in self.output_paths]
        if self.stage:
            result_dict['stage'] = self.stage.value
        return result_dict
    
    def add_failed_file(self, file_path: Union[str, Path]):
        """Add a file to the failed files list."""
        self.failed_files.append(str(file_path))
        self.failed_count += 1
    
    def add_output_path(self, path: Union[str, Path]):
        """Add an output path to the result."""
        self.output_paths.append(Path(path))


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the processor.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = self._validate_config(config)
        self.logger = logger or self._setup_logger()
        self._start_time: Optional[datetime] = None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this processor."""
        logger_name = f"geoworkflow.{self.__class__.__name__}"
        return logging.getLogger(logger_name)
    
    @abstractmethod
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def process(self) -> ProcessingResult:
        """
        Execute the main processing logic.
        
        Returns:
            ProcessingResult object with operation results
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    def _start_processing(self):
        """Mark the start of processing for timing."""
        self._start_time = datetime.now()
        self.logger.info(f"Starting {self.__class__.__name__} processing")
    
    def _finish_processing(self, result: ProcessingResult) -> ProcessingResult:
        """Mark the end of processing and calculate elapsed time."""
        if self._start_time:
            end_time = datetime.now()
            result.elapsed_time = (end_time - self._start_time).total_seconds()
        
        # Log summary
        self.logger.info(f"Completed {self.__class__.__name__} processing:")
        self.logger.info(f"  Success: {result.success}")
        self.logger.info(f"  Processed: {result.processed_count}")
        self.logger.info(f"  Failed: {result.failed_count}")
        self.logger.info(f"  Skipped: {result.skipped_count}")
        self.logger.info(f"  Elapsed time: {result.elapsed_time:.2f}s")
        
        return result
    
    def _validate_input_path(self, path: Union[str, Path], must_exist: bool = True) -> Path:
        """
        Validate an input path.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            
        Returns:
            Validated Path object
            
        Raises:
            ConfigurationError: If path is invalid
        """
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise ConfigurationError(f"Path does not exist: {path_obj}")
            
        return path_obj
    
    def _ensure_output_dir(self, path: Union[str, Path]) -> Path:
        """
        Ensure output directory exists.
        
        Args:
            path: Output directory path
            
        Returns:
            Path object for the directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj


class BaseVisualizer(ABC):
    """Abstract base class for all visualizers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 quiet: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            config: Optional configuration dictionary
            quiet: Whether to suppress verbose output
        """
        self.config = config or {}
        self.quiet = quiet
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this visualizer."""
        logger_name = f"geoworkflow.{self.__class__.__name__}"
        logger = logging.getLogger(logger_name)
        if self.quiet:
            logger.setLevel(logging.WARNING)
        return logger
    
    @abstractmethod
    def visualize(self, input_path: Union[str, Path], 
                  output_path: Union[str, Path], **kwargs) -> bool:
        """
        Create visualization.
        
        Args:
            input_path: Path to input data
            output_path: Path for output visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            True if successful, False otherwise
        """
        pass


class BaseConfig(ABC):
    """Abstract base class for configuration objects."""
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration object
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'BaseConfig':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration object
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        pass


class Pipeline:
    """Base pipeline class for orchestrating multiple processors."""
    
    def __init__(self, name: str = "GeoWorkflow Pipeline"):
        """
        Initialize the pipeline.
        
        Args:
            name: Pipeline name for logging
        """
        self.name = name
        self.processors: List[BaseProcessor] = []
        self.logger = logging.getLogger(f"geoworkflow.{self.__class__.__name__}")
    
    def add_processor(self, processor: BaseProcessor):
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor instance to add
        """
        self.processors.append(processor)
        self.logger.debug(f"Added processor: {processor.__class__.__name__}")
    
    def run(self, stop_on_error: bool = True) -> Dict[str, ProcessingResult]:
        """
        Run all processors in the pipeline.
        
        Args:
            stop_on_error: Whether to stop on first error
            
        Returns:
            Dictionary mapping processor names to results
            
        Raises:
            ProcessingError: If any processor fails and stop_on_error is True
        """
        results = {}
        
        self.logger.info(f"Starting pipeline: {self.name}")
        self.logger.info(f"Processors to run: {len(self.processors)}")
        
        for i, processor in enumerate(self.processors, 1):
            processor_name = processor.__class__.__name__
            self.logger.info(f"Running processor {i}/{len(self.processors)}: {processor_name}")
            
            try:
                result = processor.process()
                results[processor_name] = result
                
                if not result.success and stop_on_error:
                    raise ProcessingError(
                        f"Pipeline stopped due to failure in {processor_name}",
                        details={'failed_processor': processor_name}
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in processor {processor_name}: {e}")
                
                # Create error result
                error_result = ProcessingResult(
                    success=False,
                    message=str(e)
                )
                results[processor_name] = error_result
                
                if stop_on_error:
                    raise ProcessingError(
                        f"Pipeline failed at processor {processor_name}: {e}",
                        details={'failed_processor': processor_name}
                    )
        
        self.logger.info(f"Pipeline completed: {self.name}")
        return results
