<a id="geoworkflow.core.base"></a>

# geoworkflow.core.base

Base classes for the geoworkflow package.

This module provides abstract base classes that define the interface
for processors, visualizers, and other components in the geoworkflow system.

<a id="geoworkflow.core.base.ProcessingResult"></a>

## ProcessingResult Objects

```python
@dataclass
class ProcessingResult()
```

Standard result object for all processing operations.

<a id="geoworkflow.core.base.ProcessingResult.total_files"></a>

#### total\_files

```python
@property
def total_files() -> int
```

Total number of files processed.

<a id="geoworkflow.core.base.ProcessingResult.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert result to dictionary.

<a id="geoworkflow.core.base.ProcessingResult.add_failed_file"></a>

#### add\_failed\_file

```python
def add_failed_file(file_path: Union[str, Path])
```

Add a file to the failed files list.

<a id="geoworkflow.core.base.ProcessingResult.add_output_path"></a>

#### add\_output\_path

```python
def add_output_path(path: Union[str, Path])
```

Add an output path to the result.

<a id="geoworkflow.core.base.BaseProcessor"></a>

## BaseProcessor Objects

```python
class BaseProcessor(ABC)
```

Abstract base class for all data processors.

<a id="geoworkflow.core.base.BaseProcessor.process"></a>

#### process

```python
@abstractmethod
def process() -> ProcessingResult
```

Execute the main processing logic.

**Returns**:

  ProcessingResult object with operation results
  

**Raises**:

- `ProcessingError` - If processing fails

<a id="geoworkflow.core.base.BaseVisualizer"></a>

## BaseVisualizer Objects

```python
class BaseVisualizer(ABC)
```

Abstract base class for all visualizers.

<a id="geoworkflow.core.base.BaseVisualizer.visualize"></a>

#### visualize

```python
@abstractmethod
def visualize(input_path: Union[str, Path], output_path: Union[str, Path],
              **kwargs) -> bool
```

Create visualization.

**Arguments**:

- `input_path` - Path to input data
- `output_path` - Path for output visualization
- `**kwargs` - Additional visualization parameters
  

**Returns**:

  True if successful, False otherwise

<a id="geoworkflow.core.base.BaseConfig"></a>

## BaseConfig Objects

```python
class BaseConfig(ABC)
```

Abstract base class for configuration objects.

<a id="geoworkflow.core.base.BaseConfig.from_dict"></a>

#### from\_dict

```python
@classmethod
@abstractmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig'
```

Create configuration from dictionary.

**Arguments**:

- `config_dict` - Configuration dictionary
  

**Returns**:

  Configuration object

<a id="geoworkflow.core.base.BaseConfig.from_file"></a>

#### from\_file

```python
@classmethod
@abstractmethod
def from_file(cls, config_path: Union[str, Path]) -> 'BaseConfig'
```

Load configuration from file.

**Arguments**:

- `config_path` - Path to configuration file
  

**Returns**:

  Configuration object

<a id="geoworkflow.core.base.BaseConfig.to_dict"></a>

#### to\_dict

```python
@abstractmethod
def to_dict() -> Dict[str, Any]
```

Convert configuration to dictionary.

**Returns**:

  Configuration dictionary

<a id="geoworkflow.core.base.BaseConfig.validate"></a>

#### validate

```python
@abstractmethod
def validate() -> bool
```

Validate configuration.

**Returns**:

  True if valid
  

**Raises**:

- `ConfigurationError` - If validation fails

<a id="geoworkflow.core.base.Pipeline"></a>

## Pipeline Objects

```python
class Pipeline()
```

Base pipeline class for orchestrating multiple processors.

<a id="geoworkflow.core.base.Pipeline.add_processor"></a>

#### add\_processor

```python
def add_processor(processor: BaseProcessor)
```

Add a processor to the pipeline.

**Arguments**:

- `processor` - Processor instance to add

<a id="geoworkflow.core.base.Pipeline.run"></a>

#### run

```python
def run(stop_on_error: bool = True) -> Dict[str, ProcessingResult]
```

Run all processors in the pipeline.

**Arguments**:

- `stop_on_error` - Whether to stop on first error
  

**Returns**:

  Dictionary mapping processor names to results
  

**Raises**:

- `ProcessingError` - If any processor fails and stop_on_error is True

<a id="geoworkflow.core.config"></a>

# geoworkflow.core.config

Configuration management for the geoworkflow package.

This module provides utilities for loading, validating, and managing
configuration files across the geoworkflow system.

<a id="geoworkflow.core.config.ConfigManager"></a>

## ConfigManager Objects

```python
class ConfigManager()
```

Centralized configuration management.

<a id="geoworkflow.core.config.ConfigManager.load_config"></a>

#### load\_config

```python
@staticmethod
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]
```

Load configuration from a file.

Supports JSON and YAML formats.

**Arguments**:

- `config_path` - Path to configuration file
  

**Returns**:

  Configuration dictionary
  

**Raises**:

- `ConfigurationError` - If file cannot be loaded or parsed

<a id="geoworkflow.core.config.ConfigManager.save_config"></a>

#### save\_config

```python
@staticmethod
def save_config(config: Dict[str, Any],
                output_path: Union[str, Path],
                format: str = 'yaml') -> None
```

Save configuration to a file.

**Arguments**:

- `config` - Configuration dictionary
- `output_path` - Path to save configuration
- `format` - Output format ('yaml' or 'json')
  

**Raises**:

- `ConfigurationError` - If saving fails

<a id="geoworkflow.core.config.ConfigManager.validate_required_keys"></a>

#### validate\_required\_keys

```python
@staticmethod
def validate_required_keys(config: Dict[str, Any],
                           required_keys: list) -> None
```

Validate that required keys are present in configuration.

**Arguments**:

- `config` - Configuration dictionary
- `required_keys` - List of required key names
  

**Raises**:

- `ConfigurationError` - If required keys are missing

<a id="geoworkflow.core.config.ConfigManager.validate_paths"></a>

#### validate\_paths

```python
@staticmethod
def validate_paths(config: Dict[str, Any],
                   path_keys: list,
                   must_exist: bool = True) -> None
```

Validate that path values exist.

**Arguments**:

- `config` - Configuration dictionary
- `path_keys` - List of keys that should contain paths
- `must_exist` - Whether paths must exist
  

**Raises**:

- `ConfigurationError` - If paths are invalid

<a id="geoworkflow.core.config.ConfigManager.merge_configs"></a>

#### merge\_configs

```python
@staticmethod
def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]
```

Merge two configuration dictionaries.

**Arguments**:

- `base_config` - Base configuration
- `override_config` - Configuration to merge (takes precedence)
  

**Returns**:

  Merged configuration dictionary

<a id="geoworkflow.core.config.ConfigManager.create_config_object"></a>

#### create\_config\_object

```python
@classmethod
def create_config_object(cls, config_class: Type[ConfigT],
                         config_path: Union[str, Path]) -> ConfigT
```

Create a configuration object from a file.

**Arguments**:

- `config_class` - Configuration class to instantiate
- `config_path` - Path to configuration file
  

**Returns**:

  Configuration object instance

<a id="geoworkflow.core.config.validate_file_extensions"></a>

#### validate\_file\_extensions

```python
def validate_file_extensions(path: Union[str, Path],
                             valid_extensions: list) -> bool
```

Validate that a file has one of the specified extensions.

**Arguments**:

- `path` - File path to check
- `valid_extensions` - List of valid extensions (e.g., ['.tif', '.tiff'])
  

**Returns**:

  True if extension is valid

<a id="geoworkflow.core.config.resolve_path"></a>

#### resolve\_path

```python
def resolve_path(path: Union[str, Path],
                 base_path: Optional[Path] = None) -> Path
```

Resolve a path, making it absolute if it's relative.

**Arguments**:

- `path` - Path to resolve
- `base_path` - Base path for relative resolution (default: current working directory)
  

**Returns**:

  Resolved absolute path

<a id="geoworkflow.core.config.expand_path_pattern"></a>

#### expand\_path\_pattern

```python
def expand_path_pattern(pattern: str, base_dir: Union[str, Path]) -> list
```

Expand a file pattern to list of matching files.

**Arguments**:

- `pattern` - File pattern (e.g., "*.tif")
- `base_dir` - Base directory to search
  

**Returns**:

  List of matching file paths

<a id="geoworkflow.core.config.get_config_template"></a>

#### get\_config\_template

```python
def get_config_template(config_type: str) -> Dict[str, Any]
```

Get a template configuration for a specific processor type.

**Arguments**:

- `config_type` - Type of configuration ('aoi', 'extraction', 'clipping', etc.)
  

**Returns**:

  Template configuration dictionary

<a id="geoworkflow.core.exceptions"></a>

# geoworkflow.core.exceptions

Custom exceptions for the geoworkflow package.

This module defines the exception hierarchy used throughout the geoworkflow
package to provide clear error handling and debugging information.

<a id="geoworkflow.core.exceptions.GeoWorkflowError"></a>

## GeoWorkflowError Objects

```python
class GeoWorkflowError(Exception)
```

Base exception for all geoworkflow errors.

<a id="geoworkflow.core.exceptions.ConfigurationError"></a>

## ConfigurationError Objects

```python
class ConfigurationError(GeoWorkflowError)
```

Raised when there are configuration validation errors.

<a id="geoworkflow.core.exceptions.ProcessingError"></a>

## ProcessingError Objects

```python
class ProcessingError(GeoWorkflowError)
```

Raised when data processing operations fail.

<a id="geoworkflow.core.exceptions.ValidationError"></a>

## ValidationError Objects

```python
class ValidationError(GeoWorkflowError)
```

Raised when data validation fails.

<a id="geoworkflow.core.exceptions.FileOperationError"></a>

## FileOperationError Objects

```python
class FileOperationError(GeoWorkflowError)
```

Raised when file operations fail.

<a id="geoworkflow.core.exceptions.GeospatialError"></a>

## GeospatialError Objects

```python
class GeospatialError(GeoWorkflowError)
```

Raised when geospatial operations fail.

<a id="geoworkflow.core.exceptions.AlignmentError"></a>

## AlignmentError Objects

```python
class AlignmentError(GeospatialError)
```

Raised when raster alignment operations fail.

<a id="geoworkflow.core.exceptions.ClippingError"></a>

## ClippingError Objects

```python
class ClippingError(GeospatialError)
```

Raised when spatial clipping operations fail.

<a id="geoworkflow.core.exceptions.ExtractionError"></a>

## ExtractionError Objects

```python
class ExtractionError(ProcessingError)
```

Raised when data extraction operations fail.

<a id="geoworkflow.core.exceptions.VisualizationError"></a>

## VisualizationError Objects

```python
class VisualizationError(GeoWorkflowError)
```

Raised when visualization operations fail.

<a id="geoworkflow.core.exceptions.ClippingError"></a>

## ClippingError Objects

```python
class ClippingError(GeospatialError)
```

Raised when spatial clipping operations fail.

<a id="geoworkflow.core.exceptions.PipelineError"></a>

## PipelineError Objects

```python
class PipelineError(GeoWorkflowError)
```

Raised when pipeline execution fails.

<a id="geoworkflow.core.exceptions.handle_processing_error"></a>

#### handle\_processing\_error

```python
def handle_processing_error(func)
```

Decorator to handle common processing errors and convert them to GeoWorkflowError.

**Arguments**:

- `func` - Function to wrap with error handling
  

**Returns**:

  Wrapped function with error handling

<a id="geoworkflow.core.constants"></a>

# geoworkflow.core.constants

Global constants for the geoworkflow package.

This module contains constants used throughout the package including
file extensions, coordinate reference systems, and default values.

<a id="geoworkflow.core.constants.CommonCRS"></a>

## CommonCRS Objects

```python
class CommonCRS()
```

Common coordinate reference systems for African geospatial data.

<a id="geoworkflow.core.constants.CommonCRS.WGS84_UTM_33S"></a>

#### WGS84\_UTM\_33S

Common for Southern Africa

<a id="geoworkflow.core.constants.CommonCRS.WGS84_UTM_34S"></a>

#### WGS84\_UTM\_34S

Common for Southern/Eastern Africa

<a id="geoworkflow.core.constants.CommonCRS.AFRICA_ALBERS"></a>

#### AFRICA\_ALBERS

Africa Albers Equal Area Conic

<a id="geoworkflow.core.constants.CommonCRS.AFRICA_LAMBERT"></a>

#### AFRICA\_LAMBERT

Africa Lambert Conformal Conic

<a id="geoworkflow.core.constants.ProcessingStage"></a>

## ProcessingStage Objects

```python
class ProcessingStage(Enum)
```

Processing stages in the geoworkflow pipeline.

<a id="geoworkflow.core.constants.DataType"></a>

## DataType Objects

```python
class DataType(Enum)
```

Data type categories for processing decisions.

<a id="geoworkflow.core.constants.DataSource"></a>

## DataSource Objects

```python
class DataSource(Enum)
```

Known data sources with specialized processing requirements.

<a id="geoworkflow.core.constants.DEFAULT_TIMEOUT_SECONDS"></a>

#### DEFAULT\_TIMEOUT\_SECONDS

1 hour

<a id="geoworkflow.core.constants.MAX_MEMORY_RASTER_SIZE"></a>

#### MAX\_MEMORY\_RASTER\_SIZE

1 GB

<a id="geoworkflow.core.constants.MAX_VISUALIZATION_SIZE"></a>

#### MAX\_VISUALIZATION\_SIZE

100 MB

<a id="geoworkflow.core.constants.DOWNSAMPLE_THRESHOLD_PIXELS"></a>

#### DOWNSAMPLE\_THRESHOLD\_PIXELS

100 million pixels

<a id="geoworkflow.schemas.config_models"></a>

# geoworkflow.schemas.config\_models

Pydantic models for configuration validation.

This module defines Pydantic models that provide type-safe configuration
validation for all geoworkflow processors and components.

<a id="geoworkflow.schemas.config_models.LogLevel"></a>

## LogLevel Objects

```python
class LogLevel(str, Enum)
```

Valid logging levels.

<a id="geoworkflow.schemas.config_models.ResamplingMethod"></a>

## ResamplingMethod Objects

```python
class ResamplingMethod(str, Enum)
```

Valid resampling methods.

<a id="geoworkflow.schemas.config_models.ClassificationMethod"></a>

## ClassificationMethod Objects

```python
class ClassificationMethod(str, Enum)
```

Valid classification methods for visualization.

<a id="geoworkflow.schemas.config_models.AOIConfig"></a>

## AOIConfig Objects

```python
class AOIConfig(BaseModel, BaseConfig)
```

Configuration for AOI (Area of Interest) operations.

<a id="geoworkflow.schemas.config_models.ExtractionConfig"></a>

## ExtractionConfig Objects

```python
class ExtractionConfig(BaseModel, BaseConfig)
```

Configuration for data extraction operations.

<a id="geoworkflow.schemas.config_models.ClippingConfig"></a>

## ClippingConfig Objects

```python
class ClippingConfig(BaseModel, BaseConfig)
```

Configuration for spatial clipping operations.

<a id="geoworkflow.schemas.config_models.AlignmentConfig"></a>

## AlignmentConfig Objects

```python
class AlignmentConfig(BaseModel, BaseConfig)
```

Configuration for raster alignment operations.

<a id="geoworkflow.schemas.config_models.VisualizationConfig"></a>

## VisualizationConfig Objects

```python
class VisualizationConfig(BaseModel, BaseConfig)
```

Configuration for visualization operations.

<a id="geoworkflow.schemas.config_models.StatisticalEnrichmentConfig"></a>

## StatisticalEnrichmentConfig Objects

```python
class StatisticalEnrichmentConfig(BaseModel, BaseConfig)
```

Configuration for statistical enrichment operations.

<a id="geoworkflow.schemas.config_models.WorkflowConfig"></a>

## WorkflowConfig Objects

```python
class WorkflowConfig(BaseModel, BaseConfig)
```

Configuration for complete workflow pipelines.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.validate_stage_dependencies"></a>

#### validate\_stage\_dependencies

```python
@model_validator(mode='after')
def validate_stage_dependencies()
```

Validate that required stage configurations are provided.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.from_yaml"></a>

#### from\_yaml

```python
@classmethod
def from_yaml(cls, yaml_path: Union[str, Path]) -> 'WorkflowConfig'
```

Load workflow configuration from YAML file.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.to_yaml"></a>

#### to\_yaml

```python
def to_yaml(output_path: Union[str, Path]) -> None
```

Save workflow configuration to YAML file.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.validate"></a>

#### validate

```python
def validate() -> bool
```

Validate the complete workflow configuration.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.get_stage_order"></a>

#### get\_stage\_order

```python
def get_stage_order() -> List[str]
```

Get stages in their logical execution order.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.get_stage_config"></a>

#### get\_stage\_config

```python
def get_stage_config(stage_name: str) -> Optional[Dict[str, Any]]
```

Get configuration for a specific stage.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.has_stage"></a>

#### has\_stage

```python
def has_stage(stage_name: str) -> bool
```

Check if a stage is included in the workflow.

<a id="geoworkflow.schemas.config_models.WorkflowConfig.get_enrichment_dependencies"></a>

#### get\_enrichment\_dependencies

```python
def get_enrichment_dependencies() -> List[str]
```

Get the data dependencies for the enrichment stage.

<a id="geoworkflow.schemas.config_models.AlignmentConfig"></a>

## AlignmentConfig Objects

```python
class AlignmentConfig(BaseModel, BaseConfig)
```

Configuration for raster alignment operations.

<a id="geoworkflow.processors.aoi.processor"></a>

# geoworkflow.processors.aoi.processor

AOI (Area of Interest) processor for creating and managing areas of interest.

This processor transforms the legacy define_aoi.py script into a modern,
enhanced processor class using the Phase 2.1 infrastructure.

<a id="geoworkflow.processors.aoi.processor.AOIProcessor"></a>

## AOIProcessor Objects

```python
class AOIProcessor(TemplateMethodProcessor, GeospatialProcessorMixin)
```

Enhanced AOI processor for creating Areas of Interest from administrative boundaries.

This processor can:
- Extract specific countries with optional buffering
- Extract all countries and dissolve boundaries into single polygon
- Apply buffers in kilometers using proper projection
- List available countries for exploration

<a id="geoworkflow.processors.aoi.processor.AOIProcessor.process_data"></a>

#### process\_data

```python
def process_data() -> ProcessingResult
```

Execute the main AOI processing logic.

**Returns**:

  ProcessingResult with processing outcomes

<a id="geoworkflow.processors.aoi.processor.AOIProcessor.list_available_countries"></a>

#### list\_available\_countries

```python
def list_available_countries(prefix: Optional[str] = None) -> List[str]
```

List available countries in the boundaries file.

**Arguments**:

- `prefix` - Optional prefix to filter countries
  

**Returns**:

  List of available country names

<a id="geoworkflow.processors.aoi.processor.create_aoi_processor"></a>

#### create\_aoi\_processor

```python
def create_aoi_processor(config_path: Path) -> AOIProcessor
```

Factory function to create AOI processor from configuration file.

**Arguments**:

- `config_path` - Path to AOI configuration file
  

**Returns**:

  Configured AOI processor instance

<a id="geoworkflow.utils.progress_utils"></a>

# geoworkflow.utils.progress\_utils

Progress tracking utilities for the geoworkflow package.

This module provides utilities for tracking and displaying progress
during long-running operations.

<a id="geoworkflow.utils.progress_utils.ProgressTracker"></a>

## ProgressTracker Objects

```python
class ProgressTracker()
```

Progress tracker with rich console output when available,
fallback to simple logging when rich is not available.

<a id="geoworkflow.utils.progress_utils.ProgressTracker.update"></a>

#### update

```python
def update(increment: int = 1, description: Optional[str] = None)
```

Update progress.

**Arguments**:

- `increment` - Number of items completed
- `description` - Optional new description

<a id="geoworkflow.utils.progress_utils.ProgressTracker.set_description"></a>

#### set\_description

```python
def set_description(description: str)
```

Update the progress description.

<a id="geoworkflow.utils.progress_utils.ProgressTracker.close"></a>

#### close

```python
def close()
```

Close the progress tracker.

<a id="geoworkflow.utils.progress_utils.ProgressTracker.elapsed_time"></a>

#### elapsed\_time

```python
@property
def elapsed_time() -> float
```

Get elapsed time in seconds.

<a id="geoworkflow.utils.progress_utils.ProgressTracker.completion_percentage"></a>

#### completion\_percentage

```python
@property
def completion_percentage() -> float
```

Get completion percentage.

<a id="geoworkflow.utils.progress_utils.SimpleProgressTracker"></a>

## SimpleProgressTracker Objects

```python
class SimpleProgressTracker()
```

Simplified progress tracker for cases where rich is not needed.

<a id="geoworkflow.utils.progress_utils.SimpleProgressTracker.update"></a>

#### update

```python
def update(increment: int = 1)
```

Update progress.

<a id="geoworkflow.utils.progress_utils.SimpleProgressTracker.close"></a>

#### close

```python
def close()
```

Close tracker.

<a id="geoworkflow.utils.progress_utils.track_progress"></a>

#### track\_progress

```python
def track_progress(iterable,
                   description: str = "Processing",
                   disable: bool = False,
                   quiet: bool = False)
```

Track progress over an iterable.

**Arguments**:

- `iterable` - Items to iterate over
- `description` - Description of the operation
- `disable` - Whether to disable progress tracking
- `quiet` - Whether to suppress output
  

**Yields**:

  Items from the iterable

<a id="geoworkflow.utils.progress_utils.BatchProgressTracker"></a>

## BatchProgressTracker Objects

```python
class BatchProgressTracker()
```

Progress tracker for batch operations with multiple stages.

<a id="geoworkflow.utils.progress_utils.BatchProgressTracker.start_stage"></a>

#### start\_stage

```python
def start_stage(stage: str, total_items: int)
```

Start a new stage.

<a id="geoworkflow.utils.progress_utils.BatchProgressTracker.update_stage"></a>

#### update\_stage

```python
def update_stage(stage: str, increment: int = 1)
```

Update progress for current stage.

<a id="geoworkflow.utils.progress_utils.BatchProgressTracker.complete_stage"></a>

#### complete\_stage

```python
def complete_stage(stage: str)
```

Mark a stage as complete.

<a id="geoworkflow.utils.progress_utils.BatchProgressTracker.close"></a>

#### close

```python
def close()
```

Close the batch progress tracker.

<a id="geoworkflow.utils.progress_utils.create_progress_callback"></a>

#### create\_progress\_callback

```python
def create_progress_callback(tracker: ProgressTracker) -> Callable
```

Create a callback function for use with other libraries.

**Arguments**:

- `tracker` - Progress tracker instance
  

**Returns**:

  Callback function that updates the tracker

<a id="geoworkflow.utils.progress_utils.PerformanceMonitor"></a>

## PerformanceMonitor Objects

```python
class PerformanceMonitor()
```

Monitor performance metrics during processing.

<a id="geoworkflow.utils.progress_utils.PerformanceMonitor.start"></a>

#### start

```python
def start()
```

Start monitoring.

<a id="geoworkflow.utils.progress_utils.PerformanceMonitor.checkpoint"></a>

#### checkpoint

```python
def checkpoint(name: str)
```

Add a performance checkpoint.

<a id="geoworkflow.utils.progress_utils.PerformanceMonitor.get_summary"></a>

#### get\_summary

```python
def get_summary() -> dict
```

Get performance summary.

<a id="geoworkflow.utils.resource_utils"></a>

# geoworkflow.utils.resource\_utils

Resource management utilities for the geoworkflow package.

This module provides utilities for managing temporary directories,
file resources, and cleanup operations.

<a id="geoworkflow.utils.resource_utils.ResourceManager"></a>

## ResourceManager Objects

```python
class ResourceManager()
```

Context manager for handling temporary resources and cleanup.

This class provides a clean interface for managing temporary directories,
files, and other resources that need cleanup after processing.

<a id="geoworkflow.utils.resource_utils.ResourceManager.create_temp_directory"></a>

#### create\_temp\_directory

```python
def create_temp_directory(suffix: str = "") -> Path
```

Create a temporary directory.

**Arguments**:

- `suffix` - Optional suffix for directory name
  

**Returns**:

  Path to the created temporary directory

<a id="geoworkflow.utils.resource_utils.ResourceManager.create_temp_file"></a>

#### create\_temp\_file

```python
def create_temp_file(suffix: str = "", dir: Optional[Path] = None) -> Path
```

Create a temporary file.

**Arguments**:

- `suffix` - File suffix (e.g., '.tif')
- `dir` - Directory to create file in (None for system temp)
  

**Returns**:

  Path to the created temporary file

<a id="geoworkflow.utils.resource_utils.ResourceManager.cleanup"></a>

#### cleanup

```python
def cleanup()
```

Clean up all managed resources.

<a id="geoworkflow.utils.resource_utils.temp_directory"></a>

#### temp\_directory

```python
@contextmanager
def temp_directory(prefix: str = "geoworkflow_", suffix: str = "")
```

Context manager for creating a temporary directory.

**Arguments**:

- `prefix` - Prefix for directory name
- `suffix` - Suffix for directory name
  

**Yields**:

  Path to temporary directory

<a id="geoworkflow.utils.resource_utils.temp_file"></a>

#### temp\_file

```python
@contextmanager
def temp_file(suffix: str = "",
              dir: Optional[Path] = None,
              prefix: str = "geoworkflow_")
```

Context manager for creating a temporary file.

**Arguments**:

- `suffix` - File suffix
- `dir` - Directory to create file in
- `prefix` - Prefix for file name
  

**Yields**:

  Path to temporary file

<a id="geoworkflow.utils.resource_utils.ensure_directory"></a>

#### ensure\_directory

```python
def ensure_directory(path: Union[str, Path], create: bool = True) -> Path
```

Ensure a directory exists.

**Arguments**:

- `path` - Directory path
- `create` - Whether to create if it doesn't exist
  

**Returns**:

  Path object
  

**Raises**:

- `GeoWorkflowError` - If directory doesn't exist and create=False

<a id="geoworkflow.utils.resource_utils.safe_remove_file"></a>

#### safe\_remove\_file

```python
def safe_remove_file(file_path: Union[str, Path]) -> bool
```

Safely remove a file.

**Arguments**:

- `file_path` - Path to file to remove
  

**Returns**:

  True if removed successfully, False otherwise

<a id="geoworkflow.utils.resource_utils.safe_remove_directory"></a>

#### safe\_remove\_directory

```python
def safe_remove_directory(dir_path: Union[str, Path]) -> bool
```

Safely remove a directory and its contents.

**Arguments**:

- `dir_path` - Path to directory to remove
  

**Returns**:

  True if removed successfully, False otherwise

<a id="geoworkflow.utils.resource_utils.get_safe_filename"></a>

#### get\_safe\_filename

```python
def get_safe_filename(name: str, max_length: int = 255) -> str
```

Convert a string to a safe filename.

**Arguments**:

- `name` - Original name
- `max_length` - Maximum filename length
  

**Returns**:

  Safe filename string

<a id="geoworkflow.utils.resource_utils.get_disk_usage"></a>

#### get\_disk\_usage

```python
def get_disk_usage(path: Union[str, Path]) -> Dict[str, int]
```

Get disk usage statistics for a path.

**Arguments**:

- `path` - Path to check
  

**Returns**:

  Dictionary with 'total', 'used', 'free' in bytes

<a id="geoworkflow.utils.resource_utils.check_disk_space"></a>

#### check\_disk\_space

```python
def check_disk_space(path: Union[str, Path], required_bytes: int) -> bool
```

Check if sufficient disk space is available.

**Arguments**:

- `path` - Path to check
- `required_bytes` - Required space in bytes
  

**Returns**:

  True if sufficient space available

<a id="geoworkflow.utils.resource_utils.format_bytes"></a>

#### format\_bytes

```python
def format_bytes(bytes_value: int) -> str
```

Format bytes as human-readable string.

**Arguments**:

- `bytes_value` - Number of bytes
  

**Returns**:

  Formatted string (e.g., "1.5 GB")

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics"></a>

## ProcessingMetrics Objects

```python
class ProcessingMetrics()
```

Simple metrics collection for processing operations.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.start"></a>

#### start

```python
def start()
```

Mark the start of processing.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.finish"></a>

#### finish

```python
def finish()
```

Mark the end of processing.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.elapsed_time"></a>

#### elapsed\_time

```python
@property
def elapsed_time() -> float
```

Get elapsed time in seconds.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.total_files"></a>

#### total\_files

```python
@property
def total_files() -> int
```

Get total number of files.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.add_metric"></a>

#### add\_metric

```python
def add_metric(name: str, value: Any)
```

Add a custom metric.

<a id="geoworkflow.utils.resource_utils.ProcessingMetrics.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert metrics to dictionary.

