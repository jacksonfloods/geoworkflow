"""
Configuration management for the geoworkflow package.

This module provides utilities for loading, validating, and managing
configuration files across the geoworkflow system.
"""

import json
import yaml
from typing import Dict, Any, Union, Optional, Type, TypeVar
from pathlib import Path
import logging

from ..core.exceptions import ConfigurationError
from ..core.base import BaseConfig

# Type variable for configuration classes
ConfigT = TypeVar('ConfigT', bound=BaseConfig)

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration management."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Supports JSON and YAML formats.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Determine format by extension
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(content)
            elif config_path.suffix.lower() == '.json':
                config = json.loads(content)
            else:
                # Try to parse as YAML first, then JSON
                try:
                    config = yaml.safe_load(content)
                except yaml.YAMLError:
                    try:
                        config = json.loads(content)
                    except json.JSONDecodeError:
                        raise ConfigurationError(
                            f"Cannot parse configuration file: {config_path}. "
                            f"Supported formats: JSON, YAML"
                        )
            
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration must be a dictionary, got {type(config)}"
                )
                
            logger.debug(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error loading configuration from {config_path}: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: Union[str, Path], 
                   format: str = 'yaml') -> None:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration
            format: Output format ('yaml' or 'json')
            
        Raises:
            ConfigurationError: If saving fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ConfigurationError(f"Unsupported format: {format}")
                    
            logger.debug(f"Saved configuration to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration to {output_path}: {e}")
    
    @staticmethod
    def validate_required_keys(config: Dict[str, Any], required_keys: list) -> None:
        """
        Validate that required keys are present in configuration.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required key names
            
        Raises:
            ConfigurationError: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )
    
    @staticmethod
    def validate_paths(config: Dict[str, Any], path_keys: list, 
                      must_exist: bool = True) -> None:
        """
        Validate that path values exist.
        
        Args:
            config: Configuration dictionary
            path_keys: List of keys that should contain paths
            must_exist: Whether paths must exist
            
        Raises:
            ConfigurationError: If paths are invalid
        """
        for key in path_keys:
            if key in config:
                path = Path(config[key])
                if must_exist and not path.exists():
                    raise ConfigurationError(f"Path does not exist: {key} = {path}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge (takes precedence)
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    @classmethod
    def create_config_object(cls, config_class: Type[ConfigT], 
                           config_path: Union[str, Path]) -> ConfigT:
        """
        Create a configuration object from a file.
        
        Args:
            config_class: Configuration class to instantiate
            config_path: Path to configuration file
            
        Returns:
            Configuration object instance
        """
        config_dict = cls.load_config(config_path)
        return config_class.from_dict(config_dict)


def validate_file_extensions(path: Union[str, Path], 
                           valid_extensions: list) -> bool:
    """
    Validate that a file has one of the specified extensions.
    
    Args:
        path: File path to check
        valid_extensions: List of valid extensions (e.g., ['.tif', '.tiff'])
        
    Returns:
        True if extension is valid
    """
    path_obj = Path(path)
    return path_obj.suffix.lower() in [ext.lower() for ext in valid_extensions]


def resolve_path(path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
    """
    Resolve a path, making it absolute if it's relative.
    
    Args:
        path: Path to resolve
        base_path: Base path for relative resolution (default: current working directory)
        
    Returns:
        Resolved absolute path
    """
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        return path_obj
    
    if base_path is None:
        base_path = Path.cwd()
    
    return (base_path / path_obj).resolve()


def expand_path_pattern(pattern: str, base_dir: Union[str, Path]) -> list:
    """
    Expand a file pattern to list of matching files.
    
    Args:
        pattern: File pattern (e.g., "*.tif")
        base_dir: Base directory to search
        
    Returns:
        List of matching file paths
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return []
    
    return list(base_dir.glob(pattern))


def get_config_template(config_type: str) -> Dict[str, Any]:
    """
    Get a template configuration for a specific processor type.
    
    Args:
        config_type: Type of configuration ('aoi', 'extraction', 'clipping', etc.)
        
    Returns:
        Template configuration dictionary
    """
    templates = {
        'aoi': {
            'input_file': 'data/00_source/boundaries/africa_boundaries.geojson',
            'country_name_column': 'NAME_0',
            'countries': ['Angola', 'Namibia', 'Botswana'],
            'buffer_km': 100,
            'output_file': 'data/aoi/southern_africa_aoi.geojson'
        },
        'extraction': {
            'zip_folder': 'data/00_source/archives/',
            'aoi_file': 'data/aoi/southern_africa_aoi.geojson',
            'output_dir': 'data/01_extracted/',
            'raster_pattern': '*.tif',
            'vector_pattern': '*.{shp,geojson,gpkg,kml}',
            'skip_existing': True,
            'create_visualizations': True
        },
        'clipping': {
            'input_directory': 'data/01_extracted/',
            'aoi_file': 'data/aoi/southern_africa_aoi.geojson',
            'output_dir': 'data/02_clipped/',
            'raster_pattern': '*.tif',
            'all_touched': True,
            'create_visualizations': True
        },
        'alignment': {
            'reference_raster': 'data/02_clipped/copernicus/reference.tif',
            'input_directory': 'data/02_clipped/',
            'output_dir': 'data/03_processed/aligned/',
            'recursive': True,
            'preserve_directory_structure': True,
            'skip_existing': True,
            'resampling_method': 'cubic'
        },
        'visualization': {
            'input_directory': 'data/03_processed/',
            'output_dir': 'outputs/visualizations/',
            'colormap': 'viridis',
            'classification_method': 'auto',
            'n_classes': 8,
            'dpi': 150,
            'add_basemap': True,
            'overwrite': False
        },
        'pipeline': {
            'name': 'Southern Africa Workflow',
            'stages': ['extract', 'clip', 'align', 'visualize'],
            'aoi_config': {
                'countries': ['Angola', 'Namibia', 'Botswana'],
                'buffer_km': 100
            },
            'data_sources': ['copernicus', 'odiac', 'pm25'],
            'create_visualizations': True,
            'stop_on_error': True
        }
    }
    
    return templates.get(config_type, {})