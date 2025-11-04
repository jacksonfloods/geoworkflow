"""Configuration file loader for geoworkflow settings."""

from pathlib import Path
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load geoworkflow configuration settings from ~/.geoworkflow/config.yaml."""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".geoworkflow" / "config.yaml"
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load config with fallback to empty dict."""
        if ConfigLoader.DEFAULT_CONFIG_PATH.exists():
            try:
                with open(ConfigLoader.DEFAULT_CONFIG_PATH) as f:
                    config = yaml.safe_load(f)
                    return config if config else {}
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                return {}
        return {}
    
    @staticmethod
    def get_africapolis_path() -> Path:
        """Get agglomerations.gpkg path from config or use default."""
        config = ConfigLoader.load_config()
        
        if "africapolis" in config and "path" in config["africapolis"]:
            path_str = config["africapolis"]["path"]
            return Path(path_str).expanduser()
        
        # Fallback default
        default_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "00_source" / "boundaries" / "agglomerations.gpkg"
        return default_path
    
    @staticmethod
    def get_africapolis_columns() -> Dict[str, str]:
        """Get column names for AfricaPolis filtering."""
        config = ConfigLoader.load_config()
        if "africapolis" in config:
            return {
                "iso3": config["africapolis"].get("iso3_column", "ISO3"),
                "name": config["africapolis"].get("name_column", "Agglomeration_Name")
            }
        return {"iso3": "ISO3", "name": "Agglomeration_Name"}
