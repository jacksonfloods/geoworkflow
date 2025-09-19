# File: src/geoworkflow/visualization/raster/processor.py
"""
Enhanced raster visualization processor for the geoworkflow package.

This processor transforms the legacy raster_visualizer.py into an enhanced
processor class using the Phase 2.1 infrastructure, with modular color schemes.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging
import yaml
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import rasterio
    import geopandas as gpd
    import contextily as cx
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import jenkspy
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin, EnhancedProcessingResult
from geoworkflow.core.exceptions import ProcessingError, ValidationError, VisualizationError
from geoworkflow.schemas.config_models import VisualizationConfig
from geoworkflow.core.base import ProcessingResult
from geoworkflow.utils.progress_utils import track_progress
from geoworkflow.utils.resource_utils import ensure_directory


class RasterVisualizationProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Enhanced raster visualization processor with modular color schemes.
    
    This processor can:
    - Visualize rasters from any stage (02_clipped, 03_processed, etc.)
    - Use modular color schemes loaded from YAML files
    - Apply intelligent classification methods based on data type
    - Create publication-quality outputs with basemaps
    - Integrate with the clipping processor for automatic visualization
    """
    
    def __init__(self, config: Union[VisualizationConfig, Dict[str, Any]], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize visualization processor.
        
        Args:
            config: Visualization configuration object or dictionary
            logger: Optional logger instance
        """
        # Convert Pydantic model to dict for base class
        if isinstance(config, VisualizationConfig):
            config_dict = config.model_dump(mode='json')
            self.viz_config = config
        else:
            config_dict = config
            self.viz_config = VisualizationConfig.from_dict(config_dict)
        
        super().__init__(config_dict, logger)
        
        # Processing state
        self.color_schemes: Dict[str, Dict] = {}
        self.raster_files: List[Path] = []
        self.visualization_count = 0
    
    def _validate_custom_inputs(self) -> Dict[str, Any]:
        """
        Validate visualization-specific inputs and configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check geospatial libraries
        if not HAS_GEOSPATIAL_LIBS:
            validation_result["errors"].append(
                "Required libraries not available. Please install: "
                "matplotlib, rasterio, geopandas, contextily, jenkspy"
            )
            validation_result["valid"] = False
            return validation_result
        
        # Validate input path exists
        if not self.viz_config.input_path and not self.viz_config.input_directory:
            validation_result["errors"].append(
                "Either input_path or input_directory must be specified"
            )
            validation_result["valid"] = False
        
        if self.viz_config.input_path and not self.viz_config.input_path.exists():
            validation_result["errors"].append(
                f"Input path does not exist: {self.viz_config.input_path}"
            )
            validation_result["valid"] = False
        
        if self.viz_config.input_directory and not self.viz_config.input_directory.exists():
            validation_result["errors"].append(
                f"Input directory does not exist: {self.viz_config.input_directory}"
            )
            validation_result["valid"] = False
        
        # Validate output directory can be created
        try:
            ensure_directory(self.viz_config.output_dir)
            validation_result["info"]["output_dir_validated"] = str(self.viz_config.output_dir)
        except Exception as e:
            validation_result["errors"].append(
                f"Cannot create output directory: {e}"
            )
            validation_result["valid"] = False
        
        # Validate colormap exists
        try:
            plt.get_cmap(self.viz_config.colormap)
            validation_result["info"]["colormap_validated"] = self.viz_config.colormap
        except ValueError:
            validation_result["warnings"].append(
                f"Unknown colormap '{self.viz_config.colormap}', will fall back to 'viridis'"
            )
        
        return validation_result
    
    def _get_path_config_keys(self) -> List[str]:
        """Define which config keys contain paths that must exist."""
        return ["input_path", "input_directory"]  # Don't validate output_dir existence
    
    def _estimate_total_items(self) -> int:
        """Estimate total items for progress tracking."""
        if self.viz_config.input_path and self.viz_config.input_path.is_file():
            return 1
        elif self.viz_config.input_directory:
            try:
                raster_files = self._find_raster_files()
                return len(raster_files)
            except:
                return 0
        return 0
    
    def _setup_custom_processing(self) -> Dict[str, Any]:
        """Setup visualization-specific processing resources."""
        setup_info = {}
        
        # Add geospatial setup
        geo_setup = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_setup
        
        # Load color schemes
        self.log_processing_step("Loading color schemes")
        try:
            self.color_schemes = self._load_color_schemes()
            setup_info["color_schemes_loaded"] = len(self.color_schemes)
        except Exception as e:
            self.logger.warning(f"Could not load color schemes: {e}")
            setup_info["color_schemes_loaded"] = 0
        
        # Find raster files to process
        self.log_processing_step("Finding raster files")
        self.raster_files = self._find_raster_files()
        setup_info["raster_files_found"] = len(self.raster_files)
        
        # Set up matplotlib
        plt.ioff()  # Turn off interactive mode
        setup_info["matplotlib_configured"] = True
        
        return setup_info
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the main visualization logic.
        
        Returns:
            ProcessingResult with visualization outcomes
        """
        result = ProcessingResult(success=True)
        
        try:
            if not self.raster_files:
                result.message = "No raster files found to visualize"
                result.skipped_count = 1
                return result
            
            self.log_processing_step(f"Visualizing {len(self.raster_files)} raster files")
            
            # Process each raster file
            for raster_file in track_progress(
                self.raster_files,
                description="Creating visualizations",
                quiet=False
            ):
                success = self._visualize_single_raster(raster_file)
                
                if success:
                    result.processed_count += 1
                    self.visualization_count += 1
                else:
                    result.failed_count += 1
                    result.add_failed_file(raster_file)
                
                self.update_progress(1, f"Processed {raster_file.name}")
            
            # Update result
            result.message = f"Successfully created {result.processed_count} visualizations"
            result.add_output_path(self.viz_config.output_dir)
            
            # Add processing metadata
            result.metadata = {
                "output_directory": str(self.viz_config.output_dir),
                "visualizations_created": result.processed_count,
                "colormap_used": self.viz_config.colormap,
                "classification_method": self.viz_config.classification_method.value,
                "dpi": self.viz_config.dpi
            }
            
            self.add_metric("visualizations_created", result.processed_count)
            
        except Exception as e:
            result.success = False
            result.message = f"Visualization processing failed: {str(e)}"
            self.logger.error(f"Visualization processing failed: {e}")
            raise VisualizationError(f"Visualization processing failed: {str(e)}")
        
        return result
    
    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        """Cleanup visualization-specific resources."""
        cleanup_info = {}
        
        # Add geospatial cleanup
        geo_cleanup = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_cleanup
        
        # Close any matplotlib figures
        plt.close('all')
        cleanup_info["matplotlib_figures_closed"] = True
        
        # Clear data from memory
        if hasattr(self, 'raster_files'):
            del self.raster_files
            cleanup_info["raster_files_cleared"] = True
        
        cleanup_info["visualization_cleanup"] = "completed"
        
        return cleanup_info
    
    def _load_color_schemes(self) -> Dict[str, Dict]:
        """
        Load color schemes from YAML configuration files.
        
        Returns:
            Dictionary mapping dataset names to color schemes
        """
        color_schemes = {}
        
        # Look for color scheme files in config directory
        config_dir = Path("config/visualization/color_schemes")
        if not config_dir.exists():
            # Try alternative locations
            config_dir = Path("config/color_schemes")
            if not config_dir.exists():
                self.logger.info("No color schemes directory found, using defaults")
                return self._get_default_color_schemes()
        
        # Load all YAML files in the directory
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    scheme_data = yaml.safe_load(f)
                
                dataset_name = yaml_file.stem
                color_schemes[dataset_name] = scheme_data
                self.logger.debug(f"Loaded color scheme for {dataset_name}")
                
            except Exception as e:
                self.logger.warning(f"Could not load color scheme {yaml_file}: {e}")
        
        # Add defaults if none loaded
        if not color_schemes:
            color_schemes = self._get_default_color_schemes()
        
        return color_schemes
    
    def _get_default_color_schemes(self) -> Dict[str, Dict]:
        """Get default color schemes for known datasets."""
        return {
            "copernicus": {
                "type": "categorical",
                "name": "Copernicus Land Cover",
                "colors": {
                    0: "#000000",    # No Data
                    10: "#ffff64",   # Cropland
                    20: "#aaf0f0",   # Forest
                    30: "#dcf064",   # Grassland
                    40: "#c8c8c8",   # Shrubland
                    50: "#006400",   # Wetlands
                    60: "#ffb432",   # Settlement
                    70: "#ffc85a",   # Bare/sparse vegetation
                    80: "#0064c8",   # Water bodies
                    90: "#ffffff"    # Snow/ice
                },
                "labels": {
                    0: "No Data",
                    10: "Cropland",
                    20: "Forest",
                    30: "Grassland",
                    40: "Shrubland",
                    50: "Wetlands",
                    60: "Settlement",
                    70: "Bare/sparse vegetation",
                    80: "Water bodies",
                    90: "Snow/ice"
                }
            },
            "odiac": {
                "type": "continuous",
                "name": "ODIAC Emissions",
                "colormap": "plasma",
                "classification": "log"
            },
            "pm25": {
                "type": "continuous",
                "name": "PM2.5 Concentration",
                "colormap": "plasma",
                "classification": "log",
                "units": "μg/m³"
            }
        }
    
    def _find_raster_files(self) -> List[Path]:
        """Find all raster files to process."""
        raster_files = []
        
        if self.viz_config.input_path and self.viz_config.input_path.is_file():
            # Single file
            raster_files.append(self.viz_config.input_path)
        elif self.viz_config.input_directory:
            # Directory processing
            input_dir = self.viz_config.input_directory
            
            # Find raster files
            extensions = ['.tif', '.tiff', '.geotif', '.geotiff']
            for ext in extensions:
                pattern = f"**/*{ext}" if True else f"*{ext}"  # Always recursive for now
                raster_files.extend(input_dir.glob(pattern))
        
        # Apply pattern filter if specified
        if self.viz_config.pattern:
            import fnmatch
            raster_files = [
                f for f in raster_files 
                if fnmatch.fnmatch(f.name, self.viz_config.pattern)
            ]
        
        return sorted(raster_files)
    
    def _visualize_single_raster(self, raster_path: Path) -> bool:
        """
        Visualize a single raster file.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine output path
            output_filename = f"{raster_path.stem}.png"
            output_path = self.viz_config.output_dir / output_filename
            
            # Skip if exists and not overwriting
            if output_path.exists() and not self.viz_config.overwrite:
                self.logger.debug(f"Skipping existing visualization: {output_path}")
                return True
            
            # Detect dataset type for color scheme
            dataset_type = self._detect_dataset_type(raster_path)
            
            # Load and process raster data
            with rasterio.open(raster_path) as src:
                # Read first band
                data = src.read(1)
                
                # Handle NoData
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)
                
                # Apply downsampling if needed
                if self.viz_config.downsample:
                    data = self._downsample_raster(data)
                
                # Create visualization
                success = self._create_visualization(
                    data=data,
                    src=src,
                    output_path=output_path,
                    dataset_type=dataset_type,
                    title=self._generate_title(raster_path, dataset_type)
                )
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error visualizing {raster_path}: {e}")
            return False
    
    def _detect_dataset_type(self, raster_path: Path) -> str:
        """
        Detect dataset type from file path.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            Dataset type string
        """
        path_str = str(raster_path).lower()
        
        if any(keyword in path_str for keyword in ['copernicus', 'clms', 'land_cover', 'lc']):
            return 'copernicus'
        elif any(keyword in path_str for keyword in ['odiac', 'emission', 'co2']):
            return 'odiac'
        elif any(keyword in path_str for keyword in ['pm25', 'pm2.5']):
            return 'pm25'
        elif any(keyword in path_str for keyword in ['lst', 'temperature']):
            return 'temperature'
        else:
            return 'generic'
    
    def _downsample_raster(self, data: np.ndarray) -> np.ndarray:
        """Downsample raster for visualization if it's too large."""
        if data.size <= self.viz_config.max_dimension * self.viz_config.max_dimension:
            return data
        
        # Simple downsampling by taking every nth pixel
        factor = int(np.sqrt(data.size / (self.viz_config.max_dimension * self.viz_config.max_dimension)))
        return data[::factor, ::factor]
    
    def _create_visualization(self, data: np.ndarray, src, output_path: Path, 
                            dataset_type: str, title: str) -> bool:
        """
        Create the actual visualization.
        
        Args:
            data: Raster data array
            src: Rasterio dataset source
            output_path: Output file path
            dataset_type: Type of dataset
            title: Plot title
            
        Returns:
            True if successful
        """
        try:
            # Set up figure
            fig, ax = plt.subplots(1, 1, figsize=(self.viz_config.figure_width, self.viz_config.figure_height))
            
            # Get color scheme for this dataset
            color_scheme = self.color_schemes.get(dataset_type, {})
            
            if color_scheme.get('type') == 'categorical':
                # Categorical data with custom colors
                success = self._plot_categorical_data(data, ax, color_scheme)
            else:
                # Continuous data
                success = self._plot_continuous_data(data, ax, color_scheme)
            
            if not success:
                plt.close(fig)
                return False
            
            # Add title
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add basemap if requested
            if self.viz_config.add_basemap:
                self._add_basemap(ax, src)
            
            # Save figure
            plt.savefig(
                output_path,
                dpi=self.viz_config.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
    
    def _plot_categorical_data(self, data: np.ndarray, ax, color_scheme: Dict) -> bool:
        """Plot categorical data with custom colors."""
        try:
            colors_dict = color_scheme.get('colors', {})
            labels_dict = color_scheme.get('labels', {})
            
            # Get unique values
            unique_vals = np.unique(data.compressed() if np.ma.is_masked(data) else data)
            
            # Create colormap and bounds
            colors = []
            bounds = []
            labels = []
            
            for val in sorted(unique_vals):
                if val in colors_dict:
                    colors.append(colors_dict[val])
                    bounds.append(val)
                    labels.append(labels_dict.get(val, f"Class {val}"))
            
            if not colors:
                return False
            
            # Create custom colormap
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bounds + [bounds[-1] + 1], len(colors))
            
            # Plot
            im = ax.imshow(data, cmap=cmap, norm=norm, alpha=self.viz_config.raster_alpha)
            
            # Add colorbar with labels
            if self.viz_config.show_colorbar:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_ticks(bounds)
                cbar.set_ticklabels(labels)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting categorical data: {e}")
            return False
    
    def _plot_continuous_data(self, data: np.ndarray, ax, color_scheme: Dict) -> bool:
        """Plot continuous data with classification."""
        try:
            # Get colormap
            colormap = color_scheme.get('colormap', self.viz_config.colormap)
            
            # Apply classification
            classified_data, method_info = self._apply_classification(data, color_scheme)
            
            # Plot
            im = ax.imshow(
                classified_data,
                cmap=colormap,
                alpha=self.viz_config.raster_alpha
            )
            
            # Add colorbar
            if self.viz_config.show_colorbar:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                if 'units' in color_scheme:
                    cbar.set_label(color_scheme['units'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting continuous data: {e}")
            return False

    def _apply_classification(self, data: np.ndarray, color_scheme: Dict) -> Tuple[np.ndarray, str]:
        """Apply classification to continuous data."""
        # Get classification method - handle both string and enum types
        method = color_scheme.get('classification')
        if method is None:
            # Get from config, handling both enum and string cases
            if hasattr(self.viz_config.classification_method, 'value'):
                method = self.viz_config.classification_method.value
            else:
                method = str(self.viz_config.classification_method)
        
        # Get valid data (remove masked/nodata values)
        valid_data = data.compressed() if np.ma.is_masked(data) else data.flatten()
        valid_data = valid_data[np.isfinite(valid_data)]
        
        if len(valid_data) == 0:
            return data, "no_data"
        
        try:
            if method == 'log':
                # Logarithmic transformation + jenks
                log_data = np.log10(valid_data[valid_data > 0])
                if len(log_data) > 0:
                    breaks = jenkspy.jenks_breaks(log_data, n_classes=self.viz_config.n_classes)
                    breaks = [10**b for b in breaks]
                else:
                    breaks = np.linspace(valid_data.min(), valid_data.max(), self.viz_config.n_classes + 1)
            elif method == 'jenks':
                breaks = jenkspy.jenks_breaks(valid_data, n_classes=self.viz_config.n_classes)
            elif method == 'quantile':
                breaks = np.quantile(valid_data, np.linspace(0, 1, self.viz_config.n_classes + 1))
            else:
                # Equal interval (default for 'auto' and unknown methods)
                breaks = np.linspace(valid_data.min(), valid_data.max(), self.viz_config.n_classes + 1)
            
            # Apply classification
            classified_data = np.digitize(data, breaks) - 1
            return classified_data, f"{method}_classification"
            
        except Exception as e:
            self.logger.warning(f"Classification failed, using original data: {e}")
            return data, "no_classification"

        def _add_basemap(self, ax, src):
            """Add basemap to the plot."""
            try:
                if src.crs and src.crs.to_string() != 'EPSG:4326':
                    # Would need to reproject for basemap
                    self.logger.debug("Skipping basemap - raster not in WGS84")
                    return
                
                cx.add_basemap(
                    ax,
                    crs=src.crs,
                    source=cx.providers.CartoDB.Positron,
                    alpha=self.viz_config.basemap_alpha
                )
            except Exception as e:
                self.logger.debug(f"Could not add basemap: {e}")
        
        def _generate_title(self, raster_path: Path, dataset_type: str) -> str:
            """Generate title for the visualization."""
            # Get dataset name from color scheme
            color_scheme = self.color_schemes.get(dataset_type, {})
            dataset_name = color_scheme.get('name', dataset_type.title())
            
            # Use filename as base
            base_name = raster_path.stem.replace('_', ' ').title()
            
            return f"{dataset_name}\n{base_name}"


# Convenience function for integration with clipping processor
def visualize_clipped_data(input_directory: Path, output_directory: Path,
                          config_overrides: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to visualize clipped data from the clipping processor.
    
    Args:
        input_directory: Directory containing clipped rasters
        output_directory: Directory to save visualizations
        config_overrides: Optional configuration overrides
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create visualization configuration
        viz_config = VisualizationConfig(
            input_directory=input_directory,
            output_dir=output_directory,
            colormap="viridis",
            classification_method="auto",
            n_classes=8,
            dpi=150,
            add_basemap=True,
            overwrite=True
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(viz_config, key):
                    setattr(viz_config, key, value)
        
        # Create and run processor
        processor = RasterVisualizationProcessor(viz_config)
        result = processor.process()
        
        return result.success
        
    except Exception as e:
        logging.getLogger('geoworkflow.visualization').error(f"Visualization failed: {e}")
        return False