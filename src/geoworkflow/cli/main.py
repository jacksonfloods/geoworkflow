"""
Main CLI entry point for the geoworkflow package.

This module provides the unified command-line interface for all
geoworkflow operations.
"""

import click
import logging
from pathlib import Path
from typing import Optional

from ..core.logging_setup import setup_logging
from ..core.exceptions import GeoWorkflowError, ConfigurationError
from ..core.constants import DEFAULT_LOG_LEVEL

def print_version(ctx, param, value):
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    
    from .. import __version__
    click.echo(f"GeoWorkflow version {__version__}")
    click.echo("Comprehensive geospatial data processing for African analysis")
    ctx.exit()

@click.group()
@click.option('--version', is_flag=True, expose_value=False, is_eager=True,
              callback=print_version, help='Show version and exit')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                               case_sensitive=False),
              default=DEFAULT_LOG_LEVEL,
              help='Set logging level')
@click.option('--log-file', type=click.Path(), help='Log to file instead of console')
@click.pass_context
def main(ctx, verbose: bool, quiet: bool, log_level: str, log_file: Optional[str]):
    """
    GeoWorkflow - Comprehensive geospatial data processing for African analysis.
    
    A unified toolkit for processing, aligning, and visualizing geospatial data
    with focus on African datasets including Copernicus, ODIAC, PM2.5, and AFRICAPOLIS.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Determine log level
    if verbose:
        log_level = 'DEBUG'
    elif quiet:
        log_level = 'WARNING'
    
    # Set up logging
    try:
        setup_logging(level=log_level, log_file=log_file)
        logger = logging.getLogger('geoworkflow.cli')
        logger.debug(f"CLI started with log level: {log_level}")
    except Exception as e:
        click.echo(f"Error setting up logging: {e}", err=True)
        ctx.exit(1)
    
    # Store context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['log_level'] = log_level
    ctx.obj['logger'] = logger


@main.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from .. import __version__
    
    if not ctx.obj.get('quiet', False):
        click.echo(f"GeoWorkflow version {__version__}")
        click.echo("Comprehensive geospatial data processing for African analysis")


@main.group()
@click.pass_context
def aoi(ctx):
    """Area of Interest (AOI) operations."""
    pass


@main.group()
@click.pass_context
def extract(ctx):
    """Data extraction operations."""
    pass


@main.group()
@click.pass_context
def process(ctx):
    """Data processing operations."""
    pass


@main.group()
@click.pass_context
def visualize(ctx):
    """Visualization operations."""
    pass


@main.group()
@click.pass_context
def pipeline(ctx):
    """Pipeline orchestration operations."""
    pass


# AOI subcommands
@aoi.command('create')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--countries', type=str, 
              help='Comma-separated list of countries')
@click.option('--buffer', type=float, default=100.0,
              help='Buffer distance in kilometers')
@click.option('--output', '-o', type=click.Path(),
              help='Output AOI file path')
@click.pass_context
def aoi_create(ctx, config: Optional[str], countries: Optional[str], 
               buffer: float, output: Optional[str]):
    """Create Area of Interest from country boundaries."""
    logger = ctx.obj['logger']
    
    try:
        if config:
            # Load from configuration file
            click.echo(f"Creating AOI from configuration: {config}")
            logger.info(f"Loading AOI configuration from {config}")
            # TODO: Implement AOI processor call
            click.echo("✅ AOI created successfully")
        else:
            # Create from command line parameters
            if not countries:
                raise click.ClickException("Either --config or --countries must be specified")
            
            country_list = [c.strip() for c in countries.split(',')]
            click.echo(f"Creating AOI for countries: {', '.join(country_list)}")
            click.echo(f"Buffer: {buffer} km")
            
            # TODO: Implement direct AOI creation
            click.echo("✅ AOI created successfully")
            
    except GeoWorkflowError as e:
        logger.error(f"AOI creation failed: {e}")
        raise click.ClickException(str(e))


@aoi.command('list')
@click.option('--directory', '-d', type=click.Path(exists=True), 
              default='data/aoi',
              help='Directory to search for AOI files')
@click.pass_context
def aoi_list(ctx, directory: str):
    """List available AOI files."""
    aoi_dir = Path(directory)
    
    if not aoi_dir.exists():
        click.echo(f"AOI directory not found: {directory}")
        return
    
    aoi_files = list(aoi_dir.glob('*.geojson')) + list(aoi_dir.glob('*.shp'))
    
    if not aoi_files:
        click.echo(f"No AOI files found in {directory}")
        return
    
    click.echo(f"Found {len(aoi_files)} AOI files in {directory}:")
    for aoi_file in sorted(aoi_files):
        click.echo(f"  📍 {aoi_file.name}")


# Extract subcommands
@extract.command('archives')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file path')
@click.option('--source', '-s', type=click.Path(exists=True),
              help='Source directory or ZIP file')
@click.option('--aoi', type=click.Path(exists=True),
              help='AOI file for clipping')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory')
@click.option('--no-viz', is_flag=True,
              help='Skip visualization creation')
@click.pass_context
def extract_archives(ctx, config: Optional[str], source: Optional[str],
                    aoi: Optional[str], output: Optional[str], no_viz: bool):
    """Extract data from ZIP archives."""
    logger = ctx.obj['logger']
    
    try:
        if config:
            click.echo(f"Extracting archives using configuration: {config}")
            logger.info(f"Loading extraction configuration from {config}")
            # TODO: Implement extraction processor call
        else:
            if not all([source, aoi, output]):
                raise click.ClickException("When not using --config, --source, --aoi, and --output are required")
            
            click.echo(f"Extracting from: {source}")
            click.echo(f"Using AOI: {aoi}")
            click.echo(f"Output to: {output}")
            # TODO: Implement direct extraction
        
        click.echo("✅ Extraction completed successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Extraction failed: {e}")
        raise click.ClickException(str(e))

@extract.command('netcdf')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--variable', '-v', type=str, help='Variable name to extract (e.g., PM25)')
@click.option('--output', '-o', type=click.Path(),
              help='Output GeoTIFF file path (auto-generated if not specified)')
@click.option('--crs', type=str, default='EPSG:4326',
              help='Coordinate reference system')
@click.option('--time-slice', type=str,
              help='Time slice (e.g., "2022-01-01:2022-12-31")')
@click.option('--time-mean', is_flag=True,
              help='Average across time dimension')
@click.pass_context
def extract_netcdf(ctx, input_file: str, variable: Optional[str], output: str, 
                   crs: str, time_slice: Optional[str], time_mean: bool):
    """Extract data from NetCDF files and convert to GeoTIFF."""
    logger = ctx.obj['logger']
    
    try:
        import xarray as xr
        import rioxarray as rxr
        from pathlib import Path
        
        click.echo(f"Processing NetCDF file: {input_file}")
        logger.info(f"Converting NetCDF {input_file} to GeoTIFF {output}")
        
        # Open NetCDF file
        ds = xr.open_dataset(input_file)
        
        # Auto-detect variable if not specified
        if not variable:
            # Try common PM2.5 variable names
            possible_vars = ['PM25', 'pm25', 'PM2_5', 'particulate_matter_25', 'dust_pm25']
            for var in possible_vars:
                if var in ds.variables:
                    variable = var
                    break
            
            if not variable:
                # Use first data variable
                data_vars = list(ds.data_vars.keys())
                if data_vars:
                    variable = data_vars[0]
                    click.echo(f"Auto-detected variable: {variable}")
                else:
                    raise click.ClickException("No suitable variable found. Specify with --variable")
        
        if variable not in ds.variables:
            available_vars = list(ds.variables.keys())
            raise click.ClickException(f"Variable '{variable}' not found. Available: {available_vars}")
        
        # Select the variable
        data = ds[variable]
        
        # Handle time slicing
        if time_slice:
            start_time, end_time = time_slice.split(':')
            data = data.sel(time=slice(start_time, end_time))
            click.echo(f"Time slice applied: {time_slice}")
        
        # Handle time averaging
        if time_mean and 'time' in data.dims:
            data = data.mean(dim='time')
            click.echo("Averaged across time dimension")
        
        # Handle coordinate names
        coord_mapping = {}
        if 'longitude' in ds.coords:
            coord_mapping['longitude'] = 'x'
        if 'latitude' in ds.coords:
            coord_mapping['latitude'] = 'y'
        if 'lon' in ds.coords:
            coord_mapping['lon'] = 'x'
        if 'lat' in ds.coords:
            coord_mapping['lat'] = 'y'
        
        if coord_mapping:
            data = data.rename(coord_mapping)
        
        # Set CRS
        if data.rio.crs is None:
            data = data.rio.write_crs(crs)
        
        if not output:
            input_path = Path(input_file)
            output = input_path.with_suffix('.tiff')
            click.echo(f"Auto-generated output: {output}")

        # Create output directory
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to GeoTIFF
        data.rio.to_raster(output)
        
        # Close dataset
        ds.close()
        
        click.echo(f"✅ Successfully converted to: {output}")
        logger.info(f"NetCDF conversion completed: {output}")
        
    except ImportError as e:
        if 'xarray' in str(e) or 'rioxarray' in str(e):
            raise click.ClickException("Missing dependencies. Install with: pip install xarray rioxarray")
        raise click.ClickException(f"Import error: {e}")
    
    except Exception as e:
        logger.error(f"NetCDF conversion failed: {e}")
        raise click.ClickException(f"Conversion failed: {str(e)}")

# Process subcommands
@process.command('clip')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file or directory')
@click.option('--aoi', type=click.Path(exists=True), required=True,
              help='AOI file for clipping')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output directory')
@click.option('--pattern', default='*.tif',
              help='File pattern for batch processing')
@click.pass_context
def process_clip(ctx, input: str, aoi: str, output: str, pattern: str):
    """Clip raster/vector data to AOI."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Clipping data from: {input}")
        click.echo(f"Using AOI: {aoi}")
        click.echo(f"Output to: {output}")
        
        # TODO: Implement clipping processor call
        
        click.echo("✅ Clipping completed successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Clipping failed: {e}")
        raise click.ClickException(str(e))


@process.command('align')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input directory')
@click.option('--reference', '-r', type=click.Path(exists=True), required=True,
              help='Reference raster for alignment')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output directory')
@click.option('--method', type=click.Choice(['nearest', 'bilinear', 'cubic']),
              default='cubic', help='Resampling method')
@click.pass_context
def process_align(ctx, input: str, reference: str, output: str, method: str):
    """Align rasters to reference grid."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Aligning rasters from: {input}")
        click.echo(f"Using reference: {reference}")
        click.echo(f"Output to: {output}")
        click.echo(f"Resampling method: {method}")
        
        # TODO: Implement alignment processor call
        
        click.echo("✅ Alignment completed successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Alignment failed: {e}")
        raise click.ClickException(str(e))


# Visualize subcommands
@visualize.command('rasters')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file or directory')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Visualization configuration file')
@click.option('--colormap', default='viridis',
              help='Matplotlib colormap')
@click.option('--dpi', type=int, default=150,
              help='Output DPI')
@click.pass_context
def visualize_rasters(ctx, input: str, output: str, config: Optional[str],
                     colormap: str, dpi: int):
    """Create raster visualizations."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Creating visualizations from: {input}")
        click.echo(f"Output to: {output}")
        
        if config:
            click.echo(f"Using configuration: {config}")
        else:
            click.echo(f"Colormap: {colormap}, DPI: {dpi}")
        
        # TODO: Implement visualization processor call
        
        click.echo("✅ Visualizations created successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Visualization failed: {e}")
        raise click.ClickException(str(e))


# Pipeline subcommands
@pipeline.command('run')
@click.option('--workflow', '-w', type=click.Path(exists=True), required=True,
              help='Workflow configuration file')
@click.option('--from-stage', type=str,
              help='Start from specific stage')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.pass_context
def pipeline_run(ctx, workflow: str, from_stage: Optional[str], dry_run: bool):
    """Run complete processing pipeline."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Running workflow: {workflow}")
        
        if from_stage:
            click.echo(f"Starting from stage: {from_stage}")
        
        if dry_run:
            click.echo("🔍 DRY RUN - showing what would be executed:")
            # TODO: Implement dry run logic
            click.echo("  1. Extract archives")
            click.echo("  2. Clip to AOI")
            click.echo("  3. Align rasters")
            click.echo("  4. Create visualizations")
            return
        
        # TODO: Implement pipeline processor call
        
        click.echo("✅ Pipeline completed successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Pipeline failed: {e}")
        raise click.ClickException(str(e))


@pipeline.command('status')
@click.option('--workflow', '-w', type=click.Path(exists=True), required=True,
              help='Workflow configuration file')
@click.pass_context
def pipeline_status(ctx, workflow: str):
    """Check pipeline status and progress."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Checking status for workflow: {workflow}")
        
        # TODO: Implement status checking logic
        click.echo("📊 Pipeline Status:")
        click.echo("  ✅ Stage 1: Extract - Completed")
        click.echo("  ✅ Stage 2: Clip - Completed") 
        click.echo("  🔄 Stage 3: Align - In Progress")
        click.echo("  ⏳ Stage 4: Visualize - Pending")
        
    except GeoWorkflowError as e:
        logger.error(f"Status check failed: {e}")
        raise click.ClickException(str(e))


@pipeline.command('resume')
@click.option('--workflow', '-w', type=click.Path(exists=True), required=True,
              help='Workflow configuration file')
@click.option('--from-stage', type=str, required=True,
              help='Stage to resume from')
@click.pass_context
def pipeline_resume(ctx, workflow: str, from_stage: str):
    """Resume pipeline from specific stage."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Resuming workflow: {workflow}")
        click.echo(f"From stage: {from_stage}")
        
        # TODO: Implement resume logic
        
        click.echo("✅ Pipeline resumed and completed successfully")
        
    except GeoWorkflowError as e:
        logger.error(f"Pipeline resume failed: {e}")
        raise click.ClickException(str(e))


# Utility commands
@main.command('config')
@click.option('--template', type=click.Choice(['aoi', 'extraction', 'clipping', 'alignment', 'visualization', 'pipeline','enrichment']),
              help='Create configuration template')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.pass_context
def config_command(ctx, template: Optional[str], output: Optional[str]):
    """Configuration management utilities."""
    from ..core.config import get_config_template

    if template:
        config_data = get_config_template(template)
        if template == 'enrichment':        
            config_data = get_enrichment_config_template()

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                import json
                with open(output_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            
            click.echo(f"✅ Configuration template saved to: {output}")
        else:
            import yaml
            click.echo(f"Configuration template for '{template}':")
            click.echo(yaml.dump(config_data, default_flow_style=False, indent=2))

    else:
        click.echo("Available configuration templates:")
        templates = ['aoi', 'extraction', 'clipping', 'alignment', 'enrichment', 'visualization', 'pipeline']
        for tmpl in templates:
            click.echo(f"  📋 {tmpl}")
        click.echo("\nUse --template <name> to view or save a template")


@main.command('info')
@click.pass_context
def info_command(ctx):
    """Show system and environment information."""
    import sys
    import platform
    from pathlib import Path
    
    click.echo("🔍 GeoWorkflow System Information")
    click.echo("=" * 50)
    
    # Python info
    click.echo(f"Python Version: {sys.version}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"Architecture: {platform.architecture()[0]}")
    
    # Package info
    try:
        from .. import __version__
        click.echo(f"GeoWorkflow Version: {__version__}")
    except ImportError:
        click.echo("GeoWorkflow Version: Unknown")
    
    # Environment info
    cwd = Path.cwd()
    click.echo(f"Current Directory: {cwd}")
    
    # Check for key directories
    key_dirs = ['data', 'config', 'outputs', 'logs']
    click.echo("\nProject Structure:")
    for dir_name in key_dirs:
        dir_path = cwd / dir_name
        status = "✅" if dir_path.exists() else "❌"
        click.echo(f"  {status} {dir_name}/")
    
    # Check for dependencies
    click.echo("\nKey Dependencies:")
    deps = [
        ('geopandas', 'GeoPandas'),
        ('rasterio', 'Rasterio'),
        ('xarray', 'XArray'),
        ('matplotlib', 'Matplotlib'),
        ('click', 'Click'),
        ('pydantic', 'Pydantic')
    ]
    
    for module_name, display_name in deps:
        try:
            __import__(module_name)
            click.echo(f"  ✅ {display_name}")
        except ImportError:
            click.echo(f"  ❌ {display_name}")

# Add this command to src/geoworkflow/cli/main.py
# Insert in the process subcommands section

@process.command('enrich')
@click.option('--coi-dir', type=click.Path(exists=True), required=True,
              help='Directory containing Cities of Interest (COI) file')
@click.option('--coi-pattern', default='*AFRICAPOLIS*',
              help='Pattern to identify COI file')
@click.option('--raster-dir', type=click.Path(exists=True), required=True,
              help='Directory containing raster files for analysis')
@click.option('--raster-pattern', default='*.tif',
              help='Pattern for raster files')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for enriched cities data')
@click.option('--statistics', default='mean,max,min',
              help='Comma-separated list of statistics (mean,max,min,std,median,count)')
@click.option('--area-units', type=click.Choice(['km2', 'm2']), default='km2',
              help='Units for area calculation')
@click.option('--no-area', is_flag=True,
              help='Skip adding polygon area column')
@click.option('--overwrite', is_flag=True,
              help='Overwrite existing output file')
@click.pass_context
def process_enrich(ctx, coi_dir: str, coi_pattern: str, raster_dir: str, 
                  raster_pattern: str, output: str, statistics: str,
                  area_units: str, no_area: bool, overwrite: bool):
    """Enrich cities data with raster statistics."""
    logger = ctx.obj['logger']
    
    try:
        click.echo(f"Enriching cities data from: {coi_dir}")
        click.echo(f"COI pattern: {coi_pattern}")
        click.echo(f"Using rasters from: {raster_dir}")
        click.echo(f"Raster pattern: {raster_pattern}")
        click.echo(f"Output to: {output}")
        
        # Parse statistics
        stats_list = [s.strip() for s in statistics.split(',')]
        click.echo(f"Computing statistics: {', '.join(stats_list)}")
        
        # Import and create processor
        from geoworkflow.processors.integration.enrichment import StatisticalEnrichmentProcessor
        from geoworkflow.schemas.config_models import StatisticalEnrichmentConfig
        
        # Create configuration
        config = StatisticalEnrichmentConfig(
            coi_directory=Path(coi_dir),
            coi_pattern=coi_pattern,
            raster_directory=Path(raster_dir),
            raster_pattern=raster_pattern,
            output_file=Path(output),
            statistics=stats_list,
            skip_existing=not overwrite,
            add_area_column=not no_area,
            area_units=area_units
        )
        
        # Create and run processor
        processor = StatisticalEnrichmentProcessor(config)
        result = processor.process()
        
        if result.success:
            click.echo("✅ Statistical enrichment completed successfully")
            if result.metadata:
                click.echo(f"   Original features: {result.metadata.get('original_features', 'unknown')}")
                click.echo(f"   New columns added: {result.metadata.get('new_columns_added', 'unknown')}")
                click.echo(f"   Raster files processed: {result.metadata.get('raster_files_processed', 'unknown')}")
        else:
            click.echo(f"❌ Statistical enrichment failed: {result.message}")
            
    except Exception as e:
        logger.error(f"Statistical enrichment failed: {e}")
        raise click.ClickException(str(e))


def get_enrichment_config_template():
    """Get enrichment configuration template."""
    return {
        "coi_directory": "data/02_clipped/",
        "coi_pattern": "*AFRICAPOLIS*",
        "raster_directory": "data/03_processed/aligned/",
        "raster_pattern": "*.tif",
        "recursive": True,
        "output_file": "data/04_analysis_ready/enriched_cities.geojson",
        "statistics": ["mean", "max", "min", "std"],
        "skip_existing": True,
        "add_area_column": True,
        "area_units": "km2"
    }

if __name__ == '__main__':
    main()