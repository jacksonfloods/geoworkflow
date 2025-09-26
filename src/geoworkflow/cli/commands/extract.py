#/src/geoworkflow/cli/commands/extract.py
"""
Data extraction commands for the geoworkflow CLI.

This module provides commands for extracting data from archives
and converting between formats.
"""

import os
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
from geoworkflow.core.exceptions import GeoWorkflowError, ExtractionError, ConfigurationError

console = Console()


@click.group()
def extract():
    """Data extraction and conversion commands."""
    pass


@extract.command()
@click.option(
    "--source", "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Source directory containing archives or single archive file",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for extracted data",
)
@click.option(
    "--pattern",
    default="*.tif",
    help="File pattern to extract from archives",
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List archive contents without extracting",
)
@click.pass_context
def archives(ctx, source: Path, output: Path, pattern: str, list_only: bool):
    """Extract data from ZIP archives and other compressed formats."""
    
    quiet = ctx.obj.get("quiet", False)
    
    if not quiet:
        console.print(f"[blue]Extracting archives from:[/blue] {source}")
        console.print(f"[blue]Output directory:[/blue] {output}")
        console.print(f"[blue]File pattern:[/blue] {pattern}")
    
    if list_only:
        console.print("[yellow]Listing archive contents...[/yellow]")
        # TODO: Implement archive listing
        console.print("[yellow]Note: Archive extraction implementation pending[/yellow]")
    else:
        console.print("[yellow]Extracting archives...[/yellow]")
        # TODO: Implement archive extraction
        console.print("[yellow]Note: Archive extraction implementation pending[/yellow]")


@extract.command()
@click.argument("netcdf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output GeoTIFF file path",
)
@click.option(
    "--variable", "-v",
    default="PM25",
    help="Variable name to extract from NetCDF",
)
@click.option(
    "--crs",
    default="EPSG:4326",
    help="Target coordinate reference system",
)
@click.option(
    "--clip",
    type=click.Path(exists=True, path_type=Path),
    help="Vector file to clip the output",
)
def netcdf(netcdf_file: Path, output: Optional[Path], variable: str, crs: str, clip: Optional[Path]):
    """Convert NetCDF files to GeoTIFF format."""
    
    if output is None:
        output = netcdf_file.with_suffix('.tiff')
    
    console.print(f"[blue]Converting NetCDF:[/blue] {netcdf_file}")
    console.print(f"[blue]Output:[/blue] {output}")
    console.print(f"[blue]Variable:[/blue] {variable}")
    console.print(f"[blue]Target CRS:[/blue] {crs}")
    
    if clip:
        console.print(f"[blue]Clipping with:[/blue] {clip}")
    
    # TODO: Implement NetCDF conversion
    console.print("[yellow]Note: NetCDF conversion implementation pending[/yellow]")


@extract.command('open-buildings')
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration YAML file for Open Buildings extraction'
)
@click.option(
    '--aoi-file', 
    type=click.Path(exists=True, path_type=Path), 
    help='Area of Interest boundary file (GeoJSON/Shapefile) [REQUIRED]'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for extracted buildings [REQUIRED]'
)
@click.option(
    '--confidence', 
    type=float, 
    default=0.75,
    help='Building confidence threshold (0.5-1.0, default: 0.75)'
)
@click.option(
    '--format', 'export_format',
    type=click.Choice(['geojson', 'shapefile', 'csv']), 
    default='geojson',
    help='Export format (default: geojson)'
)
@click.option(
    '--service-account', 
    type=click.Path(exists=True, path_type=Path),
    help='Path to Google Cloud service account key JSON file'
)
@click.option(
    '--project-id', 
    type=str,
    help='Google Cloud Project ID (can be inferred from service account)'
)
@click.option(
    '--max-features', 
    type=int,
    help='Maximum number of buildings to extract (useful for large areas)'
)
@click.option(
    '--min-area', 
    type=float, 
    default=10.0,
    help='Minimum building area in m² (default: 10.0)'
)
@click.option(
    '--max-area', 
    type=float,
    help='Maximum building area in m² (no limit by default)'
)
@click.option(
    '--overwrite', 
    is_flag=True,
    help='Overwrite existing output files'
)
@click.pass_context
def extract_open_buildings(ctx, config, aoi_file, output_dir, confidence, export_format, 
                          service_account, project_id, max_features, min_area, max_area, overwrite):
    """
    Extract building footprints from Google Open Buildings v3 dataset via Earth Engine.
    
    This command extracts building footprints for a specified Area of Interest (AOI) 
    from Google's Open Buildings dataset using the Earth Engine API.
    
    \b
    Requirements:
    - Earth Engine access (academic accounts available)
    - Google Cloud service account or user authentication
    - AOI file in GeoJSON or Shapefile format
    
    \b
    Authentication Options (in order of precedence):
    1. --service-account: Path to service account JSON key
    2. GOOGLE_APPLICATION_CREDENTIALS environment variable
    3. Default Earth Engine user credentials
    
    \b
    Examples:
        # Basic extraction with defaults
        geoworkflow extract open-buildings --aoi-file boundary.geojson --output-dir ./buildings
        
        # Custom confidence and format with service account
        geoworkflow extract open-buildings \\
            --aoi-file accra_boundary.shp \\
            --output-dir ./accra_buildings \\
            --confidence 0.8 \\
            --format shapefile \\
            --service-account ./my-service-account-key.json
            
        # Large area with limits to avoid timeouts
        geoworkflow extract open-buildings \\
            --aoi-file large_region.geojson \\
            --output-dir ./buildings \\
            --max-features 50000 \\
            --min-area 25
        
        # Using configuration file (recommended for complex setups)
        geoworkflow extract open-buildings --config open_buildings_config.yaml
    
    \b
    Academic Setup Guide:
    1. Get Earth Engine access: https://earthengine.google.com/signup/
    2. Create Google Cloud Project & Service Account
    3. Download service account key JSON file
    4. Use --service-account option or set GOOGLE_APPLICATION_CREDENTIALS
    
    \b
    Tips:
    - Start with small AOIs for testing
    - Higher confidence thresholds = fewer but more reliable buildings
    - Use --max-features for large areas to avoid timeouts
    - GeoJSON format is most compatible across tools
    """
    
    quiet = ctx.obj.get("quiet", False)
    
    try:
        # Configuration priority: file > CLI args > defaults
        if config:
            if not quiet:
                console.print(f"[blue]Loading configuration from:[/blue] {config}")
            
            # Load from YAML config file
            import yaml
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # CLI arguments override config file values
            cli_overrides = {}
            if aoi_file: cli_overrides['aoi_file'] = aoi_file
            if output_dir: cli_overrides['output_dir'] = output_dir
            if service_account: cli_overrides['service_account_key'] = service_account
            if project_id: cli_overrides['project_id'] = project_id
            if max_features: cli_overrides['max_features'] = max_features
            if confidence != 0.75: cli_overrides['confidence_threshold'] = confidence
            if export_format != 'geojson': cli_overrides['export_format'] = export_format
            if min_area != 10.0: cli_overrides['min_area_m2'] = min_area
            if max_area: cli_overrides['max_area_m2'] = max_area
            if overwrite: cli_overrides['overwrite_existing'] = overwrite
            
            # Merge config file with CLI overrides
            config_dict.update(cli_overrides)
            
        else:
            # Build configuration entirely from CLI arguments
            if not aoi_file or not output_dir:
                raise click.ClickException(
                    "Either --config file OR both --aoi-file and --output-dir are required.\n"
                    "Use 'geoworkflow config --template open-buildings' to create a template."
                )
            
            config_dict = {
                'aoi_file': aoi_file,
                'output_dir': output_dir,
                'confidence_threshold': confidence,
                'export_format': export_format,
                'service_account_key': service_account,
                'project_id': project_id,
                'max_features': max_features,
                'min_area_m2': min_area,
                'max_area_m2': max_area,
                'overwrite_existing': overwrite
            }
            
            # Remove None values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Create and validate configuration object
        buildings_config = OpenBuildingsExtractionConfig(**config_dict)
        
        if not quiet:
            console.print("\n[bold blue]🏢 Open Buildings Extraction[/bold blue]")
            console.print(f"[blue]AOI File:[/blue] {buildings_config.aoi_file}")
            console.print(f"[blue]Output Directory:[/blue] {buildings_config.output_dir}")
            console.print(f"[blue]Confidence Threshold:[/blue] {buildings_config.confidence_threshold}")
            console.print(f"[blue]Export Format:[/blue] {buildings_config.export_format}")
            console.print(f"[blue]Min Building Area:[/blue] {buildings_config.min_area_m2} m²")
            if buildings_config.max_features:
                console.print(f"[blue]Max Features:[/blue] {buildings_config.max_features:,}")
        
        # Import and initialize processor (dynamic import to handle optional dependencies)
        try:
            from geoworkflow.processors.extraction.open_buildings import OpenBuildingsExtractionProcessor
        except ImportError as e:
            raise click.ClickException(
                f"Open Buildings processor not available: {e}\n"
                "Install Earth Engine dependencies with: pip install geoworkflow[earth-engine]\n"
                "Or manually: pip install earthengine-api google-auth google-cloud-storage"
            )
        
        # Initialize processor
        processor = OpenBuildingsExtractionProcessor(buildings_config)
        
        # Show academic setup guidance if authentication might be an issue
        if not buildings_config.service_account_key and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            console.print("\n[yellow]⚠️  No explicit authentication found.[/yellow]")
            console.print("For academic setup guidance, see:")
            console.print("https://earthengine.google.com/signup/")
        
        # Run extraction
        if not quiet:
            console.print("\n[yellow]🚀 Starting extraction...[/yellow]")
            
        result = processor.process()
        
        if result.success:
            console.print(f"\n[bold green]✅ {result.message}[/bold green]")
            if result.output_paths:
                console.print(f"[green]📁 Output:[/green] {', '.join(str(p) for p in result.output_paths)}")
            if hasattr(result, 'processed_count') and result.processed_count:
                console.print(f"[green]🏢 Buildings Extracted:[/green] {result.processed_count:,}")
        else:
            console.print(f"[bold red]❌ Extraction failed:[/bold red] {result.message}")
            ctx.exit(1)
            
    except ConfigurationError as e:
        console.print(f"\n[bold red]❌ Configuration Error:[/bold red]", err=True)
        console.print(f"[red]{e.message}[/red]", err=True)
        if "service account" in str(e).lower() or "authentication" in str(e).lower():
            console.print("\n[dim]💡 Authentication Help:[/dim]")
            console.print("1. Get Earth Engine access: https://earthengine.google.com/signup/")
            console.print("2. Create service account: https://cloud.google.com/iam/docs/service-accounts-create")
            console.print("3. Use --service-account option or set GOOGLE_APPLICATION_CREDENTIALS")
        ctx.exit(1)
        
    except ExtractionError as e:
        console.print(f"\n[bold red]❌ Extraction Error:[/bold red]", err=True)
        console.print(f"[red]{e.message}[/red]", err=True)
        if e.details:
            console.print(f"[dim]Details: {e.details}[/dim]", err=True)
        ctx.exit(1)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected Error:[/bold red] {str(e)}", err=True)
        console.print("[dim]Run with --log-level DEBUG for more details[/dim]", err=True)
        ctx.exit(1)