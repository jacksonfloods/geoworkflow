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

from click.exceptions import Exit as ClickExit

from geoworkflow.schemas.config_models import OpenBuildingsExtractionConfig
from geoworkflow.core.exceptions import GeoWorkflowError, ExtractionError, ConfigurationError

from geoworkflow.utils.earth_engine_error_handler import (
    EarthEngineErrorHandler, 
    validate_earth_engine_prerequisites
)
from geoworkflow.core.exceptions import (
    EarthEngineError,
    EarthEngineAuthenticationError,
    EarthEngineQuotaError,
    EarthEngineTimeoutError,
    get_academic_friendly_error_message
)

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
    '--method', 
    type=click.Choice(['gcs', 'earthengine', 'auto']),
    default='gcs',
    help='Extraction method: gcs (default, fastest), earthengine (legacy), auto (try gcs first)'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration YAML file for Open Buildings extraction'
)
@click.option(
    '--aoi-file', 
    type=click.Path(exists=True, path_type=Path), 
    help='Area of Interest boundary file (GeoJSON/Shapefile) [REQUIRED if no config]'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for extracted buildings [REQUIRED if no config]'
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
    '--workers', 
    type=int, 
    default=4,
    help='Number of parallel workers (GCS method only, default: 4)'
)
@click.option(
    '--service-account', 
    type=click.Path(exists=True, path_type=Path),
    help='Path to Google Cloud service account key JSON file'
)
@click.option(
    '--project-id', 
    type=str,
    help='Google Cloud Project ID (Earth Engine method only)'
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
def open_buildings(ctx, method, config, aoi_file, output_dir, confidence, export_format,
                  workers, service_account, project_id, min_area, max_area, overwrite):
    """
    Extract building footprints from Google Open Buildings v3 dataset.
    
    This command supports two extraction methods:
    
    \b
    Methods:
      gcs (default):    Direct GCS download - 3-5x faster, no quotas
      earthengine:      Earth Engine API - use for EE-specific workflows  
      auto:             Try GCS first, fallback to Earth Engine if needed
    
    \b
    Quick Start:
        # Basic extraction with GCS (fastest)
        geoworkflow extract open-buildings \\
            --aoi-file boundary.geojson \\
            --output-dir ./buildings
        
        # With custom settings
        geoworkflow extract open-buildings \\
            --aoi-file boundary.geojson \\
            --output-dir ./buildings \\
            --confidence 0.8 \\
            --workers 8 \\
            --format geojson
        
        # Using configuration file
        geoworkflow extract open-buildings --config config.yaml
    """
    
    quiet = ctx.obj.get("quiet", False)
    
    # Try GCS method first (if method is 'gcs' or 'auto')
    if method in ['gcs', 'auto']:
        try:
            from geoworkflow.processors.extraction.open_buildings_gcs import OpenBuildingsGCSProcessor
            from geoworkflow.schemas.open_buildings_gcs_config import OpenBuildingsGCSConfig
            
            if not quiet:
                console.print("[blue]Using GCS extraction method (fastest)[/blue]")
            
            # Build configuration
            if config:
                if not quiet:
                    console.print(f"[blue]Loading configuration from:[/blue] {config}")
                
                import yaml
                with open(config, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # CLI overrides
                cli_overrides = {}
                if aoi_file: cli_overrides['aoi_file'] = aoi_file
                if output_dir: cli_overrides['output_dir'] = output_dir
                if service_account: cli_overrides['service_account_key'] = service_account
                if confidence != 0.75: cli_overrides['confidence_threshold'] = confidence
                if export_format != 'geojson': cli_overrides['export_format'] = export_format
                if workers != 4: cli_overrides['num_workers'] = workers
                if min_area != 10.0: cli_overrides['min_area_m2'] = min_area
                if max_area: cli_overrides['max_area_m2'] = max_area
                if overwrite: cli_overrides['overwrite_existing'] = overwrite
                
                config_dict.update(cli_overrides)
            else:
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
                    'num_workers': workers,
                    'service_account_key': service_account,
                    'min_area_m2': min_area,
                    'max_area_m2': max_area,
                    'overwrite_existing': overwrite,
                    'use_anonymous_access': service_account is None
                }
                
                config_dict = {k: v for k, v in config_dict.items() if v is not None}
            
            # Create config and processor
            gcs_config = OpenBuildingsGCSConfig(**config_dict)
            
            if not quiet:
                console.print("\n[bold blue]Open Buildings Extraction (GCS)[/bold blue]")
                console.print(f"[blue]AOI File:[/blue] {gcs_config.aoi_file}")
                console.print(f"[blue]Output Directory:[/blue] {gcs_config.output_dir}")
                console.print(f"[blue]Confidence Threshold:[/blue] {gcs_config.confidence_threshold}")
                console.print(f"[blue]Export Format:[/blue] {gcs_config.export_format}")
                console.print(f"[blue]Parallel Workers:[/blue] {gcs_config.num_workers}")
                console.print(f"[blue]Min Building Area:[/blue] {gcs_config.min_area_m2} m²")
                if gcs_config.max_area_m2:
                    console.print(f"[blue]Max Building Area:[/blue] {gcs_config.max_area_m2} m²")
            
            # Run extraction
            processor = OpenBuildingsGCSProcessor(gcs_config)
            result = processor.process()
            
            if result.success:
                console.print(f"\n[green]✅ {result.message}[/green]")
                console.print(f"[green]Output saved to:[/green] {result.output_paths[0]}")
                return  # Success - exit
            elif method == 'auto':
                console.print("[yellow]⚠ GCS method failed, trying Earth Engine...[/yellow]")
                method = 'earthengine'  # Fall through to EE
            else:
                console.print(f"[red]❌ Extraction failed: {result.message}[/red]")
                ctx.exit(1)
                
        except ImportError as e:
            if method == 'gcs':
                raise click.ClickException(
                    f"GCS method not available: {e}\n"
                    "Install with: pip install geoworkflow[extraction]"
                )
            console.print("[yellow]⚠ GCS dependencies not available, using Earth Engine...[/yellow]")
            method = 'earthengine'
        except Exception as e:
            if method == 'gcs':
                raise click.ClickException(f"GCS extraction failed: {e}")
            console.print(f"[yellow]⚠ GCS method failed: {e}[/yellow]")
            console.print("[yellow]Falling back to Earth Engine method...[/yellow]")
            method = 'earthengine'
    
    # Earth Engine method (original implementation)
    if method == 'earthengine':
        if not quiet:
            console.print("[blue]Using Earth Engine extraction method[/blue]")
            console.print("[dim]Note: This method is slower than GCS. Consider using --method gcs[/dim]")
        
        # Original Earth Engine implementation goes here...
        # (Keep existing EE code)
        pass

@click.pass_context
def open_buildings(ctx, config, aoi_file, output_dir, confidence, export_format, 
                  service_account, project_id, max_features, min_area, max_area, overwrite):
    """
    Extract building footprints from Google Open Buildings v3 dataset via Earth Engine.
    
    This command extracts building footprints for a specified Area of Interest (AOI) 
    from Google's Open Buildings dataset using the Earth Engine API.
    
    \b
    Quick Start:
        # Create configuration template
        geoworkflow config --template open-buildings --output my_config.yaml
        
        # Edit the template with your settings, then run:
        geoworkflow extract open-buildings --config my_config.yaml
    
    \b
    Examples:
        # Basic extraction with defaults
        geoworkflow extract open-buildings --aoi-file boundary.geojson --output-dir ./buildings
        
        # Using configuration file (recommended)
        geoworkflow extract open-buildings --config open_buildings_config.yaml
    """
    
    quiet = ctx.obj.get("quiet", False)
    
    try:
        # Step 1: Validate Earth Engine prerequisites
        if not quiet:
            console.print("[blue]Validating Earth Engine setup...[/blue]")
            
        validation_result = validate_earth_engine_prerequisites()
        
        if not validation_result["valid"]:
            console.print(f"\n[bold red]Setup Error:[/bold red]")
            for error in validation_result["errors"]:
                console.print(f"[red]• {error}[/red]")
            
            if validation_result["setup_guidance"]:
                console.print(f"\n[dim]Setup Guide:[/dim]")
                for guidance in validation_result["setup_guidance"]:
                    console.print(f"[dim]• {guidance}[/dim]")
            
            ctx.exit(1)
        
        # Show warnings if any
        if validation_result.get("warnings") and not quiet:
            for warning in validation_result["warnings"]:
                console.print(f"[yellow]Warning: {warning}[/yellow]")
        
        # Step 2: Configuration handling (existing code)
        if config:
            if not quiet:
                console.print(f"[blue]Loading configuration from:[/blue] {config}")
            
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
            
            config_dict.update(cli_overrides)
            
        else:
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
            
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Step 3: Create and validate configuration object
        buildings_config = OpenBuildingsExtractionConfig(**config_dict)
        
        if not quiet:
            console.print("\n[bold blue]Open Buildings Extraction[/bold blue]")
            console.print(f"[blue]AOI File:[/blue] {buildings_config.aoi_file}")
            console.print(f"[blue]Output Directory:[/blue] {buildings_config.output_dir}")
            console.print(f"[blue]Confidence Threshold:[/blue] {buildings_config.confidence_threshold}")
            console.print(f"[blue]Export Format:[/blue] {buildings_config.export_format}")
            console.print(f"[blue]Min Building Area:[/blue] {buildings_config.min_area_m2} m²")
            if buildings_config.max_features:
                console.print(f"[blue]Max Features:[/blue] {buildings_config.max_features:,}")
        
        # Step 4: Import and initialize processor
        try:
            from geoworkflow.processors.extraction.open_buildings import OpenBuildingsExtractionProcessor
        except ImportError as e:
            raise click.ClickException(
                f"Open Buildings processor not available: {e}\n"
                "Install Earth Engine dependencies with: pip install geoworkflow[earth-engine]"
            )
        
        processor = OpenBuildingsExtractionProcessor(buildings_config)
        
        # Step 5: Run extraction
        if not quiet:
            console.print("\n[yellow]Starting extraction...[/yellow]")
            
        result = processor.process()
        
        if result.success:
            console.print(f"\n[bold green]{result.message}[/bold green]")
            if result.output_paths:
                console.print(f"[green]Output:[/green] {', '.join(str(p) for p in result.output_paths)}")
            if hasattr(result, 'processed_count') and result.processed_count:
                console.print(f"[green]Buildings Extracted:[/green] {result.processed_count:,}")
        else:
            console.print(f"[bold red]Extraction failed:[/bold red] {result.message}")
            ctx.exit(1)
            
    # Enhanced Error Handling with Academic Guidance
    except EarthEngineAuthenticationError as e:
        console.print(f"\n[bold red]Authentication Error:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        ctx.exit(1)
        
    except EarthEngineQuotaError as e:
        console.print(f"\n[bold red]Quota Exceeded:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        console.print(f"\n[dim]Tip: Try reducing area size or increasing confidence threshold[/dim]")
        ctx.exit(1)
        
    except EarthEngineTimeoutError as e:
        console.print(f"\n[bold red]Operation Timed Out:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        console.print(f"\n[dim]Tip: Break large areas into smaller chunks[/dim]")
        ctx.exit(1)
        
    except EarthEngineError as e:
        console.print(f"\n[bold red]Earth Engine Error:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        ctx.exit(1)
        
    except ConfigurationError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        if "service account" in str(e).lower() or "authentication" in str(e).lower():
            console.print("\n[dim]Authentication Help:[/dim]")
            console.print("1. Get Earth Engine access: https://earthengine.google.com/signup/")
            console.print("2. Create service account key file")
            console.print("3. Use --service-account option or set GOOGLE_APPLICATION_CREDENTIALS")
        ctx.exit(1)
    
    except ExtractionError as e:
        console.print(f"\n[bold red]Extraction Error:[/bold red]")
        console.print(f"[red]{e.message}[/red]")
        ctx.exit(1)
    
    except (SystemExit, ClickExit):
        # Let both SystemExit and Click Exit pass through without catching them
        raise
        
    except Exception as e:
        # Try to classify unknown errors as Earth Engine errors
        if 'earth engine' in str(e).lower() or 'ee.' in str(e).lower():
            try:
                ee_error = EarthEngineErrorHandler.classify_and_handle_error(e, config_dict)
                console.print(f"\n[bold red]Earth Engine Error:[/bold red]")
                console.print(f"[red]{ee_error.message}[/red]")
            except:
                console.print(f"\n[bold red]Unexpected Error:[/bold red] {str(e)}")
                console.print("[dim]Run with --log-level DEBUG for more details[/dim]")
        else:
            console.print(f"\n[bold red]Unexpected Error:[/bold red] {str(e)}")
            console.print("[dim]Run with --log-level DEBUG for more details[/dim]")
        ctx.exit(1)