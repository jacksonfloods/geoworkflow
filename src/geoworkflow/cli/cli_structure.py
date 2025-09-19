# File: src/geoworkflow/cli/__init__.py
"""
Command-line interface for the geoworkflow package.

This module provides a unified CLI that replaces the individual scripts
with a single entry point and multiple subcommands.
"""

from geoworkflow.cli.main import main

__all__ = ["main"]

# ---

# File: src/geoworkflow/cli/main.py
"""
Main CLI entry point for geoworkflow.

This module defines the main command group and coordinates all subcommands.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.traceback import install

from geoworkflow import __version__
from geoworkflow.core.logging_setup import setup_package_logging
from geoworkflow.core.exceptions import GeoWorkflowError

# Install rich traceback handling
install(show_locals=True)
console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="geoworkflow")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Set logging level",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    help="Directory to write log files",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress verbose output",
)
@click.pass_context
def cli(ctx, log_level: str, log_dir: Optional[Path], quiet: bool):
    """
    Geoworkflow: Comprehensive geospatial data processing for African analysis.
    
    This tool provides a unified interface for processing geospatial data including
    AOI creation, data extraction, clipping, alignment, and visualization.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store global options in context
    ctx.obj["quiet"] = quiet
    ctx.obj["log_level"] = log_level
    ctx.obj["log_dir"] = log_dir
    
    # Setup package logging
    setup_package_logging(level=log_level, log_dir=log_dir)
    
    if not quiet:
        console.print(f"[bold blue]Geoworkflow v{__version__}[/bold blue]")


def main():
    """Main entry point for the CLI."""
    try:
        # Import subcommands (this registers them with the main CLI group)
        from geoworkflow.cli.commands import aoi, extract, process, pipeline, visualize
        
        # Add subcommands to main group
        cli.add_command(aoi.aoi)
        cli.add_command(extract.extract)
        cli.add_command(process.process)
        cli.add_command(pipeline.pipeline)
        cli.add_command(visualize.visualize)
        
        # Run CLI
        cli()
        
    except GeoWorkflowError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}", err=True)
        if e.details:
            console.print(f"[dim]Details: {e.details}[/dim]", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]", err=True)
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}", err=True)
        console.print("[dim]Run with --log-level DEBUG for more details[/dim]", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# ---

# File: src/geoworkflow/cli/commands/__init__.py
"""
CLI command modules for geoworkflow.

This package contains all the command-line subcommands organized by functionality.
"""

# Command modules will be imported individually to register with main CLI

# ---

# File: src/geoworkflow/cli/commands/aoi.py
"""
AOI (Area of Interest) commands for the geoworkflow CLI.

This module provides commands for creating, validating, and managing
Areas of Interest.
"""

from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console

from geoworkflow.core.config import AOIConfig
from geoworkflow.core.exceptions import ConfigurationError

console = Console()


@click.group()
def aoi():
    """Area of Interest (AOI) management commands."""
    pass


@aoi.command()
@click.option(
    "--input-file", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to administrative boundaries file (GeoJSON/Shapefile)",
)
@click.option(
    "--countries",
    help="Comma-separated list of countries to extract",
)
@click.option(
    "--all-countries",
    is_flag=True,
    help="Use all countries in the boundaries file",
)
@click.option(
    "--dissolve/--no-dissolve",
    default=True,
    help="Dissolve country boundaries into single polygon",
)
@click.option(
    "--buffer", "-b",
    type=float,
    default=100,
    help="Buffer distance in kilometers",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output AOI file path",
)
@click.option(
    "--country-column",
    default="NAME_0",
    help="Column name containing country names",
)
@click.pass_context
def create(ctx, input_file: Path, countries: Optional[str], all_countries: bool,
          dissolve: bool, buffer: float, output: Path, country_column: str):
    """Create a new Area of Interest (AOI) from administrative boundaries."""
    
    try:
        # Parse countries list
        country_list = None
        if countries:
            country_list = [c.strip() for c in countries.split(",")]
        
        # Create AOI configuration
        aoi_config = AOIConfig(
            input_file=input_file,
            country_name_column=country_column,
            countries=country_list,
            use_all_countries=all_countries,
            dissolve_boundaries=dissolve,
            buffer_km=buffer,
            output_file=output,
        )
        
        # TODO: Import and use AOIProcessor when implemented
        console.print(f"[green]AOI configuration created successfully![/green]")
        console.print(f"Configuration: {aoi_config}")
        console.print("[yellow]Note: AOI processor implementation pending[/yellow]")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create AOI: {str(e)}")


@aoi.command()
@click.argument("aoi_file", type=click.Path(exists=True, path_type=Path))
def validate(aoi_file: Path):
    """Validate an AOI file."""
    try:
        # TODO: Implement AOI validation
        console.print(f"[green]AOI file is valid: {aoi_file}[/green]")
        console.print("[yellow]Note: AOI validation implementation pending[/yellow]")
        
    except Exception as e:
        console.print(f"[red]AOI validation failed: {str(e)}[/red]")
        raise click.ClickException("AOI validation failed")


@aoi.command()
@click.option(
    "--boundaries-file", "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to administrative boundaries file",
)
@click.option(
    "--country-column",
    default="NAME_0", 
    help="Column name containing country names",
)
@click.option(
    "--prefix", "-p",
    help="Show only countries starting with this prefix",
)
def list_countries(boundaries_file: Path, country_column: str, prefix: Optional[str]):
    """List available countries in a boundaries file."""
    try:
        # TODO: Implement country listing
        console.print(f"[green]Countries in {boundaries_file}:[/green]")
        console.print("[yellow]Note: Country listing implementation pending[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to list countries: {str(e)}[/red]")
        raise click.ClickException("Country listing failed")

# ---

# File: src/geoworkflow/cli/commands/extract.py
"""
Data extraction commands for the geoworkflow CLI.

This module provides commands for extracting data from archives
and converting between formats.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

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

# ---

# File: src/geoworkflow/cli/commands/process.py
"""
Data processing commands for the geoworkflow CLI.

This module provides commands for clipping, aligning, and processing
geospatial data.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()


@click.group()
def process():
    """Data processing commands."""
    pass


@process.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input raster file or directory",
)
@click.option(
    "--aoi",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Area of Interest file for clipping",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory or file",
)
@click.option(
    "--pattern",
    default="*.tif",
    help="File pattern to match (for directory input)",
)
@click.option(
    "--all-touched",
    is_flag=True,
    help="Include all pixels touched by AOI geometry",
)
def clip(input: Path, aoi: Path, output: Path, pattern: str, all_touched: bool):
    """Clip raster data to Area of Interest."""
    
    console.print(f"[blue]Clipping data:[/blue] {input}")
    console.print(f"[blue]Using AOI:[/blue] {aoi}")
    console.print(f"[blue]Output:[/blue] {output}")
    
    # TODO: Implement clipping processor
    console.print("[yellow]Note: Clipping implementation pending[/yellow]")


@process.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory containing rasters to align",
)
@click.option(
    "--reference", "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Reference raster for alignment",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for aligned rasters",
)
@click.option(
    "--method",
    type=click.Choice(["nearest", "bilinear", "cubic", "average"]),
    default="cubic",
    help="Resampling method",
)
@click.option(
    "--recursive",
    is_flag=True,
    help="Search input directory recursively",
)
def align(input: Path, reference: Path, output: Path, method: str, recursive: bool):
    """Align rasters to a reference grid."""
    
    console.print(f"[blue]Aligning rasters from:[/blue] {input}")
    console.print(f"[blue]Reference raster:[/blue] {reference}")
    console.print(f"[blue]Output directory:[/blue] {output}")
    console.print(f"[blue]Resampling method:[/blue] {method}")
    
    # TODO: Implement alignment processor
    console.print("[yellow]Note: Alignment implementation pending[/yellow]")