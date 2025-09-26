"""
AOI (Area of Interest) commands for the geoworkflow CLI.

This module provides commands for creating, validating, and managing
Areas of Interest.
"""

from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console

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
        from geoworkflow.schemas.config_models import AOIConfig
        
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