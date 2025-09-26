#src/geoworkflow/cli/main.py
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
from geoworkflow.core.logging_setup import setup_logging
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
    setup_logging(level=log_level, log_file=log_dir)
    
    if not quiet:
        console.print(f"[bold blue]Geoworkflow v{__version__}[/bold blue]")

# Add this command after the cli() function in main.py

@cli.command('config')
@click.option('--template', 
              type=click.Choice(['aoi', 'extraction', 'clipping', 'alignment', 'visualization', 'pipeline', 'enrichment', 'open-buildings']),
              help='Create configuration template')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.pass_context
def config_command(ctx, template: Optional[str], output: Optional[str]):
    """Configuration management utilities."""
    from geoworkflow.core.config import get_config_template

    if template:
        # You'll need to update get_config_template() to include 'open-buildings'
        config_data = get_config_template(template)

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
            
            console.print(f"[green]✅ Configuration template saved to: {output}[/green]")
        else:
            import yaml
            console.print(f"[blue]Configuration template for '{template}':[/blue]")
            console.print(yaml.dump(config_data, default_flow_style=False, indent=2))

    else:
        console.print("[blue]Available configuration templates:[/blue]")
        templates = ['aoi', 'extraction', 'clipping', 'alignment', 'enrichment', 'visualization', 'pipeline', 'open-buildings']
        for tmpl in templates:
            console.print(f"  📋 {tmpl}")
        console.print("\n[dim]Use --template <name> to view or save a template[/dim]")

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
        error_console = Console(stderr=True)
        error_console.print(f"[bold red]Error:[/bold red] {e.message}")
        if e.details:
            error_console.print(f"[dim]Details: {e.details}[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        error_console = Console(stderr=True)
        error_console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        error_console = Console(stderr=True)
        error_console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        error_console.print("[dim]Run with --log-level DEBUG for more details[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()