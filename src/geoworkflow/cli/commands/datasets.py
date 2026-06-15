"""
Dataset-registry commands for the geoworkflow CLI.

These help users verify the raster dataset registry (which file-naming
conventions are known, and what the registry would extract from a given file)
*before* running a pipeline -- so a hand-edited ``data/config/raster_datasets.json``
entry can be checked without launching a big job.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from geoworkflow.core.dataset_registry import (
    AmbiguousDatasetError,
    load_dataset_registry,
)

console = Console()


@click.group()
def datasets():
    """Inspect the raster dataset registry."""
    pass


@datasets.command(name="list")
@click.option(
    "--registry", "-r",
    type=click.Path(exists=True, path_type=Path),
    help="User registry JSON (default: data/config/raster_datasets.json if present).",
)
def list_datasets(registry: Optional[Path]):
    """List all registered datasets (packaged defaults + user additions)."""
    reg = load_dataset_registry(user_registry=registry)
    table = Table(title="Registered raster datasets")
    table.add_column("name", style="bold")
    table.add_column("match")
    table.add_column("variable")
    table.add_column("time")
    table.add_column("description")
    for name in reg.names():
        spec = reg.get(name)
        time_mode = spec.time.mode if spec.time else "none"
        table.add_row(name, spec.match, spec.variable or "(auto)", time_mode,
                      spec.description)
    console.print(table)


@datasets.command()
@click.argument("file_path", type=str)
@click.option(
    "--registry", "-r",
    type=click.Path(exists=True, path_type=Path),
    help="User registry JSON (default: data/config/raster_datasets.json if present).",
)
@click.option(
    "--dataset", "-d",
    help="Force a specific dataset entry instead of auto-matching.",
)
def test(file_path: str, registry: Optional[Path], dataset: Optional[str]):
    """Show what the registry would extract from FILE_PATH (no file needed on disk)."""
    reg = load_dataset_registry(user_registry=registry)
    try:
        info = reg.describe_file(file_path, dataset=dataset)
    except AmbiguousDatasetError as exc:
        console.print(f"[yellow]ambiguous:[/yellow] {exc}")
        raise SystemExit(2)

    if info.get("matched") is None:
        console.print(
            f"[red]no match[/red] for '{info['file']}'. "
            f"Known datasets: {reg.names()}. "
            "Add an entry to data/config/raster_datasets.json or pass --dataset."
        )
        raise SystemExit(1)

    console.print(
        f"[green]matched[/green]: [bold]{info['matched']}[/bold]  |  "
        f"variable={info['variable']}  |  time={info['time']}  |  "
        f"crs={info['crs']}  |  units={info['units']}  |  "
        f"categorical={info['categorical']}"
    )
