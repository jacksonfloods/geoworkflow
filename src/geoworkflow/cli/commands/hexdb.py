"""CLI for the hex database (``geoworkflow hexdb ...``).

The concrete, wired realization of the ``pipeline run`` idea for this database:
``build`` (re)generates the warehouse from a recipe; ``query``/``hexagon``/``plot``
read it; ``add-city``/``add-metric``/``remove`` edit it and keep provenance current.
"""

from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _config(warehouse, hexagglo, manifest, city_dir, global_dir, registry, complexity):
    """Build a HexDBConfig from CLI paths (only overriding what's given)."""
    from geoworkflow.schemas.config_models import HexDBConfig
    overrides = {}
    if warehouse:   overrides["warehouse_dir"] = warehouse
    if hexagglo:    overrides["hexagglo_dir"] = hexagglo
    if manifest:    overrides["manifest_csv"] = manifest
    if city_dir:    overrides["city_dir"] = city_dir
    if global_dir:  overrides["global_dir"] = global_dir
    if registry:    overrides["raster_registry"] = registry
    if complexity:  overrides["complexity_csv"] = complexity
    return HexDBConfig(**overrides)


# shared path options applied to every subcommand
def _paths(f):
    f = click.option("--warehouse", type=click.Path(path_type=Path), help="Warehouse dir (data/hexdb)")(f)
    f = click.option("--hexagglo", type=click.Path(path_type=Path), help="Hex-grid GeoPackages dir")(f)
    f = click.option("--manifest", type=click.Path(path_type=Path), help="Grid zone manifest CSV")(f)
    f = click.option("--city-dir", type=click.Path(path_type=Path), help="Per-city rasters dir")(f)
    f = click.option("--global-dir", type=click.Path(path_type=Path), help="Global rasters dir")(f)
    f = click.option("--registry", type=click.Path(path_type=Path), help="Raster dataset registry JSON")(f)
    f = click.option("--complexity", type=click.Path(path_type=Path), help="Agglomeration complexity CSV")(f)
    return f


def _open(cfg):
    from geoworkflow.store import open_hexdb
    return open_hexdb(cfg)


@click.group()
def hexdb():
    """Build and query the unified hex database."""
    pass


@hexdb.command()
@_paths
@click.option("--recipe", type=click.Path(path_type=Path), help="Recipe YAML (default: built-in default recipe)")
@click.option("--workers", default=16, show_default=True, help="Parallel workers")
@click.option("--overwrite", is_flag=True, help="Rebuild partitions even if present")
@click.option("--reuse/--no-reuse", default=True, show_default=True, help="Ingest existing grid_stats parquets")
@click.option("--grid-stats-dir", type=click.Path(path_type=Path), help="Legacy grid_stats dir for reuse")
@click.option("--fetch", is_flag=True, help="Download missing rasters from Earth Engine first")
def build(recipe, workers, overwrite, reuse, grid_stats_dir, fetch, **paths):
    """Build (or resume) the warehouse from a recipe."""
    from geoworkflow.store import builder
    from geoworkflow.schemas.config_models import HexDBRecipe
    from geoworkflow.store.recipe import default_recipe
    cfg = _config(**paths)
    rcp = HexDBRecipe.from_yaml(recipe) if recipe else default_recipe()
    if overwrite:
        rcp = rcp.model_copy(update={"overwrite": True})
    res = builder.build(rcp, cfg, max_workers=workers, reuse=reuse,
                        grid_stats_dir=grid_stats_dir, fetch=fetch)
    console.print(f"[bold green]done[/] built={res.done} skipped={res.skipped} "
                  f"failed={res.failed} in {res.seconds:.0f}s")


@hexdb.command(name="add-city")
@_paths
@click.option("--iso3", required=True)
@click.option("--name")
@click.option("--aggid", type=int)
@click.option("--overwrite", is_flag=True)
@click.option("--grid-stats-dir", type=click.Path(path_type=Path))
def add_city(iso3, name, aggid, overwrite, grid_stats_dir, **paths):
    """Add one city, built consistently with the live recipe."""
    from geoworkflow.store import builder
    status = builder.add_city(_config(**paths), iso3, name, aggid=aggid,
                              overwrite=overwrite, grid_stats_dir=grid_stats_dir)
    console.print(f"add-city {iso3}/{name or aggid}: [bold]{status}[/]")


@hexdb.command(name="add-metric")
@_paths
@click.option("--metric-yaml", required=True, type=click.Path(exists=True, path_type=Path),
              help="YAML of a single MetricSpec")
@click.option("--workers", default=8, show_default=True)
def add_metric(metric_yaml, workers, **paths):
    """Compute a new metric for all built cities; append to the live recipe."""
    import yaml
    from geoworkflow.store import builder
    from geoworkflow.schemas.config_models import MetricSpec
    metric = MetricSpec.model_validate(yaml.safe_load(Path(metric_yaml).read_text()))
    res = builder.add_metric(_config(**paths), metric, max_workers=workers)
    console.print(f"add-metric {metric.name}: built={res.done} failed={res.failed}")


@hexdb.command()
@_paths
@click.option("--iso3")
@click.option("--name")
@click.option("--aggid", type=int)
@click.option("--metric", help="Remove a variable (metric) from all cities instead of a city")
def remove(iso3, name, aggid, metric, **paths):
    """Remove a city (default) or a metric (--metric VARIABLE)."""
    from geoworkflow.store import builder
    cfg = _config(**paths)
    if metric:
        n = builder.remove_metric(cfg, metric)
        console.print(f"remove-metric {metric}: updated {n} partitions")
    else:
        status = builder.remove_city(cfg, iso3, name, aggid=aggid)
        console.print(f"remove-city {iso3}/{name or aggid}: [bold]{status}[/]")


def _variable_view(db, iso3, name, aggid, variable, statistic, sel,
                   reduce, climatology_month):
    city = db.city(iso3, name, aggid=aggid)
    v = city[variable]
    if statistic:
        v = v.use(statistic)
    if climatology_month is not None:
        return v.climatology(month=climatology_month)
    if reduce:
        return getattr(v, reduce)(sel)
    return v.sel(sel) if sel else v


@hexdb.command()
@_paths
@click.option("--iso3", required=True)
@click.option("--name")
@click.option("--aggid", type=int)
@click.option("--variable", required=True)
@click.option("--statistic")
@click.option("--sel", help="ISO time selection, e.g. 2020-06 or 2020")
@click.option("--reduce", type=click.Choice(["mean", "max", "min", "median", "std"]))
@click.option("--climatology-month", type=int, help="All-years mean for this month (1-12)")
@click.option("--out", type=click.Path(path_type=Path), help="Write result to .parquet/.csv")
def query(iso3, name, aggid, variable, statistic, sel, reduce, climatology_month, out, **paths):
    """Query per-hex values for a city/variable (UC3)."""
    db = _open(_config(**paths))
    res = _variable_view(db, iso3, name, aggid, variable, statistic, sel, reduce, climatology_month)
    df = res.to_frame()
    if out:
        (df.to_parquet if str(out).endswith(".parquet") else df.to_csv)(out, index=False)
        console.print(f"wrote {len(df)} rows -> {out}")
    else:
        console.print(df.head(20).to_string(index=False))
        console.print(f"[dim]{len(df)} hexes[/]")


@hexdb.command()
@_paths
@click.argument("gridid")
@click.option("--aggid", type=int)
@click.option("--variable", multiple=True)
@click.option("--statistic")
@click.option("--out", type=click.Path(path_type=Path))
def hexagon(gridid, aggid, variable, statistic, out, **paths):
    """All data for a single hexagon, any city (UC1)."""
    db = _open(_config(**paths))
    df = db.hexagon(gridid, aggid=aggid, variables=list(variable) or None, statistic=statistic)
    if out:
        (df.to_parquet if str(out).endswith(".parquet") else df.to_csv)(out, index=False)
        console.print(f"wrote {len(df)} rows -> {out}")
    else:
        console.print(df.to_string(index=False))


@hexdb.command()
@_paths
@click.option("--iso3", required=True)
@click.option("--name")
@click.option("--aggid", type=int)
@click.option("--variable", help="If omitted, plot the city's bare hexagons (UC2)")
@click.option("--statistic")
@click.option("--sel")
@click.option("--reduce", type=click.Choice(["mean", "max", "min", "median", "std"]))
@click.option("--climatology-month", type=int)
@click.option("-o", "--out", required=True, type=click.Path(path_type=Path))
@click.option("--cmap", default="viridis", show_default=True)
def plot(iso3, name, aggid, variable, statistic, sel, reduce, climatology_month, out, cmap, **paths):
    """Plot a city's hexagons, optionally coloured by a variable (UC2/UC3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = _open(_config(**paths))
    fig, ax = plt.subplots(figsize=(9, 9))
    if variable:
        res = _variable_view(db, iso3, name, aggid, variable, statistic, sel, reduce, climatology_month)
        res.plot(ax=ax, cmap=cmap)
        ax.set_title(f"{name or aggid} — {variable}")
    else:
        db.city(iso3, name, aggid=aggid).plot(ax=ax, facecolor="none", edgecolor="#C7D1DE", linewidth=0.4)
        ax.set_title(f"{name or aggid} — hex grid")
    ax.set_axis_off()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    console.print(f"wrote {out}")


@hexdb.command()
@_paths
@click.option("--iso3", help="List a country's cities (else list countries)")
def info(iso3, **paths):
    """Show built countries/cities."""
    db = _open(_config(**paths))
    if iso3:
        for c in db[iso3].cities():
            console.print(c)
    else:
        table = Table(title="Built countries")
        table.add_column("ISO3")
        for c in db.countries():
            table.add_row(c)
        console.print(table)


@hexdb.command()
@_paths
@click.option("--log", is_flag=True, help="Show the append-only op log")
def provenance(log, **paths):
    """Show the live recipe and provenance history."""
    from geoworkflow.store import provenance as prov
    cfg = _config(**paths)
    wh = cfg.warehouse_dir
    if log:
        for rec in prov.read_log(wh):
            console.print(rec)
    else:
        rcp = prov.read_recipe(wh)
        if rcp is None:
            console.print("[yellow]no live recipe (warehouse not built)[/]")
        else:
            console.print(f"recipe: {prov.recipe_path(wh)}")
            console.print(f"grid: {rcp.grid.side_length}m {rcp.grid.crs_mode} | "
                          f"time: {rcp.time.start}..{rcp.time.end}")
            console.print("metrics: " + ", ".join(m.name for m in rcp.metrics))
