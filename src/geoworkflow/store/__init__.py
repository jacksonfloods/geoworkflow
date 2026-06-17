"""geoworkflow.store — the unified hex database (geoworkflow's primary interface).

A single queryable database over all African cities, keyed by hexagon, backed by
a DuckDB + Parquet warehouse and built reproducibly from a YAML recipe.

    from geoworkflow.store import open_hexdb
    db = open_hexdb()
    db["KEN"]["Nairobi"]["lst_day"].climatology(month=6).plot()   # avg of all Junes
    db["KEN"]["Nairobi"]["lst_day"].sel("2020-06").to_geodataframe()
    db.hexagon("UTM32737_HQ-008188_R+031618")                     # one hexagon, any city
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from geoworkflow.schemas.config_models import (
    HexDBConfig, HexDBRecipe, GridSpec, MetricSpec, CitySelector, TimeRange,
)
from geoworkflow.store.accessors import (
    GeoHexDB, CountryView, CityView, VariableView, GroupedView,
    ClimatologyResult, Selection, NeedsReducerError,
)
from geoworkflow.store.catalog import (
    Catalog, CityRecord, AmbiguousCityError, CityNotFoundError,
)

__all__ = [
    "open_hexdb", "GeoHexDB", "HexDBConfig", "HexDBRecipe",
    "GridSpec", "MetricSpec", "CitySelector", "TimeRange",
    "CountryView", "CityView", "VariableView", "GroupedView",
    "ClimatologyResult", "Selection",
    "Catalog", "CityRecord",
    "AmbiguousCityError", "CityNotFoundError", "NeedsReducerError",
]


def open_hexdb(config: Optional[HexDBConfig] = None, **overrides) -> GeoHexDB:
    """Open the hex database for querying.

    Args:
        config: a :class:`HexDBConfig`; if omitted, defaults are used (relative to
            the current working directory). To call from *another directory* (e.g.
            a sibling project), either pass ``HexDBConfig.from_root("/path/to/
            africa_cities")`` or set the ``HEXDB_ROOT`` environment variable to that
            folder — then a bare ``open_hexdb()`` resolves the warehouse from it.
        **overrides: field overrides applied on top of ``config`` (or defaults),
            e.g. ``open_hexdb(warehouse_dir="/abs/data/hexdb")``.
    """
    if config is None:
        root = os.environ.get("HEXDB_ROOT")
        config = (HexDBConfig.from_root(root, **overrides) if root
                  else HexDBConfig(**overrides))
    elif overrides:
        config = config.model_copy(update=overrides)

    from geoworkflow.store.catalog import Catalog
    from geoworkflow.store.engine import DuckDBEngine
    from geoworkflow.store.metadata import load_metadata

    catalog = Catalog(
        Path(config.manifest_csv), Path(config.hexagglo_dir), Path(config.warehouse_dir),
        complexity_csv=config.complexity_csv,
    )
    engine = DuckDBEngine()
    metadata = load_metadata(config.warehouse_dir)
    return GeoHexDB(config, catalog, engine, metadata)
