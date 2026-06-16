"""The chained accessor API: ``db["KEN"]["Nairobi"]["lst_day"].mean("2020")``.

Lazy drill-down — each ``__getitem__`` returns the next view and touches no data
until a terminal call (a reducer, ``.climatology``, ``.to_frame/.to_geodataframe/
.plot/.values``, or ``.hexagon``). Time follows pandas/xarray conventions
(see :mod:`geoworkflow.store.timesel`): ``.sel(...)`` selects, reducers aggregate,
``.groupby_month()`` is the climatology idiom.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from geoworkflow.core.exceptions import GeoWorkflowError
from geoworkflow.store import geometry as geom
from geoworkflow.store.timesel import (
    parse_time_selector, selection_sql, validate_month, months_of_season,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False


class NeedsReducerError(GeoWorkflowError):
    """A selection spans multiple timesteps; apply a reducer (.mean(), ...)."""


def _pick_default_statistic(stats: Sequence[str]) -> str:
    for preferred in ("weighted_mean", "majority", "mean"):
        if preferred in stats:
            return preferred
    if not stats:
        raise GeoWorkflowError("variable has no statistics in the database")
    return stats[0]


class _CityResult:
    """Shared materialization for a per-hex result of ONE city.

    Subclasses implement ``_values()`` -> DataFrame[GridID, value].
    """

    def __init__(self, db, record, variable: str):
        self._db = db
        self._record = record
        self._variable = variable

    def _values(self) -> "pd.DataFrame":  # pragma: no cover - abstract
        raise NotImplementedError

    def to_frame(self) -> "pd.DataFrame":
        """Per-hex values as a DataFrame ``[GridID, value]``."""
        return self._values()

    def values(self) -> "pd.Series":
        """Per-hex values as a Series indexed by GridID."""
        df = self._values()
        return pd.Series(df["value"].values, index=df["GridID"].values, name=self._variable)

    def to_geodataframe(self, *, value_col: str = "value"):
        """Join the per-hex values onto this city's hex geometry (native UTM)."""
        gdf = geom.load_hex_geometry(self._record.gpkg)
        return geom.join_values(gdf, self._values(), out_col=value_col)

    def plot(self, *, ax=None, column: str = "value", legend: bool = True, **kw):
        """Choropleth of the per-hex values on this city's geometry."""
        gdf = self.to_geodataframe(value_col=column)
        return gdf.plot(column=column, ax=ax, legend=legend, **kw)


class Selection(_CityResult):
    """A materialized per-hex result (already reduced to one value per hex)."""

    def __init__(self, db, record, variable, frame: "pd.DataFrame"):
        super().__init__(db, record, variable)
        self._frame = frame

    def _values(self) -> "pd.DataFrame":
        return self._frame


class ClimatologyResult:
    """Per-(GridID, month) climatology; pick a month to get a per-hex map."""

    def __init__(self, db, record, variable, frame: "pd.DataFrame"):
        self._db, self._record, self._variable, self._frame = db, record, variable, frame

    def to_frame(self) -> "pd.DataFrame":
        return self._frame

    def sel(self, *, month: int) -> Selection:
        validate_month(month)
        sub = self._frame[self._frame["month"] == month][["GridID", "value"]]
        return Selection(self._db, self._record, self._variable, sub.reset_index(drop=True))


class GroupedView:
    """Result of ``.groupby_month()`` — reduce it to a climatology."""

    def __init__(self, db, record, variable, statistic):
        self._db, self._record = db, record
        self._variable, self._statistic = variable, statistic

    def _reduce(self, reducer: str) -> ClimatologyResult:
        frame = self._db._engine.climatology(
            self._record.partition, self._variable, self._statistic,
            reducer=reducer, by="month")
        return ClimatologyResult(self._db, self._record, self._variable, frame)

    def mean(self) -> ClimatologyResult:   return self._reduce("mean")
    def max(self) -> ClimatologyResult:    return self._reduce("max")
    def min(self) -> ClimatologyResult:    return self._reduce("min")
    def median(self) -> ClimatologyResult: return self._reduce("median")


class VariableView(_CityResult):
    """``db[iso3][city][variable]`` — choose statistic/time, reduce, or plot."""

    def __init__(self, db, record, variable, *, statistic=None,
                 selection: Union[str, slice, None] = None):
        super().__init__(db, record, variable)
        self._statistic = statistic or _pick_default_statistic(
            db._engine.statistics(record.partition, variable))
        self._selection = selection

    def _clone(self, **kw) -> "VariableView":
        params = dict(statistic=self._statistic, selection=self._selection)
        params.update(kw)
        return VariableView(self._db, self._record, self._variable, **params)

    # -- configuration (lazy) -------------------------------------------
    def use(self, statistic: str) -> "VariableView":
        return self._clone(statistic=statistic)

    def sel(self, time: Union[str, slice]) -> "VariableView":
        if not isinstance(time, slice):
            parse_time_selector(time)   # validate eagerly (clear error site)
        return self._clone(selection=time)

    # -- reducers (terminal -> Selection) -------------------------------
    def _reduce(self, reducer: str) -> Selection:
        frame = self._db._engine.values(
            self._record.partition, self._variable, self._statistic,
            where_time=selection_sql(self._selection), reducer=reducer)
        return Selection(self._db, self._record, self._variable, frame)

    def mean(self, time: Union[str, slice, None] = None) -> Selection:
        view = self.sel(time) if time is not None else self
        return view._reduce("mean")

    def max(self, time=None) -> Selection:
        return (self.sel(time) if time is not None else self)._reduce("max")

    def min(self, time=None) -> Selection:
        return (self.sel(time) if time is not None else self)._reduce("min")

    def median(self, time=None) -> Selection:
        return (self.sel(time) if time is not None else self)._reduce("median")

    def std(self, time=None) -> Selection:
        return (self.sel(time) if time is not None else self)._reduce("std")

    # -- climatology ----------------------------------------------------
    def groupby_month(self) -> GroupedView:
        return GroupedView(self._db, self._record, self._variable, self._statistic)

    def climatology(self, *, month: Optional[int] = None,
                    season: Optional[str] = None):
        """All-years climatology. ``month=6`` -> the all-Junes map (a Selection);
        ``season="JJA"`` -> mean over Jun/Jul/Aug; neither -> the 12-month result."""
        clim = self.groupby_month().mean()
        if month is not None and season is not None:
            raise GeoWorkflowError("pass month= or season=, not both")
        if month is not None:
            return clim.sel(month=month)
        if season is not None:
            months = months_of_season(season)
            sub = clim.to_frame()
            sub = sub[sub["month"].isin(months)]
            agg = sub.groupby("GridID", as_index=False)["value"].mean()
            return Selection(self._db, self._record, self._variable, agg)
        return clim

    # -- direct terminal (no reducer): requires an instant selection ----
    def _values(self) -> "pd.DataFrame":
        raw = self._db._engine.values(
            self._record.partition, self._variable, self._statistic,
            where_time=selection_sql(self._selection), reducer=None)
        dup = raw["GridID"].duplicated().any()
        if dup:
            raise NeedsReducerError(
                f"selection {self._selection!r} spans multiple timesteps for "
                f"'{self._variable}'; apply a reducer (.mean(), .max(), ...) or "
                f"select a single month with .sel('YYYY-MM')")
        return raw[["GridID", "value"]].reset_index(drop=True)


class CityView:
    """``db["KEN"]["Nairobi"]`` — a city's variables and geometry."""

    def __init__(self, db, record):
        self._db = db
        self.record = record

    def __getitem__(self, variable: str) -> VariableView:
        return VariableView(self._db, self.record, variable)

    def variables(self) -> List[str]:
        return self._db._engine.variables(self.record.partition)

    def to_geodataframe(self):
        """The city's hex geometry (no values) — use case 2."""
        return geom.load_hex_geometry(self.record.gpkg)

    def plot(self, **kw):
        """Plot the city's hexagons — use case 2."""
        return self.to_geodataframe().plot(**kw)

    def hexagon(self, gridid: str, **kw) -> "pd.DataFrame":
        return self._db._engine.hexagon(self.record.partition, gridid, **kw)


class CountryView:
    """``db["KEN"]`` — the cities of one country."""

    def __init__(self, db, iso3: str):
        self._db = db
        self.iso3 = iso3

    def __getitem__(self, name: str) -> CityView:
        rec = self._db._catalog.resolve(self.iso3, name)
        return CityView(self._db, rec)

    def city(self, *, aggid: int) -> CityView:
        return CityView(self._db, self._db._catalog.resolve(self.iso3, aggid=aggid))

    def cities(self) -> List[str]:
        return self._db._built_cities(self.iso3)


class GeoHexDB:
    """Entry point: a queryable view over the whole hex-database warehouse.

    Use :func:`geoworkflow.store.open_hexdb` to construct one. Index by ISO3 to
    drill in (``db["KEN"]["Nairobi"]["lst_day"]``); use :meth:`hexagon` for a
    single hexagon in any city; the ``add_*``/``remove_*`` methods edit the DB
    (delegating to the builder) and keep provenance current.
    """

    def __init__(self, config, catalog, engine, metadata):
        self._config = config
        self._catalog = catalog
        self._engine = engine
        self._metadata = metadata
        self._warehouse = Path(config.warehouse_dir)

    # -- drill-down -----------------------------------------------------
    def __getitem__(self, iso3: str) -> CountryView:
        return CountryView(self, iso3)

    def city(self, iso3: str, name: Optional[str] = None, *, aggid: Optional[int] = None) -> CityView:
        return CityView(self, self._catalog.resolve(iso3, name, aggid=aggid))

    # -- built-state (partition existence) ------------------------------
    def _partition_aggids(self, iso3: Optional[str] = None) -> List[int]:
        pat = (f"ISO3={iso3}/aggid=*/part.parquet" if iso3
               else "ISO3=*/aggid=*/part.parquet")
        out = []
        for p in self._warehouse.glob(pat):
            try:
                out.append(int(p.parent.name.split("=", 1)[1]))
            except (IndexError, ValueError):
                continue
        return out

    def countries(self) -> List[str]:
        return sorted({self._catalog.resolve_aggid(a).iso3
                       for a in self._partition_aggids()})

    def _built_cities(self, iso3: str) -> List[str]:
        return sorted(self._catalog.resolve_aggid(a).display_name
                      for a in self._partition_aggids(iso3))

    def legend(self, variable: str):
        """``{code: (label, color)}`` for a categorical variable (e.g. landcover)."""
        return self._metadata.legend(variable)

    # -- single hexagon, any city (use case 1) --------------------------
    def hexagon(self, gridid: str, *, aggid: Optional[int] = None,
                variables: Optional[List[str]] = None,
                statistic: Optional[str] = None) -> "pd.DataFrame":
        if aggid is not None:
            rec = self._catalog.resolve_aggid(aggid)
            return self._engine.hexagon(rec.partition, gridid,
                                        variables=variables, statistic=statistic)
        candidates = [r for r in self._catalog.find_hexagon(gridid)
                      if r.partition.exists()]
        if not candidates:
            raise GeoWorkflowError(
                f"no built city in the UTM zone of GridID {gridid!r}; "
                f"pass aggid= if you know the city")
        located = self._engine.locate_gridid([r.partition for r in candidates], gridid)
        if len(located) == 0:
            raise GeoWorkflowError(f"GridID {gridid!r} not found in any built city")
        if len(located) > 1:
            pairs = list(located.itertuples(index=False, name=None))
            raise GeoWorkflowError(
                f"GridID {gridid!r} occurs in {len(located)} cities {pairs}; "
                f"pass aggid= to disambiguate")
        aid = int(located.iloc[0]["aggid"])
        rec = self._catalog.resolve_aggid(aid)
        return self._engine.hexagon(rec.partition, gridid,
                                    variables=variables, statistic=statistic)

    # -- mutation (delegate to builder; keeps provenance current) -------
    def add_city(self, iso3: str, name: Optional[str] = None, *,
                 aggid: Optional[int] = None, overwrite: bool = False):
        from geoworkflow.store import builder
        return builder.add_city(self._config, iso3, name, aggid=aggid, overwrite=overwrite)

    def remove_city(self, iso3: str, name: Optional[str] = None, *,
                    aggid: Optional[int] = None):
        from geoworkflow.store import builder
        return builder.remove_city(self._config, iso3, name, aggid=aggid)

    def add_metric(self, metric, *, max_workers: int = 8):
        from geoworkflow.store import builder
        return builder.add_metric(self._config, metric, max_workers=max_workers)

    def remove_metric(self, variable: str):
        from geoworkflow.store import builder
        return builder.remove_metric(self._config, variable)

    def close(self):
        self._engine.close()
