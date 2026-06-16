"""DuckDB query engine over the Hive-partitioned Parquet warehouse.

The accessors pass an explicit, scoped list of partition files (one file for a
city, a country's files for a country query), so DuckDB only ever opens what the
query needs — a single-city query never touches the other ~9,000 partitions.
Time selection/aggregation is pushed down to SQL (``WHERE year/month`` and
``GROUP BY GridID``), the SQL equivalents of the old ``hexcube`` helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from geoworkflow.core.exceptions import GeoWorkflowError

try:
    import duckdb
    import pandas as pd
    HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    HAS_DUCKDB = False

# reducer name -> DuckDB aggregate
_REDUCERS = {
    "mean": "avg", "avg": "avg", "max": "max", "min": "min",
    "median": "median", "std": "stddev_samp", "sum": "sum", "count": "count",
}

Source = Union[str, Path, Sequence[Union[str, Path]]]


def _source_arg(source: Source) -> Union[str, List[str]]:
    """Normalize a glob string, single path, or path list for ``read_parquet``."""
    if isinstance(source, str):
        return source            # glob or single-file path string
    if isinstance(source, Path):
        return [str(source)]     # one partition file
    return [str(p) for p in source]


def _reducer_sql(reducer: str) -> str:
    try:
        return _REDUCERS[reducer]
    except KeyError:
        raise GeoWorkflowError(
            f"unknown reducer {reducer!r}; use one of {sorted(_REDUCERS)}")


class DuckDBEngine:
    """Thin DuckDB wrapper exposing the queries the accessors need."""

    def __init__(self):
        if not HAS_DUCKDB:  # pragma: no cover
            raise GeoWorkflowError("duckdb is required for the hex database engine")
        self.con = duckdb.connect(database=":memory:")

    # -- helpers --------------------------------------------------------
    def _read(self, source: Source) -> str:
        """``read_parquet(...)`` table expression bound to parameter ``$src``."""
        return "read_parquet($src, union_by_name=true)"

    def _run(self, sql: str, params: dict) -> "pd.DataFrame":
        return self.con.execute(sql, params).df()

    # -- introspection --------------------------------------------------
    def variables(self, source: Source) -> List[str]:
        df = self._run(
            f"SELECT DISTINCT variable FROM {self._read(source)} ORDER BY variable",
            {"src": _source_arg(source)})
        return df["variable"].tolist()

    def statistics(self, source: Source, variable: str) -> List[str]:
        df = self._run(
            f"SELECT DISTINCT statistic FROM {self._read(source)} "
            f"WHERE variable = $var ORDER BY statistic",
            {"src": _source_arg(source), "var": variable})
        return df["statistic"].tolist()

    # -- the value query ------------------------------------------------
    def values(
        self,
        source: Source,
        variable: str,
        statistic: str,
        *,
        where_time: str = "",
        reducer: Optional[str] = None,
    ) -> "pd.DataFrame":
        """Per-hex values for one (variable, statistic).

        With ``reducer`` -> one row per GridID (``GROUP BY``); without -> the raw
        rows (GridID, time, value) for the selected timesteps.
        """
        where = ["variable = $var", "statistic = $stat"]
        if where_time:
            where.append(where_time)
        where_sql = " AND ".join(where)
        params = {"src": _source_arg(source), "var": variable, "stat": statistic}

        if reducer:
            agg = _reducer_sql(reducer)
            sql = (f"SELECT GridID, {agg}(value) AS value "
                   f"FROM {self._read(source)} WHERE {where_sql} "
                   f"GROUP BY GridID ORDER BY GridID")
        else:
            sql = (f"SELECT GridID, time, value FROM {self._read(source)} "
                   f"WHERE {where_sql} ORDER BY GridID, time")
        return self._run(sql, params)

    def climatology(
        self,
        source: Source,
        variable: str,
        statistic: str,
        *,
        reducer: str = "mean",
        by: str = "month",
    ) -> "pd.DataFrame":
        """Per-hex climatology: one value per (GridID, month|season-month)."""
        if by != "month":
            raise GeoWorkflowError("climatology currently groups by month only")
        agg = _reducer_sql(reducer)
        sql = (f"SELECT GridID, month(time) AS month, {agg}(value) AS value "
               f"FROM {self._read(source)} WHERE variable = $var AND statistic = $stat "
               f"GROUP BY GridID, month(time) ORDER BY GridID, month")
        return self._run(sql, {"src": _source_arg(source), "var": variable, "stat": statistic})

    # -- single hexagon (use case 1) ------------------------------------
    def hexagon(
        self,
        source: Source,
        gridid: str,
        *,
        variables: Optional[List[str]] = None,
        statistic: Optional[str] = None,
    ) -> "pd.DataFrame":
        """All tidy rows for one GridID (optionally filtered to variables/stat)."""
        where = ["GridID = $gid"]
        params = {"src": _source_arg(source), "gid": gridid}
        if variables:
            where.append("variable IN (SELECT unnest($vars))")
            params["vars"] = list(variables)
        if statistic:
            where.append("statistic = $stat")
            params["stat"] = statistic
        sql = (f"SELECT GridID, ISO3, aggid, variable, time, statistic, value, units "
               f"FROM {self._read(source)} WHERE {' AND '.join(where)} "
               f"ORDER BY variable, statistic, time")
        return self._run(sql, params)

    def locate_gridid(self, source: Source, gridid: str) -> "pd.DataFrame":
        """Distinct (ISO3, aggid) partitions that contain a GridID."""
        sql = (f"SELECT DISTINCT ISO3, aggid FROM {self._read(source)} "
               f"WHERE GridID = $gid")
        return self._run(sql, {"src": _source_arg(source), "gid": gridid})

    def close(self):
        self.con.close()
