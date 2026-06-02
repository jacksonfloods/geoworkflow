"""
Pluggable registry of zonal statistics for grid/raster summarization.

A *zonal statistic* reduces the raster cells that intersect a polygon (for
example a hexagon) to a single value. Statistics are computed with
``exactextract``, whose coverage-fraction weighting makes the default
``weighted_mean`` a true area-weighted mean even when a polygon is smaller than
one raster pixel (the case rasterstats gets wrong, returning null).

Two kinds of statistic are supported:

1. **op-backed** — identified by an exactextract operation string (e.g.
   ``"mean"``, ``"sum"``). These run in exactextract's C++ core and are fast.
   Most built-ins are of this kind.
2. **reducer-backed** — a Python callable ``reducer(values, coverage) -> float``
   that post-processes the raw cell values and per-cell coverage fractions that
   exactextract returns for each feature. Use this to add metrics exactextract
   does not provide.

Add a statistic with :func:`register_statistic`::

    # op-backed (delegates to an exactextract op)
    register_statistic("range_max", op="max", description="maximum value")

    # reducer-backed (decorator over f(values, coverage) -> float)
    @register_statistic("range", description="max minus min")
    def _value_range(values, coverage):
        return float(values.max() - values.min())

Parameterized statistics are resolved on demand: ``percentile_90`` / ``p90``
and ``quantile_0.9`` both map to exactextract ``quantile(q=0.9)`` — no
registration needed.

This module is deliberately lightweight (numpy only). The compute engine that
actually runs exactextract lives in :mod:`geoworkflow.utils.zonal_utils`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# A reducer receives the intersecting cell values and their coverage fractions
# (both 1-D numpy arrays of equal length) and returns a single float.
ReducerFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ZonalStatistic:
    """Definition of a single zonal statistic.

    Exactly one of ``exactextract_op`` or ``reducer`` must be set.
    """

    name: str
    exactextract_op: Optional[str] = None
    reducer: Optional[ReducerFn] = None
    description: str = ""
    categorical: bool = False  # hint: meaningful on categorical rasters (e.g. land cover)

    def __post_init__(self) -> None:
        if (self.exactextract_op is None) == (self.reducer is None):
            raise ValueError(
                f"Statistic '{self.name}' must define exactly one of "
                "`exactextract_op` or `reducer` (got "
                f"op={self.exactextract_op!r}, reducer={self.reducer!r})."
            )

    @property
    def is_reducer(self) -> bool:
        """True if this statistic is computed by a Python reducer."""
        return self.reducer is not None


# The canonical registry. Keys are statistic names as users request them.
STATISTIC_REGISTRY: Dict[str, ZonalStatistic] = {}


def _register(stat: ZonalStatistic, *, overwrite: bool = False) -> ZonalStatistic:
    if stat.name in STATISTIC_REGISTRY and not overwrite:
        raise ValueError(
            f"Statistic '{stat.name}' is already registered. "
            "Pass overwrite=True to replace it."
        )
    STATISTIC_REGISTRY[stat.name] = stat
    return stat


def register_statistic(
    name: str,
    *,
    op: Optional[str] = None,
    description: str = "",
    categorical: bool = False,
    overwrite: bool = False,
):
    """Register a zonal statistic.

    Two usage forms:

    * **op-backed** — pass ``op`` (an exactextract operation string). Returns the
      created :class:`ZonalStatistic`::

          register_statistic("range_max", op="max")

    * **reducer-backed** — omit ``op`` and use as a decorator over a reducer
      ``f(values, coverage) -> float``. Returns the original function so it
      remains callable::

          @register_statistic("range", description="max minus min")
          def value_range(values, coverage):
              return float(values.max() - values.min())
    """
    if op is not None:
        stat = ZonalStatistic(
            name=name,
            exactextract_op=op,
            description=description,
            categorical=categorical,
        )
        _register(stat, overwrite=overwrite)
        return stat

    def _decorator(reducer: ReducerFn) -> ReducerFn:
        stat = ZonalStatistic(
            name=name,
            reducer=reducer,
            description=description,
            categorical=categorical,
        )
        _register(stat, overwrite=overwrite)
        return reducer

    return _decorator


def unregister_statistic(name: str) -> None:
    """Remove a statistic from the registry (no-op if absent)."""
    STATISTIC_REGISTRY.pop(name, None)


def available_statistics() -> Dict[str, str]:
    """Return a ``{name: description}`` mapping of all registered statistics."""
    return {name: stat.description for name, stat in sorted(STATISTIC_REGISTRY.items())}


# Patterns for on-demand parameterized statistics.
_PERCENTILE_RE = re.compile(r"^(?:percentile_|p)(\d{1,3}(?:\.\d+)?)$")
_QUANTILE_RE = re.compile(r"^quantile_(\d*\.?\d+)$")


def resolve_statistic(name: str) -> ZonalStatistic:
    """Resolve a statistic name to a :class:`ZonalStatistic`.

    Looks up the registry first, then recognizes parameterized forms:

    * ``percentile_<n>`` / ``p<n>`` (n in 0..100) -> ``quantile(q=n/100)``
    * ``quantile_<q>`` (q in 0..1) -> ``quantile(q=<q>)``

    Raises:
        KeyError: if the name is neither registered nor a recognized form.
    """
    if name in STATISTIC_REGISTRY:
        return STATISTIC_REGISTRY[name]

    match = _PERCENTILE_RE.match(name)
    if match:
        pct = float(match.group(1))
        if not 0.0 <= pct <= 100.0:
            raise KeyError(f"Percentile out of range in '{name}' (need 0-100).")
        return ZonalStatistic(
            name=name,
            exactextract_op=f"quantile(q={pct / 100.0})",
            description=f"{match.group(1)}th percentile",
        )

    match = _QUANTILE_RE.match(name)
    if match:
        q = float(match.group(1))
        if not 0.0 <= q <= 1.0:
            raise KeyError(f"Quantile out of range in '{name}' (need 0-1).")
        return ZonalStatistic(
            name=name,
            exactextract_op=f"quantile(q={q})",
            description=f"quantile q={q}",
        )

    raise KeyError(
        f"Unknown statistic '{name}'. Registered: {sorted(STATISTIC_REGISTRY)}. "
        "Parameterized forms: percentile_<n>, p<n>, quantile_<q>."
    )


def resolve_statistics(names: List[str]) -> List[ZonalStatistic]:
    """Resolve a list of statistic names, preserving order and de-duplicating."""
    resolved: List[ZonalStatistic] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        resolved.append(resolve_statistic(name))
    return resolved


# ---------------------------------------------------------------------------
# Built-in statistics
# ---------------------------------------------------------------------------

# `weighted_mean` is the project default: exactextract's `mean` weights each
# cell value by the fraction of the cell covered by the polygon, i.e. a true
# area-weighted mean. `mean` is registered as an alias for discoverability.
register_statistic(
    "weighted_mean",
    op="mean",
    description="Area-weighted mean (each cell weighted by polygon-coverage fraction).",
)
register_statistic("mean", op="mean", description="Alias of weighted_mean.")
register_statistic("min", op="min", description="Minimum intersecting cell value.")
register_statistic("max", op="max", description="Maximum intersecting cell value.")
register_statistic("sum", op="sum", description="Coverage-weighted sum of cell values.")
register_statistic(
    "count",
    op="count",
    description="Sum of coverage fractions (effective number of pixels covered).",
)
register_statistic("median", op="median", description="Coverage-weighted median.")
register_statistic("stdev", op="stdev", description="Coverage-weighted standard deviation.")
register_statistic("std", op="stdev", description="Alias of stdev.")
register_statistic("variance", op="variance", description="Coverage-weighted variance.")
register_statistic(
    "coefficient_of_variation",
    op="coefficient_of_variation",
    description="Coverage-weighted stdev / mean.",
)
register_statistic(
    "majority",
    op="majority",
    description="Most common value (coverage-weighted). For categorical rasters.",
    categorical=True,
)
register_statistic(
    "minority",
    op="minority",
    description="Least common value (coverage-weighted). For categorical rasters.",
    categorical=True,
)
register_statistic(
    "variety",
    op="variety",
    description="Number of distinct values. For categorical rasters.",
    categorical=True,
)


@register_statistic(
    "range",
    description="Max minus min of intersecting cell values (reducer-backed example).",
)
def _value_range(values: np.ndarray, coverage: np.ndarray) -> float:
    return float(np.nanmax(values) - np.nanmin(values))
