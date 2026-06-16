"""Time selection grammar for the hex database — pandas/xarray conventions.

The query API deliberately keeps **selection** and **aggregation** separate, the
way pandas (`df.loc["2020-06"]` partial-string indexing) and xarray
(`ds.sel(time="2020-06")`, `ds.groupby("time.month").mean()`) do:

  * *Selection* picks which timesteps via an ISO partial-string or a ``slice`` —
    ``"2020"`` (a whole year), ``"2020-06"`` (one month), ``slice("2019-06",
    "2020-05")`` (a range). It never aggregates.
  * *Aggregation* ("all Junes") is the explicit xarray groupby-climatology idiom,
    handled in the accessors via :func:`months_of_season` etc., not by a string.

So this module is locale-free on purpose: it parses **ISO strings and integer
months only** — no ``"June"`` / month-name parsing (ambiguous, English-only). A
recurring-month average is expressed as ``.groupby_month().mean().sel(month=6)``
or ``.climatology(month=6)``, where ``6`` is an int.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

__all__ = [
    "TimeSelectorError",
    "TimeSelector",
    "parse_time_selector",
    "selection_sql",
    "SEASONS",
    "months_of_season",
    "season_of_month",
    "validate_month",
]


class TimeSelectorError(ValueError):
    """Raised for an unparseable or ambiguous time selector."""


_YEAR_RE = re.compile(r"^(?P<year>\d{4})$")
_YEAR_MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})$")

# Standard meteorological seasons (climatology convention).
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def validate_month(month: int) -> int:
    """Return ``month`` if it is a valid 1–12 integer, else raise."""
    if not isinstance(month, int) or isinstance(month, bool) or not (1 <= month <= 12):
        raise TimeSelectorError(f"month must be an integer 1-12, got {month!r}")
    return month


def months_of_season(season: str) -> tuple:
    """Months belonging to a season name (e.g. ``"JJA"`` -> ``(6, 7, 8)``)."""
    key = season.upper()
    if key not in SEASONS:
        raise TimeSelectorError(
            f"unknown season {season!r}; expected one of {sorted(SEASONS)}"
        )
    return SEASONS[key]


def season_of_month(month: int) -> str:
    """Season name containing ``month`` (e.g. ``7`` -> ``"JJA"``)."""
    validate_month(month)
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class TimeSelector:
    """A parsed ISO selection: an optional ``year`` and/or ``month`` filter.

    ``(2020, 6)`` is a single month; ``(2020, None)`` a whole year;
    ``(None, None)`` the full period.
    """

    year: Optional[int] = None
    month: Optional[int] = None

    @property
    def is_instant(self) -> bool:
        """True when both year and month are pinned (selects one timestep)."""
        return self.year is not None and self.month is not None

    def sql(self, col: str = "time") -> str:
        """SQL ``WHERE`` fragment for this selection (``""`` = no filter)."""
        parts = []
        if self.year is not None:
            parts.append(f"year({col}) = {self.year}")
        if self.month is not None:
            parts.append(f"month({col}) = {self.month}")
        return " AND ".join(parts)


def parse_time_selector(s: Union[str, None]) -> TimeSelector:
    """Parse a single ISO token into a :class:`TimeSelector`.

    Accepted: ``None`` / ``""`` / ``"all"`` (full period), ``"YYYY"`` (a year),
    ``"YYYY-MM"`` / ``"YYYY-M"`` (one month). Everything else — month names,
    bare days (``"2020-06-15"``), out-of-range months — raises
    :class:`TimeSelectorError`.
    """
    if s is None:
        return TimeSelector()
    if not isinstance(s, str):
        raise TimeSelectorError(
            f"time selector must be a string or None, got {type(s).__name__}"
        )
    token = s.strip()
    if token == "" or token.lower() == "all":
        return TimeSelector()

    m = _YEAR_RE.match(token)
    if m:
        return TimeSelector(year=int(m.group("year")))

    m = _YEAR_MONTH_RE.match(token)
    if m:
        month = validate_month(int(m.group("month")))
        return TimeSelector(year=int(m.group("year")), month=month)

    raise TimeSelectorError(
        f"unparseable time selector {s!r}; use ISO forms 'YYYY' or 'YYYY-MM' "
        f"(month names are not accepted — use an integer month= for climatology)"
    )


def _iso_bound(token: str, *, upper: bool) -> str:
    """Turn an ISO ``"YYYY"`` / ``"YYYY-MM"`` endpoint into a full date string,
    snapping to the first (lower) or last (upper) day of the period."""
    sel = parse_time_selector(token)
    if sel.year is None:
        raise TimeSelectorError(f"slice bound must be 'YYYY' or 'YYYY-MM', got {token!r}")
    if sel.month is None:
        # whole year: lower = Jan 1; upper = first of next year (exclusive, use with '<')
        return f"{sel.year + 1:04d}-01-01" if upper else f"{sel.year:04d}-01-01"
    if upper:
        # whole month: first of next month (exclusive upper bound, use with '<')
        ny, nm = (sel.year + 1, 1) if sel.month == 12 else (sel.year, sel.month + 1)
        return f"{ny:04d}-{nm:02d}-01"
    return f"{sel.year:04d}-{sel.month:02d}-01"


def selection_sql(selection: Union[str, slice, None], col: str = "time") -> str:
    """SQL ``WHERE`` fragment for a selection token or ``slice`` (``""`` = all).

    A ``slice(start, stop)`` becomes a half-open range
    ``col >= start AND col < stop_exclusive`` so that ``slice("2019", "2020")``
    spans all of 2019–2020 inclusive of whole-period endpoints.
    """
    if isinstance(selection, slice):
        if selection.step is not None:
            raise TimeSelectorError("time slice step is not supported")
        parts = []
        if selection.start is not None:
            parts.append(f"{col} >= '{_iso_bound(str(selection.start), upper=False)}'")
        if selection.stop is not None:
            parts.append(f"{col} < '{_iso_bound(str(selection.stop), upper=True)}'")
        return " AND ".join(parts)
    return parse_time_selector(selection).sql(col)
