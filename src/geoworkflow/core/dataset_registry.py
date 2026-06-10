"""
User-editable registry of raster dataset naming conventions.

Different datasets encode their variable and acquisition time in their file
names in different (and often quirky) ways. Rather than hard-code that knowledge
in Python — which would force a code change and a pull request every time a new
dataset arrives — the lab describes datasets in a JSON file that anyone can edit.

Each entry is a *named recipe*::

    {
      "name": "odiac",
      "match": "odiac*_*.tif",                       # glob: which files this is
      "variable": "odiac_co2",
      "units": "gC/m2/day",
      "time": { "regex": "_(?P<year2>\\\\d{2})(?P<month>\\\\d{2})\\\\.tif$" }
    }

* ``match`` is a **glob** (familiar ``*`` wildcards) used to recognize files.
* ``time`` is the only place a **regex** appears, and only to mark which
  characters are the date. Named groups ``year`` / ``year2`` (2-digit) /
  ``month`` / ``day`` are recognized. Alternatives: ``{"coord": "time"}`` to read
  a netCDF time axis, ``{"static": "2020-01"}`` for a fixed date, or omit ``time``
  for a dataset with no temporal dimension.

A default registry ships inside the package. Users keep their own additions in
``data/raster_datasets.json``; on load the two are merged, with user entries
overriding package entries of the same ``name``.
"""

from __future__ import annotations

import fnmatch
import json
import re
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Conventional location for the user's own additions, relative to the working
# directory (the workspace data dir per the project's output convention).
DEFAULT_USER_REGISTRY = Path("data/raster_datasets.json")

# Recognized named groups in a time regex.
_YEAR_GROUPS = ("year", "year2")


class TimeSpec(BaseModel):
    """How to determine the timestamp of a file in a dataset.

    Exactly one mode may be set (or none, meaning the dataset is not temporal):
      * ``regex``  - extract from the filename via named groups.
      * ``coord``  - read from a coordinate of the file (e.g. a netCDF time axis).
      * ``static`` - a fixed date string (e.g. "2020-01"), parsed to a Timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    regex: Optional[str] = None
    coord: Optional[str] = None
    static: Optional[str] = None

    @field_validator("regex")
    @classmethod
    def _regex_compiles(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                re.compile(v)
            except re.error as exc:
                raise ValueError(f"invalid time regex {v!r}: {exc}")
        return v

    @field_validator("static")
    @classmethod
    def _static_parses(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                pd.Timestamp(v)
            except Exception as exc:  # noqa: BLE001 - surface a clear message
                raise ValueError(f"invalid static date {v!r}: {exc}")
        return v

    @model_validator(mode="after")
    def _exactly_one_or_none(self) -> "TimeSpec":
        set_modes = [m for m in ("regex", "coord", "static") if getattr(self, m) is not None]
        if len(set_modes) > 1:
            raise ValueError(
                f"time spec must set at most one of regex/coord/static, got {set_modes}"
            )
        return self

    @property
    def mode(self) -> str:
        if self.regex is not None:
            return "regex"
        if self.coord is not None:
            return "coord"
        if self.static is not None:
            return "static"
        return "none"


class RasterDatasetSpec(BaseModel):
    """One dataset's naming convention and metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    match: str
    variable: Optional[str] = None
    crs: Optional[str] = None
    units: Optional[str] = None
    nodata: Optional[float] = None
    categorical: bool = False
    description: str = ""
    time: Optional[TimeSpec] = None

    def matches(self, filename: str) -> bool:
        """True if ``filename`` (basename) matches this dataset's glob."""
        return fnmatch.fnmatch(filename, self.match)

    def time_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract a timestamp from the filename for regex/static specs.

        Returns None for non-temporal datasets, ``coord`` specs (the time lives
        in the file, resolved by the raster reader), or when a regex does not
        match. Day defaults to 1 when only year/month are captured.
        """
        if self.time is None:
            return None
        if self.time.mode == "static":
            return pd.Timestamp(self.time.static)
        if self.time.mode == "regex":
            match = re.search(self.time.regex, filename)
            if not match:
                return None
            groups = match.groupdict()
            year = _resolve_year(groups)
            month = int(groups.get("month") or 1)
            day = int(groups.get("day") or 1)
            return pd.Timestamp(year=year, month=month, day=day)
        # coord / none: nothing to extract from the name
        return None


def _resolve_year(groups: Dict[str, Optional[str]]) -> int:
    if groups.get("year"):
        return int(groups["year"])
    if groups.get("year2"):
        # 2-digit year: this lab's data is all 21st century.
        return 2000 + int(groups["year2"])
    raise ValueError("time regex must capture a 'year' or 'year2' group")


class AmbiguousDatasetError(ValueError):
    """Raised when a filename matches more than one dataset spec."""


class DatasetRegistry:
    """A collection of :class:`RasterDatasetSpec` keyed by name."""

    def __init__(self, specs: Optional[Dict[str, RasterDatasetSpec]] = None):
        self._specs: Dict[str, RasterDatasetSpec] = dict(specs or {})

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> List[str]:
        return sorted(self._specs)

    def get(self, name: str) -> RasterDatasetSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"Unknown dataset '{name}'. Registered: {self.names()}"
            )

    def add(self, spec: RasterDatasetSpec, *, overwrite: bool = True) -> None:
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"dataset '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def match(self, path) -> Optional[RasterDatasetSpec]:
        """Return the single spec whose glob matches ``path`` (by basename).

        Returns None if nothing matches. Raises :class:`AmbiguousDatasetError`
        if more than one spec matches, so the caller can ask for an explicit
        dataset name instead of guessing.
        """
        filename = Path(path).name
        hits = [s for s in self._specs.values() if s.matches(filename)]
        if not hits:
            return None
        if len(hits) > 1:
            raise AmbiguousDatasetError(
                f"'{filename}' matches multiple datasets: {[s.name for s in hits]}. "
                "Pass an explicit dataset name to disambiguate."
            )
        return hits[0]

    def describe_file(self, path, dataset: Optional[str] = None) -> Dict[str, object]:
        """Resolve what the registry would extract from ``path``.

        Powers the ``datasets test`` helper. ``dataset`` forces a specific entry;
        otherwise the file is auto-matched.
        """
        filename = Path(path).name
        spec = self.get(dataset) if dataset else self.match(path)
        if spec is None:
            return {"file": filename, "matched": None}
        time_value = spec.time_from_filename(filename)
        time_mode = spec.time.mode if spec.time else "none"
        return {
            "file": filename,
            "matched": spec.name,
            "variable": spec.variable,
            "time": (str(time_value.date()) if time_value is not None else
                     ("<from file coord>" if time_mode == "coord" else None)),
            "crs": spec.crs,
            "units": spec.units,
            "categorical": spec.categorical,
        }


def _parse_specs(payload: dict, source: str) -> List[RasterDatasetSpec]:
    if "datasets" not in payload or not isinstance(payload["datasets"], list):
        raise ValueError(f"{source}: expected a top-level 'datasets' list")
    specs: List[RasterDatasetSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(payload["datasets"]):
        try:
            spec = RasterDatasetSpec(**entry)
        except Exception as exc:  # noqa: BLE001 - prepend context for the user
            raise ValueError(f"{source}: dataset #{i} is invalid: {exc}")
        if spec.name in seen:
            raise ValueError(f"{source}: duplicate dataset name '{spec.name}'")
        seen.add(spec.name)
        specs.append(spec)
    return specs


def _load_packaged_defaults() -> Dict[str, RasterDatasetSpec]:
    try:
        text = (
            resources.files("geoworkflow")
            .joinpath("config", "raster_datasets.json")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    payload = json.loads(text)
    return {s.name: s for s in _parse_specs(payload, "packaged raster_datasets.json")}


def load_dataset_registry(
    user_registry: Optional[Path] = None,
    *,
    use_defaults: bool = True,
) -> DatasetRegistry:
    """Load the dataset registry, merging packaged defaults with user additions.

    Args:
        user_registry: Path to a user JSON. If None, the ``GEOWORKFLOW_REGISTRY``
            environment variable is used when set, else ``data/raster_datasets.json``
            relative to the working directory. Prefer passing an explicit path
            (or setting the env var) in scripts — the CWD-relative default only
            works when running from the workspace root.
        use_defaults: Include the registry shipped inside the package.

    Returns:
        A :class:`DatasetRegistry`. User entries override packaged entries of the
        same name.
    """
    import os

    specs: Dict[str, RasterDatasetSpec] = {}
    if use_defaults:
        specs.update(_load_packaged_defaults())

    if user_registry is not None:
        path = Path(user_registry)
    elif os.environ.get("GEOWORKFLOW_REGISTRY"):
        path = Path(os.environ["GEOWORKFLOW_REGISTRY"])
    else:
        path = DEFAULT_USER_REGISTRY
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for spec in _parse_specs(payload, str(path)):
            specs[spec.name] = spec  # user overrides packaged by name

    return DatasetRegistry(specs)
