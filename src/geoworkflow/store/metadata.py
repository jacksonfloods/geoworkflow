"""Variable metadata + categorical legends for the hex database.

When the per-city NetCDF cubes were retired, the CF attributes they carried
(``long_name``/``units`` and the landcover ``flag_values``/``flag_meanings``)
needed a new home. This module is it: built-in defaults for the standard
variables and the **Copernicus LC100 class legend with official colors**, plus a
loader that merges a warehouse ``metadata.json`` override on top.

The legend is what makes ``Selection.plot()`` colour a categorical map correctly
and exposes the nice LC100 palette to users (``db.legend("landcover")``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

__all__ = ["VARIABLE_METADATA", "LANDCOVER_LEGEND", "Metadata", "load_metadata"]

# variable -> (long_name, units)
VARIABLE_METADATA: Dict[str, Dict[str, Optional[str]]] = {
    "PM25":        {"long_name": "PM2.5 concentration", "units": "ug/m3"},
    "odiac_co2":   {"long_name": "ODIAC fossil-fuel CO2 emissions", "units": "gC/m2/day"},
    "lst_day":     {"long_name": "MODIS daytime land surface temperature", "units": "K"},
    "lst_night":   {"long_name": "MODIS nighttime land surface temperature", "units": "K"},
    "landcover":   {"long_name": "Copernicus CGLS-LC100 discrete land cover", "units": None},
}

# Copernicus Global Land Cover (CGLS-LC100) discrete_classification:
# class code -> (label, official hex colour).
LANDCOVER_LEGEND: Dict[int, Tuple[str, str]] = {
    0:   ("Unknown", "#282828"),
    20:  ("Shrubs", "#FFBB22"),
    30:  ("Herbaceous vegetation", "#FFFF4C"),
    40:  ("Cultivated / cropland", "#F096FF"),
    50:  ("Urban / built-up", "#FA0000"),
    60:  ("Bare / sparse vegetation", "#B4B4B4"),
    70:  ("Snow and ice", "#F0F0F0"),
    80:  ("Permanent water bodies", "#0032C8"),
    90:  ("Herbaceous wetland", "#0096A0"),
    100: ("Moss and lichen", "#FAE6A0"),
    111: ("Closed forest, evergreen needle leaf", "#58481F"),
    112: ("Closed forest, evergreen broad leaf", "#009900"),
    113: ("Closed forest, deciduous needle leaf", "#70663E"),
    114: ("Closed forest, deciduous broad leaf", "#00CC00"),
    115: ("Closed forest, mixed", "#4E751F"),
    116: ("Closed forest, unknown", "#007800"),
    121: ("Open forest, evergreen needle leaf", "#666000"),
    122: ("Open forest, evergreen broad leaf", "#8DB400"),
    123: ("Open forest, deciduous needle leaf", "#8D7400"),
    124: ("Open forest, deciduous broad leaf", "#A0DC00"),
    125: ("Open forest, mixed", "#929900"),
    126: ("Open forest, unknown", "#648C00"),
    200: ("Oceans, seas", "#000080"),
}


class Metadata:
    """Variable long-names/units and categorical legends, with overrides."""

    def __init__(
        self,
        variables: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        legends: Optional[Dict[str, Dict[int, Tuple[str, str]]]] = None,
    ):
        self.variables = {**VARIABLE_METADATA, **(variables or {})}
        self.legends = {"landcover": dict(LANDCOVER_LEGEND), **(legends or {})}

    def long_name(self, variable: str) -> Optional[str]:
        base = variable.rsplit("_", 1)[0] if variable not in self.variables else variable
        meta = self.variables.get(variable) or self.variables.get(base)
        return meta["long_name"] if meta else None

    def units(self, variable: str) -> Optional[str]:
        base = variable.rsplit("_", 1)[0] if variable not in self.variables else variable
        meta = self.variables.get(variable) or self.variables.get(base)
        return meta["units"] if meta else None

    def legend(self, variable: str) -> Optional[Dict[int, Tuple[str, str]]]:
        """``{code: (label, color)}`` for a categorical variable, else None."""
        base = variable.rsplit("_", 1)[0]   # landcover_majority -> landcover
        return self.legends.get(variable) or self.legends.get(base)

    def colors(self, variable: str) -> Optional[Dict[int, str]]:
        leg = self.legend(variable)
        return {code: color for code, (_, color) in leg.items()} if leg else None

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variables": self.variables,
            "legends": {v: {str(k): list(t) for k, t in leg.items()}
                        for v, leg in self.legends.items()},
        }
        path.write_text(json.dumps(payload, indent=2))
        return path


def load_metadata(warehouse_dir: Path) -> Metadata:
    """Load metadata, merging ``<warehouse>/metadata.json`` over the defaults."""
    path = Path(warehouse_dir) / "metadata.json"
    if not path.exists():
        return Metadata()
    payload = json.loads(path.read_text())
    legends = {
        v: {int(k): tuple(t) for k, t in leg.items()}
        for v, leg in payload.get("legends", {}).items()
    }
    return Metadata(variables=payload.get("variables"), legends=legends)
