"""
Categorical raster legends (class code -> name).

Used by the NetCDF builder to emit CF ``flag_values`` / ``flag_meanings`` so a
cube's categorical variables (e.g. land cover) are self-describing to anyone
opening the file, with or without geoworkflow.
"""

from typing import Dict

# Copernicus Global Land Service — Land Cover 100 m (CGLS-LC100), Collection 3,
# ``discrete_classification`` band. Standard legend.
COPERNICUS_LC100: Dict[int, str] = {
    0: "unknown",
    20: "shrubs",
    30: "herbaceous_vegetation",
    40: "cropland",
    50: "urban_built_up",
    60: "bare_sparse_vegetation",
    70: "snow_and_ice",
    80: "permanent_water_bodies",
    90: "herbaceous_wetland",
    100: "moss_and_lichen",
    111: "closed_forest_evergreen_needle_leaf",
    112: "closed_forest_evergreen_broad_leaf",
    113: "closed_forest_deciduous_needle_leaf",
    114: "closed_forest_deciduous_broad_leaf",
    115: "closed_forest_mixed",
    116: "closed_forest_unknown",
    121: "open_forest_evergreen_needle_leaf",
    122: "open_forest_evergreen_broad_leaf",
    123: "open_forest_deciduous_needle_leaf",
    124: "open_forest_deciduous_broad_leaf",
    125: "open_forest_mixed",
    126: "open_forest_unknown",
    200: "oceans_seas",
}

# Default legend lookup by tidy-table variable name. The NetCDF builder falls
# back to these when no legend is supplied, so land-cover cubes are annotated
# automatically.
DEFAULT_LEGENDS: Dict[str, Dict[int, str]] = {
    "landcover": COPERNICUS_LC100,
}
