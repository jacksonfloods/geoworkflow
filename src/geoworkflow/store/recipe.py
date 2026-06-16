"""Recipe helpers — the default build spec and a starter template.

The default recipe reproduces the database's current composition (PM2.5, ODIAC
CO2, MODIS LST day/night, Copernicus LC100), so the one-time reuse migration and
a from-scratch rebuild produce the same columns. It is also what
`geoworkflow hexdb build` writes as a starter template for users to edit.
"""

from __future__ import annotations

from pathlib import Path

from geoworkflow.schemas.config_models import (
    HexDBRecipe, GridSpec, TimeRange, CitySelector, MetricSpec, MetricSource,
)


def default_recipe() -> HexDBRecipe:
    """The recipe matching the current database (4 datasets, 250 m UTM grids)."""
    return HexDBRecipe(
        grid=GridSpec(side_length=250.0, crs_mode="utm_local"),
        cities=CitySelector(all=True),
        time=TimeRange(start="2019-01", end="2024-12"),
        metrics=[
            MetricSpec(name="PM25", scope="global", statistics=["weighted_mean"]),
            MetricSpec(name="odiac", scope="global", statistics=["weighted_mean"]),
            MetricSpec(
                name="mod11a1_lst", scope="city", statistics=["weighted_mean"],
                source=MetricSource(
                    ee_asset="MODIS/061/MOD11A1",
                    bands=["LST_Day_1km", "LST_Night_1km"],
                    band_tags={"LST_Day_1km": "day", "LST_Night_1km": "night"},
                    cadence="monthly", qc_band="QC_Day", scale_factor=0.02,
                    scale_m=1000.0, filename_template="{city}_lst_{band_tag}_{time}.tif",
                ),
            ),
            MetricSpec(
                name="copernicus_lc100", scope="city",
                statistics=["majority", "variety"], annual=True,
                source=MetricSource(
                    ee_asset="COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019",
                    bands=["discrete_classification"], cadence="static",
                    scale_m=100.0, static_label="2019",
                    filename_template="{city}_lc100_{time}.tif",
                ),
            ),
        ],
    )


def write_template(path: Path) -> Path:
    """Write the default recipe to ``path`` as a starter YAML template."""
    return default_recipe().to_yaml(path)
