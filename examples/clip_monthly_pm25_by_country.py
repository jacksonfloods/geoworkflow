"""
Clip every monthly 2023 global PM2.5 GeoTIFF to one raster per African country.

The input folder contains files named like:
V6GL02.04.CNNPM25.GL.202301-202301.tif

For each monthly raster, this script creates an output directory named by month,
for example:
/home/sjs96_file_share/data/country/pm25/202301

Inside each month folder, it writes one country-specific GeoTIFF per country.
"""

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


# Folder containing the monthly global PM2.5 rasters.
RASTER_DIR = Path("/home/sjs96_file_share/data/global/PM25/2023")

# Input African country boundaries. This is a GeoPackage even though the
# workflow is the same for a shapefile.
BOUNDARY_PATH = Path("/home/sjs96_file_share/data/boundaries/africa_boundaries.gpkg")

# Parent output folder. Each monthly folder, such as 202301, is created here.
OUTPUT_ROOT = Path("/home/sjs96_file_share/data/country/pm25")

# Match the month from filenames such as:
# V6GL02.04.CNNPM25.GL.202301-202301.tif
MONTH_PATTERN = re.compile(r"(?P<month>20\d{4})-20\d{4}\.tif$", re.IGNORECASE)

# The expected country-name field. If it is missing, the script checks common
# alternatives and reports which one it uses.
PREFERRED_NAME_FIELD = "NAME"
NAME_FIELD_CANDIDATES = ("NAME", "NAME_0", "ADMIN", "COUNTRY", "country", "name")


def safe_filename(value: object) -> str:
    """Convert a country name to a filesystem-safe filename stem."""
    name = str(value).strip()
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._ ")
    return name or "Unknown_Country"


def choose_name_field(columns) -> str:
    """Find the best available country-name field in the boundary data."""
    for field in NAME_FIELD_CANDIDATES:
        if field in columns:
            if field != PREFERRED_NAME_FIELD:
                print(
                    f'Field "{PREFERRED_NAME_FIELD}" was not found. '
                    f'Using "{field}" instead.'
                )
            return field

    available = ", ".join(columns)
    raise ValueError(
        "Could not find a country-name field. "
        f"Checked {NAME_FIELD_CANDIDATES}. Available fields: {available}"
    )


def repair_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid polygons before masking."""
    invalid = ~gdf.geometry.is_valid
    if not invalid.any():
        return gdf

    print(f"Repairing {invalid.sum()} invalid geometries.")
    gdf = gdf.copy()

    try:
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
    except AttributeError:
        # Fallback for older GeoPandas/Shapely versions.
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    return gdf


def default_nodata_for_dtype(dtype: str):
    """Choose a nodata value if the source raster does not define one."""
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.floating):
        return -9999.0
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return np.iinfo(np_dtype).max
    if np.issubdtype(np_dtype, np.signedinteger):
        return np.iinfo(np_dtype).min

    raise ValueError(f"Unsupported raster dtype for nodata selection: {dtype}")


def month_from_raster_path(raster_path: Path) -> str:
    """Extract YYYYMM from a PM2.5 monthly raster filename."""
    match = MONTH_PATTERN.search(raster_path.name)
    if not match:
        raise ValueError(f"Could not extract month from raster name: {raster_path.name}")
    return match.group("month")


def find_monthly_rasters(raster_dir: Path) -> list[Path]:
    """Return the monthly input TIFFs, excluding already-clipped outputs."""
    rasters = []

    for raster_path in sorted(raster_dir.glob("*.tif")):
        if "_clipped" in raster_path.stem.lower():
            continue
        if MONTH_PATTERN.search(raster_path.name):
            rasters.append(raster_path)

    if not rasters:
        raise FileNotFoundError(f"No monthly PM2.5 TIFFs found in {raster_dir}")

    return rasters


def clip_raster_by_country(
    raster_path: Path,
    countries: gpd.GeoDataFrame,
    name_field: str,
) -> None:
    """Clip one monthly raster by country and write outputs to its month folder."""
    month = month_from_raster_path(raster_path)
    output_dir = OUTPUT_ROOT / month
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {raster_path.name} -> {output_dir}")

    with rasterio.open(raster_path) as src:
        countries_for_raster = countries
        if countries_for_raster.crs != src.crs:
            countries_for_raster = countries_for_raster.to_crs(src.crs)

        nodata = src.nodata
        if nodata is None:
            nodata = default_nodata_for_dtype(src.dtypes[0])

        used_filenames = set()

        for index, row in countries_for_raster.iterrows():
            country_name = row[name_field]
            filename_stem = safe_filename(country_name)

            # Include the month in the file name so files remain clear if moved.
            output_name = f"{filename_stem}_PM25_{month}.tif"
            if output_name in used_filenames:
                output_name = f"{filename_stem}_{index}_PM25_{month}.tif"
            used_filenames.add(output_name)

            output_path = output_dir / output_name

            try:
                clipped_data, clipped_transform = mask(
                    src,
                    [mapping(row.geometry)],
                    crop=True,
                    nodata=nodata,
                    filled=True,
                )
            except ValueError as exc:
                # Rasterio raises ValueError when a geometry does not overlap.
                print(f"Skipping {country_name}: {exc}")
                continue

            output_profile = src.profile.copy()
            output_profile.update(
                {
                    "height": clipped_data.shape[1],
                    "width": clipped_data.shape[2],
                    "transform": clipped_transform,
                    "crs": src.crs,
                    "nodata": nodata,
                }
            )

            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(clipped_data)

            print(f"Saved {output_path}")


def main() -> None:
    """Load boundaries once, then clip every monthly PM2.5 raster by country."""
    monthly_rasters = find_monthly_rasters(RASTER_DIR)

    countries = gpd.read_file(BOUNDARY_PATH)
    name_field = choose_name_field(countries.columns)

    countries = countries[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    countries = repair_invalid_geometries(countries)

    for raster_path in monthly_rasters:
        clip_raster_by_country(raster_path, countries, name_field)


if __name__ == "__main__":
    main()
