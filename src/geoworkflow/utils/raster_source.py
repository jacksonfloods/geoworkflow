"""
Raster source layer: normalize heterogeneous raster inputs into a stream of
clipped, single-band slices the zonal engine can consume.

The zonal engine (:func:`geoworkflow.utils.zonal_utils.compute_zonal_statistics`)
summarizes **one single-band raster** at a time. Real inputs are messier:

* GeoTIFFs, static or monthly (time encoded in the filename),
* netCDFs with a data variable that may be 2-D ``(lat, lon)`` or carry a time
  axis ``(time, lat, lon)``,
* and very large global grids (e.g. the 13000x36000 PM2.5 netCDF) that must be
  windowed to the AOI before any array is materialized.

:func:`open_raster_slices` handles all of that and yields :class:`RasterSlice`
objects tagged with ``(variable, time)`` resolved via the dataset registry
(:mod:`geoworkflow.core.dataset_registry`). It does **not** compute statistics or
write outputs — that is the grid-statistics processor's job.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr
    from rasterio.io import MemoryFile
    from rasterio.windows import Window, from_bounds
    from rasterio.errors import WindowError
    HAS_RASTER_SOURCE_LIBS = True
except ImportError:  # pragma: no cover - minimal installs
    HAS_RASTER_SOURCE_LIBS = False

from geoworkflow.core.dataset_registry import (
    AmbiguousDatasetError,
    DatasetRegistry,
    RasterDatasetSpec,
    load_dataset_registry,
)

logger = logging.getLogger(__name__)

_NETCDF_EXTS = {".nc", ".nc4"}
_GEOTIFF_EXTS = {".tif", ".tiff", ".geotiff", ".geotif"}
_RASTER_EXTS = _NETCDF_EXTS | _GEOTIFF_EXTS

AOIInput = Union["gpd.GeoDataFrame", str, Path, None]


@dataclass
class RasterSlice:
    """One single-band, AOI-clipped raster surface with its metadata.

    The pixel data is materialized (the clip is AOI-sized, hence small). Use
    :meth:`open` to get a transient single-band rasterio dataset for the engine.
    """

    array: np.ndarray
    transform: "rasterio.Affine"
    crs: Optional[str]
    variable: str
    time: Optional[pd.Timestamp] = None
    source_path: Optional[Path] = None
    band: int = 1
    nodata: Optional[float] = None
    units: Optional[str] = None
    categorical: bool = False
    dataset_name: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def is_empty(self) -> bool:
        return self.array.size == 0

    @contextmanager
    def open(self):
        """Yield a transient single-band rasterio dataset backed by a MemoryFile."""
        profile = {
            "driver": "GTiff",
            "height": int(self.array.shape[0]),
            "width": int(self.array.shape[1]),
            "count": 1,
            "dtype": self.array.dtype,
            "crs": self.crs,
            "transform": self.transform,
        }
        if self.nodata is not None:
            profile["nodata"] = self.nodata
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(self.array, 1)
            with memfile.open() as src:
                yield src


# ---------------------------------------------------------------------------
# Input enumeration and AOI handling
# ---------------------------------------------------------------------------

def _enumerate_inputs(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    recursive: bool,
) -> List[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            globber = p.rglob if recursive else p.glob
            files.extend(
                f for f in globber("*")
                if f.is_file() and f.suffix.lower() in _RASTER_EXTS
            )
        elif p.is_file():
            files.append(p)
        else:
            logger.warning("Input not found, skipping: %s", p)
    return sorted(set(files))


def _load_aoi(aoi: AOIInput) -> Optional["gpd.GeoDataFrame"]:
    if aoi is None:
        return None
    if isinstance(aoi, (str, Path)):
        return gpd.read_file(aoi)
    return aoi  # assume GeoDataFrame


def _aoi_bounds_in_crs(aoi_gdf: "gpd.GeoDataFrame", crs) -> tuple:
    if aoi_gdf.crs is None:
        warnings.warn("AOI has no CRS; assuming it matches the raster CRS.")
        projected = aoi_gdf
    elif crs is not None and str(aoi_gdf.crs) != str(crs):
        projected = aoi_gdf.to_crs(crs)
    else:
        projected = aoi_gdf
    minx, miny, maxx, maxy = projected.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


# ---------------------------------------------------------------------------
# GeoTIFF and netCDF slicing
# ---------------------------------------------------------------------------

def _geotiff_slices(
    path: Path,
    spec: Optional[RasterDatasetSpec],
    aoi_gdf: Optional["gpd.GeoDataFrame"],
    default_crs: str,
) -> List[RasterSlice]:
    with rasterio.open(path) as src:
        # A registry `crs` is authoritative: it overrides a file's (possibly
        # mislabeled) CRS tag, then falls back to the file's CRS, then default_crs.
        crs = (spec.crs if spec and spec.crs else None) or src.crs or default_crs
        nodata = src.nodata if src.nodata is not None else (spec.nodata if spec else None)

        if aoi_gdf is not None:
            bounds = _aoi_bounds_in_crs(aoi_gdf, crs)
            window = from_bounds(*bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            try:
                window = window.intersection(Window(0, 0, src.width, src.height))
            except WindowError:
                logger.warning("AOI does not overlap %s; skipping.", path.name)
                return []
            if window.width <= 0 or window.height <= 0:
                logger.warning("AOI does not overlap %s; skipping.", path.name)
                return []
            array = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            array = src.read(1)
            transform = src.transform

    variable = (spec.variable if spec and spec.variable else path.stem)
    time = spec.time_from_filename(path.name) if spec else None
    units = spec.units if spec else None
    categorical = spec.categorical if spec else False
    return [
        RasterSlice(
            array=array, transform=transform, crs=str(crs) if crs else None,
            variable=variable, time=time, source_path=path, band=1,
            nodata=nodata, units=units, categorical=categorical,
            dataset_name=spec.name if spec else None,
        )
    ]


def _spatial_dim_names(da: "xr.DataArray") -> tuple:
    xname = next((d for d in da.dims if str(d).lower() in ("x", "lon", "longitude")), None)
    yname = next((d for d in da.dims if str(d).lower() in ("y", "lat", "latitude")), None)
    return xname, yname


def _netcdf_slices(
    path: Path,
    spec: Optional[RasterDatasetSpec],
    aoi_gdf: Optional["gpd.GeoDataFrame"],
    default_crs: str,
) -> List[RasterSlice]:
    slices: List[RasterSlice] = []
    with xr.open_dataset(path) as ds:
        # Which variables to extract.
        if spec and spec.variable and spec.variable in ds.data_vars:
            var_names = [spec.variable]
        elif spec and spec.variable:
            logger.warning(
                "Variable '%s' not in %s; falling back to all data variables.",
                spec.variable, path.name,
            )
            var_names = list(ds.data_vars)
        else:
            var_names = list(ds.data_vars)

        for var in var_names:
            da = ds[var]
            xname, yname = _spatial_dim_names(da)
            if xname is None or yname is None:
                logger.warning("No lon/lat dims found for %s in %s; skipping.",
                               var, path.name)
                continue

            # Normalize to north-up BEFORE any rioxarray ops. Many netCDFs (e.g.
            # the PM2.5 grid) store latitude ascending, which yields a positive
            # pixel-height (south-up) transform that GeoTIFF/exactextract
            # mishandle (flipped extents, all-NaN results). sortby is a plain
            # xarray op; doing it here avoids dropping rio's spatial-dim metadata.
            yvals = np.asarray(da[yname].values)
            if yvals.ndim == 1 and yvals.size > 1 and yvals[0] < yvals[-1]:
                da = da.sortby(yname, ascending=False)

            da = da.rio.set_spatial_dims(x_dim=xname, y_dim=yname, inplace=False)
            # A registry `crs` is authoritative (overrides the file's tag), then the
            # file's own CRS, then default_crs.
            file_crs = da.rio.crs
            crs = (spec.crs if spec and spec.crs else None) or (
                str(file_crs) if file_crs else default_crs
            )
            da = da.rio.write_crs(crs)

            if aoi_gdf is not None:
                minx, miny, maxx, maxy = _aoi_bounds_in_crs(aoi_gdf, crs)
                try:
                    da = da.rio.clip_box(minx, miny, maxx, maxy)
                except Exception as exc:  # NoDataInBounds and friends
                    logger.warning("AOI does not overlap %s/%s (%s); skipping.",
                                   path.name, var, exc)
                    continue

            slices.extend(
                _slices_from_dataarray(da, var, xname, yname, spec, path, crs)
            )
    return slices


def _slices_from_dataarray(da, var, xname, yname, spec, path, crs) -> List[RasterSlice]:
    spatial = {xname, yname}
    extra_dims = [d for d in da.dims if d not in spatial]

    units = (spec.units if spec and spec.units else da.attrs.get("units"))
    categorical = spec.categorical if spec else False
    nodata = da.rio.nodata
    if nodata is None and spec is not None:
        nodata = spec.nodata

    def _make(slice2d, time_value) -> RasterSlice:
        arr = np.asarray(slice2d.values)
        return RasterSlice(
            array=arr, transform=slice2d.rio.transform(), crs=str(crs) if crs else None,
            variable=var, time=time_value, source_path=path, band=1,
            nodata=float(nodata) if nodata is not None else None,
            units=units, categorical=categorical,
            dataset_name=spec.name if spec else None,
        )

    # No extra (temporal) dimension: time comes from the filename via the spec.
    if not extra_dims:
        time_value = spec.time_from_filename(path.name) if spec else None
        return [_make(da, time_value)]

    # A temporal dimension (prefer a real coordinate). Handle the first extra dim.
    time_dim = next((d for d in extra_dims if str(d).lower() in ("time", "t")), extra_dims[0])
    if len(extra_dims) > 1:
        logger.warning(
            "%s/%s has multiple non-spatial dims %s; iterating '%s', taking index 0 of the rest.",
            path.name, var, extra_dims, time_dim,
        )
        for d in extra_dims:
            if d != time_dim:
                da = da.isel({d: 0})

    out: List[RasterSlice] = []
    has_coord = time_dim in da.coords
    for i in range(da.sizes[time_dim]):
        slice2d = da.isel({time_dim: i})
        if has_coord:
            time_value = pd.Timestamp(np.asarray(da[time_dim].values)[i])
        else:
            time_value = spec.time_from_filename(path.name) if spec else None
        out.append(_make(slice2d, time_value))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def open_raster_slices(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    aoi: AOIInput = None,
    registry: Optional[DatasetRegistry] = None,
    dataset: Optional[str] = None,
    default_crs: str = "EPSG:4326",
    recursive: bool = True,
) -> Iterator[RasterSlice]:
    """Yield AOI-clipped single-band :class:`RasterSlice`s from raster inputs.

    Args:
        inputs: A file, directory, or list of either. Directories are searched
            for ``.tif/.tiff/.nc/.nc4`` (recursively unless ``recursive=False``).
        aoi: Vector AOI (GeoDataFrame or path) used to window/clip each raster
            *before* reading pixels — essential for large global grids. ``None``
            reads the full raster (logged as a warning).
        registry: Dataset registry for resolving variable/time/CRS. Defaults to
            :func:`geoworkflow.core.dataset_registry.load_dataset_registry`.
        dataset: Force a specific registry entry for every input (skip matching).
        default_crs: CRS to assume when a file declares none (e.g. PM2.5 netCDF).
        recursive: Recurse into subdirectories when an input is a directory.

    Yields:
        :class:`RasterSlice` objects, one per (file, variable, timestep).
    """
    if not HAS_RASTER_SOURCE_LIBS:
        raise ImportError(
            "The raster source layer requires rasterio, rioxarray, xarray and "
            "geopandas. Install the geoworkflow conda environment."
        )

    if registry is None:
        registry = load_dataset_registry()
    aoi_gdf = _load_aoi(aoi)
    if aoi_gdf is None:
        logger.warning("No AOI provided; reading full rasters (may be large).")

    files = _enumerate_inputs(inputs, recursive)
    if not files:
        logger.warning("No raster files found in inputs: %s", inputs)

    for path in files:
        spec = _resolve_spec(registry, path, dataset)
        ext = path.suffix.lower()
        if ext in _NETCDF_EXTS:
            file_slices = _netcdf_slices(path, spec, aoi_gdf, default_crs)
        elif ext in _GEOTIFF_EXTS:
            file_slices = _geotiff_slices(path, spec, aoi_gdf, default_crs)
        else:
            logger.warning("Unsupported raster extension %s; skipping %s", ext, path.name)
            continue
        for sl in file_slices:
            yield sl


def _resolve_spec(
    registry: DatasetRegistry, path: Path, dataset: Optional[str]
) -> Optional[RasterDatasetSpec]:
    if dataset is not None:
        return registry.get(dataset)
    try:
        spec = registry.match(path)
    except AmbiguousDatasetError as exc:
        raise AmbiguousDatasetError(
            f"{exc} (file: {path}). Pass dataset=... to open_raster_slices."
        )
    if spec is None:
        logger.info(
            "No dataset registry entry matched %s; using filename stem as the "
            "variable and no time. Add an entry to data/raster_datasets.json to "
            "control this.", path.name,
        )
    return spec
