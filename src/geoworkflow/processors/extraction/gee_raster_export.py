"""
Declarative Google Earth Engine raster downloads (per AOI or per hex grid).

This is the generic counterpart to :class:`SatelliteImageryProcessor` (which is
specialized for Sentinel-2 RGB): one :class:`GEERasterExportConfig` describes a
dataset — its EE asset, bands, temporal cadence, optional QC masking and value
scaling — and the processor downloads it clipped to each target's bounding box,
named so the dataset registry can parse the files, skipping anything already on
disk (re-running resumes).

Example — the two datasets this lab pulls for every city::

    from geoworkflow.processors.extraction import export_gee_rasters

    # Copernicus LC100 land cover, one static image per city
    export_gee_rasters(
        source="COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019",
        bands=["discrete_classification"],
        grid_dir="../data/boundaries/hexagglo",
        output_dir="../data/city", dataset="copernicus_lc100",
        filename_template="{city}_lc100_{time}.tif", static_label="2019",
        scale_m=100, service_account_key="../.keys/<key>.json",
    )

    # MOD11A1 LST day+night, monthly means, QC-masked, scaled to Kelvin
    export_gee_rasters(
        source="MODIS/061/MOD11A1",
        bands=["LST_Day_1km", "LST_Night_1km"],
        band_tags={"LST_Day_1km": "day", "LST_Night_1km": "night"},
        cadence="monthly", start="2019-01", end="2024-12",
        qc_band="QC_Day", scale_factor=0.02,
        grid_dir="../data/boundaries/hexagglo",
        output_dir="../data/city", dataset="mod11a1_lst",
        filename_template="{city}_lst_{band_tag}_{time}.tif",
        scale_m=1000, service_account_key="../.keys/<key>.json",
    )

Note: MOD11A1's night band pairs with QC_Night; per-band QC bands are resolved
automatically for the common ``*_Day_*``/``*_Night_*`` naming, otherwise
``qc_band`` is used as given.
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import ee
    import requests
    HAS_EE = True
except ImportError:  # pragma: no cover - minimal installs
    HAS_EE = False

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:  # pragma: no cover
    HAS_GPD = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.base import ProcessingResult
from geoworkflow.core.exceptions import ProcessingError
from geoworkflow.schemas.config_models import GEERasterExportConfig

logger = logging.getLogger(__name__)

# GeoTIFF magic bytes (little/big endian) — a failed EE download returns
# HTML/JSON, which must not be written to disk as a .tif.
_TIFF_MAGIC = (b"II", b"MM")


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without Earth Engine)
# ---------------------------------------------------------------------------

def iter_periods(cadence: str, start: Optional[str], end: Optional[str]
                 ) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Expand a cadence into ``(time_label, ee_start, ee_end)`` periods.

    monthly: labels ``YYYYMM``, periods [first of month, first of next month).
    yearly:  labels ``YYYY``,   periods [Jan 1, Jan 1 next year).
    static:  one period with no dates (the source is a plain Image).
    """
    if cadence == "static":
        return [(None, None, None)]

    import pandas as pd

    if cadence == "monthly":
        months = pd.period_range(start=start, end=end, freq="M")
        return [
            (f"{p.year}{p.month:02d}",
             f"{p.year}-{p.month:02d}-01",
             f"{(p + 1).year}-{(p + 1).month:02d}-01")
            for p in months
        ]
    if cadence == "yearly":
        years = range(int(str(start)[:4]), int(str(end)[:4]) + 1)
        return [(f"{y}", f"{y}-01-01", f"{y + 1}-01-01") for y in years]
    raise ValueError(f"Unknown cadence: {cadence}")


def city_slug(grid_path: Union[str, Path]) -> str:
    """Derive a city slug from a grid filename: '4858_kinshasa_hex' -> 'kinshasa'."""
    name = Path(grid_path).stem
    if name.endswith("_hex"):
        name = name[: -len("_hex")]
    if name and name[0].isdigit() and "_" in name:
        name = name.split("_", 1)[1]
    return name


def qc_band_for(band: str, qc_band: Optional[str]) -> Optional[str]:
    """Resolve the QC band for a data band.

    MODIS-style products pair Day/Night data bands with Day/Night QC bands; if
    ``band`` contains ``Night`` and ``qc_band`` contains ``Day`` (or vice
    versa), the QC band is switched to match. Otherwise ``qc_band`` as given.
    """
    if qc_band is None:
        return None
    if "Night" in band and "Day" in qc_band:
        return qc_band.replace("Day", "Night")
    if "Day" in band and "Night" in qc_band:
        return qc_band.replace("Night", "Day")
    return qc_band


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class GEERasterExportProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """Download an EE image/collection clipped per AOI or per hex grid."""

    def __init__(
        self,
        config: Union[GEERasterExportConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, GEERasterExportConfig):
            config_dict = config.model_dump(mode="json")
            self.gee_config = config
        else:
            config_dict = config
            self.gee_config = GEERasterExportConfig(**config_dict)

        super().__init__(config_dict, logger)
        self.targets: List[Tuple[str, str, Path]] = []  # (iso3, city, vector path)

    # ------------------------------------------------------------------
    # Template method hooks
    # ------------------------------------------------------------------

    def _get_path_config_keys(self) -> List[str]:
        return []

    def _validate_custom_inputs(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "info": {}}
        cfg = self.gee_config

        if not HAS_EE or not HAS_GPD:
            result["valid"] = False
            result["errors"].append(
                "Requires earthengine-api, requests and geopandas."
            )
            return result

        if cfg.aoi_file is not None and not cfg.aoi_file.exists():
            result["valid"] = False
            result["errors"].append(f"aoi_file does not exist: {cfg.aoi_file}")
        if cfg.grid_dir is not None and not cfg.grid_dir.is_dir():
            result["valid"] = False
            result["errors"].append(f"grid_dir is not a directory: {cfg.grid_dir}")
        if cfg.service_account_key is not None and not cfg.service_account_key.exists():
            result["valid"] = False
            result["errors"].append(
                f"service_account_key does not exist: {cfg.service_account_key}"
            )
        try:
            iter_periods(cfg.cadence, cfg.start, cfg.end)
        except Exception as exc:
            result["valid"] = False
            result["errors"].append(f"Invalid start/end for cadence: {exc}")
        return result

    def _setup_custom_processing(self) -> Dict[str, Any]:
        cfg = self.gee_config
        self.log_processing_step("Authenticating with Earth Engine")
        self._authenticate()

        self.log_processing_step("Discovering targets")
        self.targets = self._discover_targets()
        if not self.targets:
            raise ProcessingError(
                f"No targets found (grid_dir={cfg.grid_dir}, aoi_file={cfg.aoi_file})"
            )
        return {
            "targets": [c for _, c, _ in self.targets],
            "periods": len(iter_periods(cfg.cadence, cfg.start, cfg.end)),
            "bands": list(cfg.bands),
        }

    def _estimate_total_items(self) -> int:
        cfg = self.gee_config
        n_targets = len(self.targets) or 1
        return n_targets * len(iter_periods(cfg.cadence, cfg.start, cfg.end)) * len(cfg.bands)

    def process_data(self) -> ProcessingResult:
        cfg = self.gee_config
        result = ProcessingResult(success=True)
        failures: List[Tuple[str, str]] = []

        periods = iter_periods(cfg.cadence, cfg.start, cfg.end)
        for iso3, city, vector_path in self.targets:
            region = self._region_of(vector_path)
            out_base = cfg.output_dir / iso3 / cfg.dataset

            # Collect this city's still-needed (image, out_path) layers; everything
            # already on disk is counted as skipped (resumable at file granularity).
            specs: List[Tuple["ee.Image", Path]] = []
            for time_label, ee_start, ee_end in periods:
                for band in cfg.bands:
                    out_path = out_base / self._filename(city, iso3, band, time_label)
                    if cfg.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                        result.skipped_count += 1
                        self.update_progress(1, f"{city} (skip)")
                        continue
                    specs.append((self._build_image(band, ee_start, ee_end), out_path))
            if not specs:
                continue

            # One multi-band request per chunk (batch_size=None -> whole city at
            # once), then slice locally into the per-file outputs.
            chunk = cfg.batch_size or len(specs)
            for k in range(0, len(specs), chunk):
                written, chunk_failures = self._download_batch(specs[k:k + chunk], region)
                result.processed_count += len(written)
                result.failed_count += len(chunk_failures)
                for name, err in chunk_failures:
                    failures.append((name, err))
                    self.logger.warning("FAILED %s: %s", name, err)
                self.update_progress(min(chunk, len(specs) - k), f"{city} {cfg.dataset}")

        result.message = (
            f"{cfg.dataset}: downloaded {result.processed_count}, "
            f"skipped {result.skipped_count} existing, failed {result.failed_count} "
            f"across {len(self.targets)} target(s) -> {cfg.output_dir}"
        )
        result.metadata = {
            "dataset": cfg.dataset,
            "source": cfg.source,
            "targets": [c for _, c, _ in self.targets],
            "failures": failures,
        }
        if failures and result.processed_count == 0 and result.skipped_count == 0:
            result.success = False
        return result

    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        self.targets = []
        return {"targets_cleared": True}

    # ------------------------------------------------------------------
    # Earth Engine specifics
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        from geoworkflow.utils.earth_engine_utils import EarthEngineAuth

        cfg = self.gee_config
        if cfg.skip_auth:
            # Caller initialized Earth Engine already (e.g. once before a thread
            # pool of per-city downloads); reuse that process-global session.
            return
        email = cfg.service_account_email
        project = cfg.project_id
        # The SA key JSON already carries the email and project; read them so
        # callers only need the key path.
        if cfg.service_account_key is not None:
            payload = json.loads(cfg.service_account_key.read_text())
            email = email or payload.get("client_email")
            project = project or payload.get("project_id")
        EarthEngineAuth.authenticate(
            service_account_key=cfg.service_account_key,
            service_account_email=email,
            project_id=project,
        )

    def _discover_targets(self) -> List[Tuple[str, str, Path]]:
        cfg = self.gee_config
        if cfg.aoi_file is not None:
            iso3 = cfg.iso3 or "AOI"
            city = cfg.aoi_city or city_slug(cfg.aoi_file)
            return [(iso3, city, cfg.aoi_file)]
        grids = sorted(cfg.grid_dir.glob(cfg.grid_pattern))
        return [(p.parent.name, city_slug(p), p) for p in grids]

    def _region_of(self, vector_path: Path) -> "ee.Geometry":
        gdf = gpd.read_file(vector_path).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        return ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])

    def _filename(self, city: str, iso3: str, band: str, time_label: Optional[str]) -> str:
        cfg = self.gee_config
        return cfg.filename_template.format(
            city=city, iso3=iso3, dataset=cfg.dataset,
            band_tag=cfg.band_tags.get(band, band.lower()),
            time=time_label if time_label is not None else cfg.static_label,
        )

    def _build_image(self, band: str, ee_start: Optional[str], ee_end: Optional[str]) -> "ee.Image":
        cfg = self.gee_config
        if cfg.cadence == "static":
            image = ee.Image(cfg.source).select(band)
        else:
            col = ee.ImageCollection(cfg.source).filterDate(ee_start, ee_end)
            qc = qc_band_for(band, cfg.qc_band)
            if qc is not None:
                col = col.map(
                    lambda im: im.select(band).updateMask(
                        im.select(qc).bitwiseAnd(cfg.qc_bit_mask).lte(cfg.qc_max)
                    )
                )
            else:
                col = col.select(band)
            image = getattr(col, cfg.composite)()
        if cfg.scale_factor is not None:
            image = image.multiply(cfg.scale_factor)
        return image

    def _fetch_geotiff(self, image: "ee.Image", region: "ee.Geometry") -> bytes:
        """getDownloadURL + GET one image as GeoTIFF bytes, with retry/backoff.

        Validates the GeoTIFF magic so an EE error payload (HTML/JSON) is raised,
        not silently written. Returns the raw bytes (single- or multi-band).
        """
        cfg = self.gee_config
        last_error: Optional[Exception] = None
        for attempt in range(cfg.retries):
            try:
                url = image.getDownloadURL({
                    "region": region,
                    "scale": cfg.scale_m,
                    "crs": cfg.output_crs,
                    "format": "GEO_TIFF",
                })
                resp = requests.get(url, timeout=cfg.timeout_s)
                resp.raise_for_status()
                if resp.content[:2] not in _TIFF_MAGIC:
                    raise ValueError("response is not a GeoTIFF (EE error payload?)")
                return resp.content
            except Exception as exc:  # noqa: BLE001 - retry with backoff
                last_error = exc
                if attempt < cfg.retries - 1:
                    _time.sleep(3 * (attempt + 1))
        raise ProcessingError(f"download failed after {cfg.retries} attempts: {last_error}")

    def _download(self, image: "ee.Image", region: "ee.Geometry", out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self._fetch_geotiff(image, region))

    def _download_batch(
        self, specs: List[Tuple["ee.Image", Path]], region: "ee.Geometry",
    ) -> Tuple[List[Path], List[Tuple[str, str]]]:
        """Download many single-band layers as ONE multi-band GeoTIFF, then split
        it into the individual per-file outputs — one HTTP request instead of N.

        Bands are renamed to a positional ``b{i}`` so names never collide, and EE
        preserves band order, so output band *i* maps to ``specs[i]``'s path. If a
        multi-band request fails (e.g. exceeds EE's request-size limit) the batch
        is bisected and retried, down to single layers — so an oversized city
        degrades to more requests rather than failing. Returns
        ``(written_paths, [(filename, error), ...])``.
        """
        written: List[Path] = []
        failures: List[Tuple[str, str]] = []
        self._download_batch_rec(specs, region, written, failures)
        return written, failures

    def _download_batch_rec(self, specs, region, written, failures) -> None:
        if not specs:
            return
        images = [img.rename(f"b{i}") for i, (img, _) in enumerate(specs)]
        out_paths = [op for _, op in specs]
        combined = ee.Image.cat(images) if len(images) > 1 else images[0]
        try:
            self._slice_to_files(self._fetch_geotiff(combined, region), out_paths)
            written.extend(out_paths)
        except Exception as exc:  # noqa: BLE001
            if len(specs) == 1:
                failures.append((out_paths[0].name, str(exc)[:200]))
                return
            mid = len(specs) // 2  # bisect on failure (handles size-limit errors)
            self._download_batch_rec(specs[:mid], region, written, failures)
            self._download_batch_rec(specs[mid:], region, written, failures)

    def _slice_to_files(self, content: bytes, out_paths: List[Path]) -> None:
        """Write each band of a multi-band GeoTIFF (in ``content``) to its own
        single-band file, preserving CRS/transform/dtype — same on-disk product
        as the per-request path, so the registry and Pipeline 1 are unaffected."""
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(content) as mem, mem.open() as src:
            if src.count != len(out_paths):
                raise ProcessingError(
                    f"expected {len(out_paths)} bands, got {src.count}")
            profile = src.profile.copy()
            profile.update(count=1)
            for i, out_path in enumerate(out_paths, start=1):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(src.read(i), 1)


def export_gee_rasters(**kwargs: Any) -> ProcessingResult:
    """Convenience wrapper: build a :class:`GEERasterExportConfig` and run."""
    config = GEERasterExportConfig(**{
        k: (Path(v) if k in ("aoi_file", "grid_dir", "output_dir", "service_account_key")
            and v is not None else v)
        for k, v in kwargs.items()
    })
    return GEERasterExportProcessor(config).process()
