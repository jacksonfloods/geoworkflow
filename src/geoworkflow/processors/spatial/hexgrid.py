"""
Hexagonal grid processor for geoworkflow.

Generates a flat-top hexagonal grid over any AOI vector file and saves the result
as a GeoJSON. Grid cells are aligned to a global origin so grids generated for
different AOIs tile seamlessly.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json
import logging
import math

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from geoworkflow.core.enhanced_base import TemplateMethodProcessor, GeospatialProcessorMixin
from geoworkflow.core.base import ProcessingResult
from geoworkflow.core.exceptions import ProcessingError
from geoworkflow.schemas.config_models import HexGridConfig

# Africa Albers Equal Area Conic — matches the metric CRS that the default
# grid_origin_x/y values (-3_000_000, -2_000_000) are expressed in.
_WORKING_CRS = "ESRI:102022"


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _axial_round(q_float: float, r_float: float) -> Tuple[int, int]:
    s_float = -q_float - r_float
    q = round(q_float)
    r = round(r_float)
    s = round(s_float)
    q_diff = abs(q - q_float)
    r_diff = abs(r - r_float)
    s_diff = abs(s - s_float)
    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    return int(q), int(r)


def _axial_to_cartesian(
    q: int, r: int, origin_x: float, origin_y: float, side_length: float
) -> Tuple[float, float]:
    x = origin_x + side_length * math.sqrt(3) * (q + r / 2)
    y = origin_y + side_length * 1.5 * r
    return x, y


def _cartesian_to_axial(
    x: float, y: float, origin_x: float, origin_y: float, side_length: float
) -> Tuple[int, int]:
    rel_x = x - origin_x
    rel_y = y - origin_y
    q_float = (math.sqrt(3) / 3 * rel_x - 1 / 3 * rel_y) / side_length
    r_float = (2 / 3 * rel_y) / side_length
    return _axial_round(q_float, r_float)


def _make_hex_polygon(center_x: float, center_y: float, side_length: float) -> Polygon:
    vertices = [
        (
            center_x + side_length * math.cos(math.pi / 3 * i + math.pi / 6),
            center_y + side_length * math.sin(math.pi / 3 * i + math.pi / 6),
        )
        for i in range(6)
    ]
    vertices.append(vertices[0])
    return Polygon(vertices)


def _hex_id(q: int, r: int, system: str = "SSA") -> str:
    """Positional GridID: ``<system>_HQ{q}_R{r}``.

    ``system`` is the lattice namespace — ``SSA`` for the global Albers lattice,
    ``UTM<epsg>`` (e.g. ``UTM32628``) for a per-zone UTM lattice. The id is a pure
    function of (q, r), so the same physical hexagon always gets the same id.
    """
    return f"{system}_HQ{q:+07d}_R{r:+07d}"


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class HexGridProcessor(TemplateMethodProcessor, GeospatialProcessorMixin):
    """
    Generates a flat-top hexagonal grid clipped to an AOI and saves it as GeoJSON.

    Grid cells are aligned to a global origin (default: Sub-Saharan Africa in
    ESRI:102022) so grids from different AOIs tile without gaps or overlaps.
    """

    def __init__(
        self,
        config: Union[HexGridConfig, Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, HexGridConfig):
            config_dict = config.model_dump(mode="json")
            self.hexgrid_config = config
        else:
            config_dict = config
            self.hexgrid_config = HexGridConfig(**config_dict)

        super().__init__(config_dict, logger)
        self.aoi_gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # Template method hooks
    # ------------------------------------------------------------------

    def _get_path_config_keys(self) -> List[str]:
        return ["aoi_file"]

    def _estimate_total_items(self) -> int:
        return 0

    def _validate_custom_inputs(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "info": {}}

        if not HAS_GEOSPATIAL_LIBS:
            result["errors"].append(
                "geopandas, shapely, and numpy are required. "
                "Install with: pip install geopandas shapely numpy"
            )
            result["valid"] = False
            return result

        if not self.hexgrid_config.aoi_file.exists():
            result["errors"].append(
                f"AOI file does not exist: {self.hexgrid_config.aoi_file}"
            )
            result["valid"] = False

        try:
            self.hexgrid_config.output_file.parent.mkdir(parents=True, exist_ok=True)
            result["info"]["output_dir"] = str(self.hexgrid_config.output_file.parent)
        except Exception as exc:
            result["errors"].append(f"Cannot create output directory: {exc}")
            result["valid"] = False

        return result

    def _setup_custom_processing(self) -> Dict[str, Any]:
        setup_info: Dict[str, Any] = {}
        geo_info = self.setup_geospatial_processing()
        setup_info["geospatial"] = geo_info

        self.log_processing_step("Loading AOI")
        try:
            self.aoi_gdf = gpd.read_file(self.hexgrid_config.aoi_file)
            if self.aoi_gdf.crs is None:
                self.logger.warning("AOI has no CRS — assuming EPSG:4326")
                self.aoi_gdf.set_crs("EPSG:4326", inplace=True)
            setup_info["aoi_features"] = len(self.aoi_gdf)
            setup_info["aoi_crs"] = str(self.aoi_gdf.crs)
        except Exception as exc:
            raise ProcessingError(f"Failed to load AOI: {exc}")

        return setup_info

    def process_data(self) -> ProcessingResult:
        result = ProcessingResult(success=True)

        try:
            if (
                self.hexgrid_config.skip_existing
                and self.hexgrid_config.output_file.exists()
            ):
                result.message = (
                    f"Skipping — output already exists: {self.hexgrid_config.output_file}"
                )
                result.skipped_count = 1
                result.add_output_path(self.hexgrid_config.output_file)
                return result

            cfg = self.hexgrid_config
            # Pick the lattice CRS by mode (the origin is shared in both).
            if cfg.crs_mode == "utm_local":
                if cfg.utm_epsg is not None:
                    working_crs = f"EPSG:{cfg.utm_epsg}"          # pinned zone
                else:
                    working_crs = self.aoi_gdf.estimate_utm_crs()  # library, not DIY
                aoi_projected = self.aoi_gdf.to_crs(working_crs)
                epsg = aoi_projected.crs.to_epsg()
                system = f"UTM{epsg}"
            else:
                working_crs = _WORKING_CRS
                aoi_projected = self.aoi_gdf.to_crs(working_crs)
                epsg = None
                system = "SSA"
            self.log_processing_step(f"Working CRS {working_crs} (GridID system {system})")
            self.update_progress(1, "AOI reprojected")

            self.log_processing_step("Generating hex grid")
            hex_gdf = self._generate_hex_grid(aoi_projected, working_crs, system)
            # True (equal-area) area per hex, computed in ESRI:102022 regardless of the
            # storage CRS -> correct downstream densities (a 4326 grid's .area would be
            # in square degrees).
            hex_gdf["area_m2"] = hex_gdf.to_crs(_WORKING_CRS).geometry.area
            hex_gdf = hex_gdf[["GridID", "q", "r", "area_m2", "geometry"]]
            self.add_metric("hexagons_generated", len(hex_gdf))
            self.update_progress(1, f"Generated {len(hex_gdf)} hexagons")

            self.log_processing_step(f"Reprojecting to {cfg.output_crs}")
            hex_gdf = hex_gdf.to_crs(cfg.output_crs)
            self.update_progress(1, "Reprojected output")

            self.log_processing_step(f"Saving to {cfg.output_file}")
            hex_gdf.to_file(cfg.output_file, driver="GeoJSON")
            # Provenance sidecar: the stored grid is in output_crs (e.g. 4326), which
            # hides how it was built. Record the lattice spec so the two modes (and the
            # morphology grids) can never be silently mixed or mis-joined.
            provenance = {
                "crs_mode": cfg.crs_mode,
                "working_crs": str(working_crs),
                "utm_epsg": epsg,
                "side_length_m": cfg.side_length,
                "grid_origin": [cfg.grid_origin_x, cfg.grid_origin_y],
                "output_crs": cfg.output_crs,
                "clip_to_aoi": cfg.clip_to_aoi,
                "hexagon_count": len(hex_gdf),
                "generator": "geoworkflow HexGridProcessor",
            }
            meta_path = cfg.output_file.with_suffix(cfg.output_file.suffix + ".meta.json")
            meta_path.write_text(json.dumps(provenance, indent=2))
            self.update_progress(1, "Saved")

            result.processed_count = len(hex_gdf)
            result.message = (
                f"Generated {len(hex_gdf)} hexagons "
                f"(side_length={cfg.side_length}m, {system}) -> {cfg.output_file}"
            )
            result.add_output_path(cfg.output_file)
            result.metadata = {**provenance, "output_file": str(cfg.output_file)}

        except Exception as exc:
            result.success = False
            result.message = f"Hex grid generation failed: {exc}"
            self.logger.error(result.message)
            raise ProcessingError(result.message)

        return result

    def _cleanup_custom_processing(self) -> Dict[str, Any]:
        cleanup_info: Dict[str, Any] = {}
        geo_info = self.cleanup_geospatial_resources()
        cleanup_info["geospatial"] = geo_info

        if self.aoi_gdf is not None:
            del self.aoi_gdf
            self.aoi_gdf = None
            cleanup_info["aoi_gdf_cleared"] = True

        return cleanup_info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_hex_grid(
        self, aoi_projected: gpd.GeoDataFrame, working_crs, system: str
    ) -> gpd.GeoDataFrame:
        """Generate hexagons on the lattice; keep those intersecting the AOI (or the bbox)."""
        import numpy as np
        import shapely

        aoi_union = aoi_projected.geometry.union_all()
        min_x, min_y, max_x, max_y = aoi_union.bounds

        sl = self.hexgrid_config.side_length
        ox = self.hexgrid_config.grid_origin_x
        oy = self.hexgrid_config.grid_origin_y

        min_q, max_r = _cartesian_to_axial(min_x, max_y, ox, oy, sl)
        max_q, min_r = _cartesian_to_axial(max_x, min_y, ox, oy, sl)

        padding = 3
        min_q -= padding
        max_q += padding
        min_r -= padding
        max_r += padding

        hex_width = sl * math.sqrt(3)
        hex_height = sl * 2

        # Enumerate candidate cells (bbox pre-filter), build their polygons.
        candidates = []
        polygons = []
        for r in range(min_r, max_r + 1):
            for q in range(min_q, max_q + 1):
                cx, cy = _axial_to_cartesian(q, r, ox, oy, sl)
                if (
                    cx - hex_width / 2 > max_x
                    or cx + hex_width / 2 < min_x
                    or cy - hex_height / 2 > max_y
                    or cy + hex_height / 2 < min_y
                ):
                    continue
                candidates.append((q, r))
                polygons.append(_make_hex_polygon(cx, cy, sl))

        if self.hexgrid_config.clip_to_aoi:
            # One vectorized intersects pass with the AOI as the *prepared subject*.
            # GEOS only uses a prepared geometry's spatial index when it is the
            # subject of the predicate; per-candidate `poly.intersects(aoi_union)`
            # re-scans every AOI vertex each time (Nairobi, 304k vertices x 95k
            # candidates: ~245 s). Prepared + vectorized: <1 s for the same input.
            shapely.prepare(aoi_union)
            mask = shapely.intersects(aoi_union, np.array(polygons, dtype=object))
        else:
            # Fill the bounding box (no AOI clip) — closer to the unclipped grids.
            mask = np.ones(len(polygons), dtype=bool)

        records = [
            {"GridID": _hex_id(q, r, system), "q": q, "r": r, "geometry": poly}
            for (q, r), poly, keep in zip(candidates, polygons, mask)
            if keep
        ]
        if not records:
            raise ProcessingError("No hexagons generated — check AOI bounds and side_length")

        return gpd.GeoDataFrame(records, crs=working_crs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hex_grid_processor(config_path: Path) -> HexGridProcessor:
    config = HexGridConfig.from_file(config_path)
    return HexGridProcessor(config)
