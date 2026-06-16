"""City catalog — resolve ISO3 / name / Agglomeration_ID to file paths.

Manifest-driven (``grid_zone_manifest.csv`` + optional
``agglomerations_complexity.csv``): the single place that maps a human
``db["KEN"]["Nairobi"]`` request to a stable ``Agglomeration_ID`` and the files
that back it (the per-city hex ``.gpkg`` and warehouse partition). Duplicate city
names (≈95 of them, e.g. "Malindi") raise :class:`AmbiguousCityError` so the
caller disambiguates with an explicit ``aggid``.

The manifest knows *all* candidate cities (for building); whether a city is
actually *built* is determined elsewhere by partition existence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from geoworkflow.core.exceptions import GeoWorkflowError

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False


class CityNotFoundError(GeoWorkflowError):
    """No city matched the given ISO3 / name / aggid."""


class AmbiguousCityError(GeoWorkflowError):
    """A name matched more than one city; disambiguate with an aggid."""

    def __init__(self, iso3: str, name: str, aggids: List[int]):
        self.iso3, self.name, self.aggids = iso3, name, aggids
        super().__init__(
            f"'{name}' in {iso3} matches {len(aggids)} agglomerations "
            f"{sorted(aggids)}; pass aggid= to disambiguate "
            f"(e.g. db['{iso3}'].city(aggid={sorted(aggids)[0]}))"
        )


_GRIDID_EPSG_RE = re.compile(r"^UTM(?P<epsg>\d+)_")


def _norm(name: str) -> str:
    """Normalize a city name for matching: casefold, spaces/hyphens -> '_'."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().casefold())


@dataclass(frozen=True)
class CityRecord:
    """Everything needed to locate one city's data."""

    aggid: int
    iso3: str
    name: str                 # manifest slug, e.g. "nairobi"
    display_name: str         # proper name, e.g. "Nairobi"
    utm_epsg: int
    gpkg: Path
    partition: Path           # warehouse partition .parquet
    combined_parquet: Optional[Path] = None   # legacy grid_stats (for reuse migration)
    n_nodes: Optional[int] = None


class Catalog:
    """Resolve cities and their file paths from the grid zone manifest."""

    def __init__(
        self,
        manifest_csv: Path,
        hexagglo_dir: Path,
        warehouse_dir: Path,
        *,
        complexity_csv: Optional[Path] = None,
        grid_stats_dir: Optional[Path] = None,
    ):
        if not HAS_PANDAS:  # pragma: no cover
            raise GeoWorkflowError("pandas is required for the city catalog")
        self.hexagglo_dir = Path(hexagglo_dir)
        self.warehouse_dir = Path(warehouse_dir)
        self.grid_stats_dir = Path(grid_stats_dir) if grid_stats_dir else None

        man = pd.read_csv(manifest_csv)
        man["Agglomeration_ID"] = man["Agglomeration_ID"].astype(int)

        # Actual on-disk filenames use a slug that differs from the manifest
        # 'name' for almost every city (e.g. manifest 'Lagos' -> file '5199_lagos').
        # So map aggid -> real file by globbing + parsing the leading id, rather
        # than constructing paths from the manifest name. (Falls back to a
        # constructed path when nothing is globbed, e.g. in unit tests.)
        self._gpkg_by_aggid = self._glob_aggid_files(self.hexagglo_dir, "*/*_hex.gpkg", "_hex.gpkg")
        self._slug_by_aggid = {
            aid: p.name[:-len("_hex.gpkg")].split("_", 1)[1]
            for aid, p in self._gpkg_by_aggid.items()
            if "_" in p.name[:-len("_hex.gpkg")]
        }
        self._combined_by_aggid = self._glob_combined()

        nodes: Dict[int, int] = {}
        display: Dict[int, str] = {}
        if complexity_csv and Path(complexity_csv).exists():
            cx = pd.read_csv(complexity_csv)
            cx = cx.dropna(subset=["Agglomeration_ID"])
            cx["Agglomeration_ID"] = cx["Agglomeration_ID"].astype(int)
            for _, r in cx.iterrows():
                aid = int(r["Agglomeration_ID"])
                if pd.notna(r.get("n_nodes")):
                    nodes[aid] = int(r["n_nodes"])
                if pd.notna(r.get("Agglomeration_Name")):
                    display[aid] = str(r["Agglomeration_Name"])

        self._by_aggid: Dict[int, CityRecord] = {}
        # (iso3, normalized name) -> [aggid, ...]   (lists capture duplicates)
        self._by_name: Dict[tuple, List[int]] = {}
        self._by_zone: Dict[int, List[int]] = {}

        for _, row in man.iterrows():
            aid = int(row["Agglomeration_ID"])
            iso3 = str(row["ISO3"])
            name = str(row["name"])
            epsg = int(row["utm_epsg"])
            slug = self._slug_by_aggid.get(aid, name)
            rec = CityRecord(
                aggid=aid, iso3=iso3, name=slug,
                display_name=display.get(aid, slug.replace("_", " ").title()),
                utm_epsg=epsg,
                gpkg=self._gpkg_by_aggid.get(aid) or (
                    self.hexagglo_dir / iso3 / f"{aid}_{name}_hex.gpkg"),
                partition=self.warehouse_dir / f"ISO3={iso3}" / f"aggid={aid}" / "part.parquet",
                combined_parquet=self._combined_by_aggid.get(aid) or self._resolve_combined(aid, name),
                n_nodes=nodes.get(aid),
            )
            self._by_aggid[aid] = rec
            # Index by every spelling a user might type: actual slug + manifest name.
            for alias in {_norm(slug), _norm(name), _norm(rec.display_name)}:
                self._by_name.setdefault((iso3, alias), [])
                if aid not in self._by_name[(iso3, alias)]:
                    self._by_name[(iso3, alias)].append(aid)
            self._by_zone.setdefault(epsg, []).append(aid)

    @staticmethod
    def _glob_aggid_files(root: Path, pattern: str, suffix: str) -> Dict[int, Path]:
        """Map aggid -> file by globbing and parsing the leading integer id."""
        out: Dict[int, Path] = {}
        for p in Path(root).glob(pattern):
            head = p.name[:-len(suffix)].split("_", 1)[0]
            if head.isdigit():
                out[int(head)] = p
        return out

    def _glob_combined(self) -> Dict[int, Path]:
        """Map aggid -> legacy combined.parquet (id-prefixed, or bare-name by slug)."""
        if self.grid_stats_dir is None or not self.grid_stats_dir.exists():
            return {}
        out = self._glob_aggid_files(self.grid_stats_dir, "*_combined.parquet", "_combined.parquet")
        slug_to_aggid = {slug: aid for aid, slug in self._slug_by_aggid.items()}
        for p in self.grid_stats_dir.glob("*_combined.parquet"):
            stem = p.name[:-len("_combined.parquet")]
            if not stem.split("_", 1)[0].isdigit():        # bare name -> slug
                aid = slug_to_aggid.get(stem)
                if aid is not None:
                    out.setdefault(aid, p)
        return out

    def _resolve_combined(self, aggid: int, name: str) -> Optional[Path]:
        """Legacy grid_stats parquet path: ``<aggid>_<name>`` then bare ``<name>``
        (7 early cities), for the one-time reuse migration."""
        if self.grid_stats_dir is None:
            return None
        idn = self.grid_stats_dir / f"{aggid}_{name}_combined.parquet"
        if idn.exists():
            return idn
        bare = self.grid_stats_dir / f"{name}_combined.parquet"
        return bare if bare.exists() else None

    # -- resolution -----------------------------------------------------
    def resolve_aggid(self, aggid: int) -> CityRecord:
        try:
            return self._by_aggid[int(aggid)]
        except KeyError:
            raise CityNotFoundError(f"no agglomeration with id {aggid}")

    def resolve(self, iso3: str, name: Optional[str] = None, *,
                aggid: Optional[int] = None) -> CityRecord:
        if aggid is not None:
            rec = self.resolve_aggid(aggid)
            if rec.iso3 != iso3:
                raise CityNotFoundError(f"aggid {aggid} is in {rec.iso3}, not {iso3}")
            return rec
        if name is None:
            raise CityNotFoundError("provide a name or aggid")
        hits = self._by_name.get((iso3, _norm(name)))
        if not hits:
            raise CityNotFoundError(f"no city '{name}' in {iso3}")
        if len(hits) > 1:
            raise AmbiguousCityError(iso3, name, hits)
        return self._by_aggid[hits[0]]

    # -- listings -------------------------------------------------------
    def countries(self) -> List[str]:
        return sorted({r.iso3 for r in self._by_aggid.values()})

    def cities(self, iso3: str) -> List[str]:
        return sorted(r.display_name for r in self._by_aggid.values() if r.iso3 == iso3)

    def records(self, iso3: Optional[str] = None) -> List[CityRecord]:
        recs = self._by_aggid.values()
        return [r for r in recs if iso3 is None or r.iso3 == iso3]

    def cities_in_zone(self, utm_epsg: int) -> List[CityRecord]:
        """Cities sharing a UTM zone — candidates for a GridID lookup."""
        return [self._by_aggid[a] for a in self._by_zone.get(int(utm_epsg), [])]

    def find_hexagon(self, gridid: str) -> List[CityRecord]:
        """Candidate cities for a GridID, pruned by its embedded UTM EPSG.

        Returns every city in the GridID's UTM zone; the engine then checks which
        one actually contains the hexagon. Empty list if the EPSG is unparseable.
        """
        m = _GRIDID_EPSG_RE.match(str(gridid))
        if not m:
            return []
        return self.cities_in_zone(int(m.group("epsg")))

    def resolve_selector(self, selector) -> List[CityRecord]:
        """Expand a :class:`CitySelector` into concrete CityRecords."""
        if getattr(selector, "all", False):
            return self.records()
        if selector.aggid is not None:
            return [self.resolve_aggid(a) for a in selector.aggid]
        if selector.iso3 is not None:
            return [r for iso in selector.iso3 for r in self.records(iso)]
        if selector.name is not None:
            # names without ISO3: search all countries, error on ambiguity
            out = []
            for nm in selector.name:
                matches = [r for r in self._by_aggid.values() if _norm(r.name) == _norm(nm)]
                if not matches:
                    raise CityNotFoundError(f"no city named '{nm}'")
                if len(matches) > 1:
                    raise AmbiguousCityError(
                        "/".join(sorted({m.iso3 for m in matches})), nm,
                        [m.aggid for m in matches])
                out.append(matches[0])
            return out
        raise CityNotFoundError("empty city selector")
