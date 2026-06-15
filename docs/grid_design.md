# Hexagonal grid — design & decision record

Why the hex grid is built the way it is. This is the methods record for the grid
(useful for a paper's methods section and for whoever maintains this next), and it
names the standard GIS problems behind each choice so the decisions can be
checked against the literature.

## The grid in one paragraph

Flat-top hexagons, **250 m side length**, addressed by axial coordinates `(q, r)`
from a fixed global origin `(-3,000,000, -2,000,000) m`. The generator
(`processors/spatial/hexgrid.py`) runs in one of two CRS modes (`crs_mode`):
**`albers_global`** (continental equal-area, ESRI:102022) or **`utm_local`**
(per-city UTM). Both share the origin and the axial math, so a UTM grid for a
city reproduces the lab's morphology grids exactly (verified: 0 m residual,
IoU 1.0). GridIDs are positional — `SSA_HQ{q}_R{r}` (Albers) or
`UTM{epsg}_HQ{q}_R{r}` (UTM) — and every hexagon carries a true `area_m2`.

## The three GIS problems behind the choices

The grid sits at the intersection of three classic, well-documented problems.
None can be "solved" — each is a deliberate, documented choice.

### 1. Map-projection distortion (the property tradeoff)
No flat projection preserves area, shape, and distance at once. Equal-area
(Albers) distorts shape; conformal (UTM) distorts area. We need both properties
for different uses, so we offer **both modes** rather than pick one:
- **Albers** — equal-area + one continental lattice → the right backbone for the
  raster-statistic **cubes** (densities comparable everywhere; seamless tiling;
  globally stable IDs; immune to zone-flip, see below).
- **UTM** — conformal → faithful local **shape** for geometry-sensitive
  morphology/connectivity metrics, and it **coincides** with the lab's existing
  morphology grids so the two datasets join hex-for-hex.

The area cost of UTM is mitigated by storing a true `area_m2` (below).
*Refs: Snyder, "Map Projections: A Working Manual" (USGS PP 1395); Tissot's indicatrix.*

### 2. MAUP — the Modifiable Areal Unit Problem
Results on aggregated cells depend on the cells' **size** (scale effect) and
**placement** (zonation effect). Our response is to **fix and document** both:
**250 m** side (chosen to match the morphology grids — measured, not assumed) and
a fixed global origin, so the lattice is reproducible and shared. MAUP is not
removable; recording the cell size and origin in the methods is what makes
results interpretable and comparable. *Ref: Openshaw (1984), CATMOG 38.*

### 3. Change of support / areal interpolation
"Support" = the patch a value belongs to. Combining values across supports
(pixels→hexes, or his hexes→ours) is the change-of-support problem. We avoid the
lossy form of it two ways:
- **Raster → hex**: exactextract computes exact coverage fractions (no uniformity
  assumption) — more rigorous than areal interpolation; see Pipeline 1.
- **His hexes → ours**: we make the lattices **coincide** (same CRS+origin+size in
  UTM mode), so the transfer is a 1:1 geometry join, not interpolation.
- When a transfer *is* unavoidable, classify each variable **intensive vs
  extensive** (means/concentrations → average; counts/totals → sum) and use
  PySAL `tobler` (library, not hand-rolled). *Ref: Gotway & Young (2002), JASA 97:632.*

## Zone pinning (reproducibility-critical)

Per-city UTM has one robustness hazard: the zone is chosen from the AOI's
position, so a future AfricaPolis boundary edit could flip a city across a 6°
zone line and silently relocate its whole grid. We neutralize this with the
**grid zone manifest** (`data/config/grid_zone_manifest.csv`, keyed by
`Agglomeration_ID`): each city's UTM zone is decided **once**
(`geopandas.estimate_utm_crs` — validated to match the morphology grids' zones),
**recorded**, and **read** on every regeneration — never recomputed from a moving
centroid. See `core/grid_registry.py`.

## GridIDs — positional, never sequential

The ID is a pure function of position `(q, r)`, so the same physical hexagon
always gets the same ID, re-derivable by anyone with no coordination — the
property that makes a shared grid actually shareable, and that keeps results
reproducible across runs and people. We do **not** use a sequential
(creation-order) scheme: it is unstable under regeneration and not joinable.
In UTM the prefix carries the zone (`UTM{epsg}_...`) so IDs stay globally unique.

## `area_m2` — true area per hexagon

Stored as a column, computed in the equal-area CRS regardless of the grid's
storage CRS. This makes downstream **densities** correct without the consumer
needing to reproject (a grid stored in EPSG:4326 would give area in square
degrees). In UTM it also records the small true-vs-nominal area drift
(~0.2%) that the conformal projection introduces.

## Provenance

Each grid is written with a `*.meta.json` sidecar recording `crs_mode`,
`working_crs`, `utm_epsg`, `side_length`, `origin`, and `clip_to_aoi`, and the
cube carries the same in its attributes — so an Albers grid and a UTM grid (or
the morphology grids vs ours) can never be silently mixed.

## Practical guidance: which mode?

- **Building the raster-statistic cubes as a standalone, cross-city comparable
  product** → `albers_global`.
- **Joining per-hex with the morphology grids, or geometry-sensitive metrics** →
  `utm_local` (coincides with those grids; pin the zone via the manifest).

The lab's current 7-city cube set is built in **`utm_local`** so the cubes join
the morphology grids directly.
