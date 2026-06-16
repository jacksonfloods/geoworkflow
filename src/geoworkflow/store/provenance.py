"""Living provenance for the hex database.

Two files in the warehouse make the database self-describing and keep its
provenance *current* as it grows:

  * ``_recipe.yaml`` — the canonical current build spec (the live recipe). Read
    by add-city / add-metric so new pieces are built identically to the rest.
  * ``_provenance.jsonl`` — an append-only op log (one JSON line per
    build / add-city / add-metric / remove) with a timestamp, the git commit, and
    the operation's tally.

:func:`check_consistency` guards the warehouse against heterogeneity: an op whose
grid/time invariants differ from the live recipe is rejected (changing those
needs a full rebuild, since GridIDs/columns would change).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from geoworkflow.core.exceptions import GeoWorkflowError
from geoworkflow.schemas.config_models import HexDBRecipe

RECIPE_FILE = "_recipe.yaml"
LOG_FILE = "_provenance.jsonl"


def recipe_path(warehouse_dir) -> Path:
    return Path(warehouse_dir) / RECIPE_FILE


def log_path(warehouse_dir) -> Path:
    return Path(warehouse_dir) / LOG_FILE


def write_recipe(warehouse_dir, recipe: HexDBRecipe) -> Path:
    """Persist the live recipe (the DB's current composition)."""
    return recipe.to_yaml(recipe_path(warehouse_dir))


def read_recipe(warehouse_dir) -> Optional[HexDBRecipe]:
    """Load the live recipe, or None if the warehouse has none yet."""
    path = recipe_path(warehouse_dir)
    return HexDBRecipe.from_yaml(path) if path.exists() else None


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def append_op(warehouse_dir, op: str, **details) -> None:
    """Append one operation record to the append-only provenance log."""
    path = log_path(warehouse_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "op": op,
        "git": _git_commit(),
        **details,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_log(warehouse_dir) -> list:
    """Return the op log as a list of dicts (empty if none)."""
    path = log_path(warehouse_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_consistency(warehouse_dir, recipe: HexDBRecipe) -> None:
    """Raise if ``recipe``'s build invariants differ from the live recipe.

    Prevents a heterogeneous warehouse: grid params and time range must match
    everything already built. Changing them requires a fresh/overwriting build.
    """
    live = read_recipe(warehouse_dir)
    if live is None:
        return
    if live.invariants() != recipe.invariants():
        raise GeoWorkflowError(
            "build invariants (grid/time) differ from the live warehouse recipe; "
            "this would create a heterogeneous database. Rebuild with overwrite, "
            "or match the existing recipe at " + str(recipe_path(warehouse_dir))
        )
