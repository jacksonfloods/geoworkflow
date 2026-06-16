"""Tests for the recipe model + default recipe."""
import pytest

from geoworkflow.schemas.config_models import HexDBRecipe
from geoworkflow.store.recipe import default_recipe, write_template


def test_default_recipe_metrics():
    r = default_recipe()
    names = [m.name for m in r.metrics]
    assert names == ["PM25", "odiac", "mod11a1_lst", "copernicus_lc100"]
    lc = next(m for m in r.metrics if m.name == "copernicus_lc100")
    assert lc.annual and lc.statistics == ["majority", "variety"]
    lst = next(m for m in r.metrics if m.name == "mod11a1_lst")
    assert lst.source.ee_asset == "MODIS/061/MOD11A1"


def test_recipe_yaml_roundtrip(tmp_path):
    r = default_recipe()
    p = r.to_yaml(tmp_path / "recipe.yaml")
    assert HexDBRecipe.from_yaml(p) == r


def test_write_template(tmp_path):
    p = write_template(tmp_path / "tmpl.yaml")
    assert p.exists()
    assert HexDBRecipe.from_yaml(p) == default_recipe()


def test_invariants_change_with_grid():
    a = default_recipe()
    b = default_recipe()
    b.grid.side_length = 500.0
    assert a.invariants() != b.invariants()
