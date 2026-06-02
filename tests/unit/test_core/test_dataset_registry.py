"""
Unit tests for the raster dataset registry (geoworkflow.core.dataset_registry).
"""

import json

import pandas as pd
import pytest

from geoworkflow.core.dataset_registry import (
    AmbiguousDatasetError,
    DatasetRegistry,
    RasterDatasetSpec,
    TimeSpec,
    load_dataset_registry,
)


class TestPackagedDefaults:
    def test_defaults_load(self):
        reg = load_dataset_registry(user_registry=None)
        # conventional user file may or may not exist; defaults must be present
        assert "pm25" in reg
        assert "odiac" in reg

    def test_pm25_spec(self):
        reg = load_dataset_registry()
        pm25 = reg.get("pm25")
        assert pm25.variable == "PM25"
        assert pm25.crs == "EPSG:4326"
        assert pm25.time.mode == "regex"


class TestTimeExtraction:
    def test_pm25_filename_to_month(self):
        reg = load_dataset_registry()
        ts = reg.get("pm25").time_from_filename(
            "V6GL02.04.CNNPM25.GL.201901-201901.nc"
        )
        assert ts == pd.Timestamp("2019-01-01")

    def test_odiac_two_digit_year(self):
        reg = load_dataset_registry()
        ts = reg.get("odiac").time_from_filename(
            "odiac2024_1km_excl_intl_2101.tif"
        )
        # trailing _2101 -> 2021-01, NOT the leading 2024 version
        assert ts == pd.Timestamp("2021-01-01")

    def test_static_dataset(self):
        reg = load_dataset_registry()
        ts = reg.get("worldpop_2020").time_from_filename("anything.tif")
        assert ts == pd.Timestamp("2020-01-01")

    def test_regex_no_match_returns_none(self):
        reg = load_dataset_registry()
        assert reg.get("pm25").time_from_filename("not_a_pm25_file.nc") is None


class TestMatching:
    def test_match_odiac(self):
        reg = load_dataset_registry()
        spec = reg.match("/data/global/odiac/2021/odiac2024_1km_excl_intl_2101.tif")
        assert spec is not None and spec.name == "odiac"

    def test_match_none(self):
        reg = load_dataset_registry()
        assert reg.match("totally_unknown_file.tif") is None

    def test_match_ambiguous_raises(self):
        reg = DatasetRegistry({
            "a": RasterDatasetSpec(name="a", match="*shared*.tif"),
            "b": RasterDatasetSpec(name="b", match="*shared*.tif"),
        })
        with pytest.raises(AmbiguousDatasetError):
            reg.match("my_shared_raster.tif")

    def test_describe_file(self):
        reg = load_dataset_registry()
        info = reg.describe_file("odiac2024_1km_excl_intl_2101.tif")
        assert info["matched"] == "odiac"
        assert info["variable"] == "odiac_co2"
        assert info["time"] == "2021-01-01"


class TestUserOverrideMerge:
    def test_user_overrides_and_adds(self, temp_dir):
        user = temp_dir / "raster_datasets.json"
        user.write_text(json.dumps({
            "datasets": [
                {"name": "pm25", "match": "*CNNPM25*.nc", "variable": "PM25",
                 "units": "CHANGED"},
                {"name": "my_dataset", "match": "mine_*.tif", "variable": "x",
                 "time": {"regex": "mine_(?P<year>\\d{4}).tif"}},
            ]
        }))
        reg = load_dataset_registry(user_registry=user)
        assert reg.get("pm25").units == "CHANGED"   # overridden
        assert "my_dataset" in reg                   # added
        assert "odiac" in reg                        # packaged default retained


class TestValidation:
    def test_bad_regex_rejected(self, temp_dir):
        user = temp_dir / "bad.json"
        user.write_text(json.dumps({
            "datasets": [
                {"name": "broken", "match": "*.tif", "time": {"regex": "(unclosed"}}
            ]
        }))
        with pytest.raises(ValueError):
            load_dataset_registry(user_registry=user, use_defaults=False)

    def test_duplicate_names_rejected(self, temp_dir):
        user = temp_dir / "dup.json"
        user.write_text(json.dumps({
            "datasets": [
                {"name": "x", "match": "a*.tif"},
                {"name": "x", "match": "b*.tif"},
            ]
        }))
        with pytest.raises(ValueError):
            load_dataset_registry(user_registry=user, use_defaults=False)

    def test_timespec_multiple_modes_rejected(self):
        with pytest.raises(ValueError):
            TimeSpec(regex="x(?P<year>\\d{4})", static="2020-01")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError):
            RasterDatasetSpec(name="x", match="*.tif", bogus_field=1)
