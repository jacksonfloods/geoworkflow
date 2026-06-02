"""
Unit tests for the zonal statistic registry (geoworkflow.core.statistics).
"""

import numpy as np
import pytest

from geoworkflow.core import statistics as st
from geoworkflow.core.statistics import (
    ZonalStatistic,
    available_statistics,
    register_statistic,
    resolve_statistic,
    resolve_statistics,
    unregister_statistic,
)


class TestBuiltinRegistry:
    def test_core_builtins_present(self):
        names = set(available_statistics())
        assert {"weighted_mean", "mean", "min", "max", "sum", "count",
                "median", "stdev", "variance", "range"}.issubset(names)

    def test_weighted_mean_is_coverage_weighted_mean_op(self):
        stat = resolve_statistic("weighted_mean")
        assert stat.exactextract_op == "mean"
        assert stat.is_reducer is False

    def test_mean_is_alias_of_weighted_mean(self):
        assert resolve_statistic("mean").exactextract_op == "mean"
        assert resolve_statistic("std").exactextract_op == "stdev"

    def test_range_is_reducer_backed(self):
        stat = resolve_statistic("range")
        assert stat.is_reducer is True
        assert stat.exactextract_op is None


class TestParameterizedResolution:
    @pytest.mark.parametrize("name,expected", [
        ("percentile_90", "quantile(q=0.9)"),
        ("p90", "quantile(q=0.9)"),
        ("p25", "quantile(q=0.25)"),
        ("quantile_0.5", "quantile(q=0.5)"),
    ])
    def test_percentile_and_quantile_forms(self, name, expected):
        assert resolve_statistic(name).exactextract_op == expected

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            resolve_statistic("not_a_real_statistic")

    def test_out_of_range_percentile_raises(self):
        with pytest.raises(KeyError):
            resolve_statistic("p150")

    def test_resolve_list_dedupes_preserving_order(self):
        resolved = resolve_statistics(["max", "weighted_mean", "max", "min"])
        assert [s.name for s in resolved] == ["max", "weighted_mean", "min"]


class TestRegistration:
    def test_register_op_statistic_and_cleanup(self):
        try:
            stat = register_statistic("custom_op_stat", op="max", description="d")
            assert isinstance(stat, ZonalStatistic)
            assert resolve_statistic("custom_op_stat").exactextract_op == "max"
        finally:
            unregister_statistic("custom_op_stat")
        assert "custom_op_stat" not in available_statistics()

    def test_register_reducer_via_decorator(self):
        try:
            @register_statistic("custom_reducer", description="sum of values")
            def _reducer(values, coverage):
                return float(np.sum(values))

            stat = resolve_statistic("custom_reducer")
            assert stat.is_reducer is True
            # decorator returns the original callable
            assert _reducer(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 3.0
        finally:
            unregister_statistic("custom_reducer")

    def test_duplicate_registration_requires_overwrite(self):
        try:
            register_statistic("dup_stat", op="min")
            with pytest.raises(ValueError):
                register_statistic("dup_stat", op="max")
            # overwrite succeeds
            register_statistic("dup_stat", op="max", overwrite=True)
            assert resolve_statistic("dup_stat").exactextract_op == "max"
        finally:
            unregister_statistic("dup_stat")

    def test_zonalstatistic_requires_exactly_one_kind(self):
        with pytest.raises(ValueError):
            ZonalStatistic(name="bad", exactextract_op="mean",
                           reducer=lambda v, c: 0.0)
        with pytest.raises(ValueError):
            ZonalStatistic(name="bad", exactextract_op=None, reducer=None)
