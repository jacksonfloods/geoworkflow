"""Tests for the ISO time-selection grammar (pure, no deps)."""
import pytest

from geoworkflow.store.timesel import (
    TimeSelector,
    TimeSelectorError,
    parse_time_selector,
    selection_sql,
    months_of_season,
    season_of_month,
    validate_month,
)


@pytest.mark.parametrize("token,expected", [
    (None, TimeSelector(None, None)),
    ("", TimeSelector(None, None)),
    ("all", TimeSelector(None, None)),
    ("ALL", TimeSelector(None, None)),
    ("2020", TimeSelector(2020, None)),
    ("2020-06", TimeSelector(2020, 6)),
    ("2020-6", TimeSelector(2020, 6)),
    (" 2020-06 ", TimeSelector(2020, 6)),
    ("2019-12", TimeSelector(2019, 12)),
])
def test_parse_accepted(token, expected):
    assert parse_time_selector(token) == expected


@pytest.mark.parametrize("bad", [
    "June", "Jun", "06", "6",            # month names / bare months not allowed
    "2020-13", "2020-00",                # out of range month
    "2020-06-15",                        # bare day
    "2019..2020", "20",                  # ambiguous / non-ISO
    "garbage",
])
def test_parse_rejected(bad):
    with pytest.raises(TimeSelectorError):
        parse_time_selector(bad)


def test_parse_rejects_non_string():
    with pytest.raises(TimeSelectorError):
        parse_time_selector(2020)  # type: ignore[arg-type]


def test_is_instant():
    assert TimeSelector(2020, 6).is_instant
    assert not TimeSelector(2020, None).is_instant
    assert not TimeSelector(None, None).is_instant


def test_selector_sql():
    assert TimeSelector(None, None).sql() == ""
    assert TimeSelector(2020, None).sql() == "year(time) = 2020"
    assert TimeSelector(None, 6).sql() == "month(time) = 6"
    assert TimeSelector(2020, 6).sql("t") == "year(t) = 2020 AND month(t) = 6"


def test_selection_sql_tokens():
    assert selection_sql(None) == ""
    assert selection_sql("2020") == "year(time) = 2020"
    assert selection_sql("2020-06") == "year(time) = 2020 AND month(time) = 6"


def test_selection_sql_slice():
    sql = selection_sql(slice("2019-06", "2020-05"))
    assert "time >= '2019-06-01'" in sql
    assert "time < '2020-06-01'" in sql       # exclusive upper = first of next month

    sql_year = selection_sql(slice("2019", "2020"))
    assert "time >= '2019-01-01'" in sql_year
    assert "time < '2021-01-01'" in sql_year  # through end of 2020

    assert selection_sql(slice("2020", None)) == "time >= '2020-01-01'"
    assert selection_sql(slice(None, "2020")) == "time < '2021-01-01'"


def test_slice_step_rejected():
    with pytest.raises(TimeSelectorError):
        selection_sql(slice("2019", "2020", 2))


def test_seasons():
    assert months_of_season("JJA") == (6, 7, 8)
    assert months_of_season("djf") == (12, 1, 2)
    assert season_of_month(7) == "JJA"
    assert season_of_month(1) == "DJF"
    with pytest.raises(TimeSelectorError):
        months_of_season("XYZ")


@pytest.mark.parametrize("bad", [0, 13, -1, True, 6.0])
def test_validate_month_rejects(bad):
    with pytest.raises(TimeSelectorError):
        validate_month(bad)  # type: ignore[arg-type]
