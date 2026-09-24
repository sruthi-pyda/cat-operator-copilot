"""Tests for predicted-vs-actual comparison (Feature 05).

The interval test is the one that matters: a P50 can be well off while the
P10-P90 band was perfectly reasonable, and a band wide enough to always contain
the answer is not a good band either.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from features.dashboard.comparison import ComparisonRow, compare_prediction_to_outcome

OUTCOME = {
    "actual_task_duration_min": 52.98,
    "actual_fuel_used_l": 14.0,
    "actual_idle_time_min": 7.82,
    "actual_cycle_count": 42,
}


def prediction(eta=(25.0, 31.0, 41.0), fuel=(4.2, 5.1, 6.4)):
    return SimpleNamespace(
        eta_p10=eta[0], eta_p50=eta[1], eta_p90=eta[2],
        fuel_p10=fuel[0], fuel_p50=fuel[1], fuel_p90=fuel[2],
        confidence=0.81,
    )


def test_both_quantities_are_reported():
    rows = compare_prediction_to_outcome(prediction(), OUTCOME)
    assert [r.metric for r in rows] == ["duration", "fuel"]


def test_error_is_actual_minus_predicted():
    duration = compare_prediction_to_outcome(prediction(), OUTCOME)[0]
    assert duration.error == pytest.approx(52.98 - 31.0)
    assert duration.error > 0   # the task overran the estimate


def test_an_actual_outside_the_band_is_reported_as_outside():
    duration = compare_prediction_to_outcome(prediction(), OUTCOME)[0]
    assert duration.within_interval is False


def test_an_actual_inside_the_band_is_reported_as_inside():
    rows = compare_prediction_to_outcome(prediction(eta=(40.0, 50.0, 60.0)), OUTCOME)
    assert rows[0].within_interval is True


@pytest.mark.parametrize("actual", [25.0, 41.0])
def test_the_band_is_inclusive_at_both_edges(actual):
    rows = compare_prediction_to_outcome(
        prediction(), {**OUTCOME, "actual_task_duration_min": actual}
    )
    assert rows[0].within_interval is True


# --- a zeroed field means "not predicted", not "predicted zero" ---------------

def test_a_zeroed_fuel_field_is_absent_not_a_prediction_of_zero():
    """predict_task returns ETA and zeroes fuel; 0.0 L must not read as a result."""
    rows = compare_prediction_to_outcome(prediction(fuel=(0.0, 0.0, 0.0)), OUTCOME)
    fuel = rows[1]
    assert fuel.predicted is False
    assert fuel.predicted_p50 is None
    assert fuel.error is None
    assert fuel.within_interval is None
    assert fuel.to_dict()["predicted P50"] == "—"


def test_a_zeroed_eta_field_is_absent_too():
    """predict_fuel returns fuel and zeroes ETA."""
    rows = compare_prediction_to_outcome(prediction(eta=(0.0, 0.0, 0.0)), OUTCOME)
    assert rows[0].predicted is False
    assert rows[1].predicted is True


def test_the_actual_is_always_shown_even_with_no_prediction():
    rows = compare_prediction_to_outcome(prediction(fuel=(0.0, 0.0, 0.0)), OUTCOME)
    assert rows[1].to_dict()["actual"] == "14.0 L"


# --- presentation ------------------------------------------------------------

def test_payload_is_rendered_with_units_and_signs():
    payload = compare_prediction_to_outcome(prediction(), OUTCOME)[0].to_dict()
    assert payload["predicted P50"] == "31.0 min"
    assert payload["P10-P90"] == "25.0 - 41.0"
    assert payload["actual"] == "53.0 min"
    assert payload["error"].startswith("+")
    assert payload["actual within range"] == "no"


def test_an_underrun_error_is_signed_negative():
    rows = compare_prediction_to_outcome(
        prediction(), {**OUTCOME, "actual_task_duration_min": 20.0}
    )
    assert rows[0].to_dict()["error"].startswith("-")


def test_rows_are_immutable():
    row = compare_prediction_to_outcome(prediction(), OUTCOME)[0]
    assert isinstance(row, ComparisonRow)
    with pytest.raises(AttributeError):
        row.actual = 0.0
