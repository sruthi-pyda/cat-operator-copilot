"""Tests for the dashboard's adapter ports and its leakage boundary (Feature 05).

The adapter tests deliberately assert the *current* integration state: the other
features are not written yet, so the dashboard must report them as unavailable
rather than show a number. When a teammate lands their module these flip, which
is the point -- they double as an integration tripwire.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from features.dashboard.adapters import (
    FEATURE_PREDICTION,
    FEATURE_SAFETY,
    AdapterResult,
    get_prediction,
    get_safety,
    integration_status,
    resolve,
)
from features.dashboard.data import ACTUAL_COLUMNS, DashboardData, MissingDatasetError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data() -> DashboardData:
    return DashboardData(FIXTURES)


# --- leakage boundary --------------------------------------------------------

def test_pre_task_context_excludes_every_outcome_column(data):
    context = data.pre_task_context("S000001")
    for column in ACTUAL_COLUMNS:
        assert column not in context


def test_pre_task_context_keeps_the_predictive_inputs(data):
    context = data.pre_task_context("S000001")
    assert context["task_type"] == "grading"
    assert context["soil_hardness_index"] == 0.575
    assert context["congestion_level"] == "medium"


def test_outcome_returns_only_the_outcome_columns(data):
    outcome = data.outcome("S000001")
    assert set(outcome) == set(ACTUAL_COLUMNS)
    assert outcome["actual_task_duration_min"] == 52.98


def test_session_context_carries_no_outcome_anywhere(data):
    """A SessionContext handed to any feature must not contain a target."""
    context = data.build_session_context("S000001")
    flat = repr(context)
    for column in ACTUAL_COLUMNS:
        assert column not in flat
    assert "52.98" not in flat     # actual duration
    assert "14.0" not in flat      # actual fuel


def test_session_context_is_populated_from_operator_machine_and_site(data):
    context = data.build_session_context("S000001")
    assert context.operator_id == "OP1003"
    assert context.operator.machine_skill == "advanced"
    assert context.machine.machine_type == "excavator"
    assert context.task.task_type == "grading"
    assert context.site.hardness == 0.575
    assert context.weather.visibility_m == 4488.0
    assert context.temporal.recent_idle_ratio == 0.1718
    assert context.data_quality.synthetic_flag is True


def test_unknown_session_raises(data):
    with pytest.raises(KeyError):
        data.pre_task_context("S999999")


def test_a_missing_table_explains_how_to_get_the_dataset(tmp_path):
    with pytest.raises(MissingDatasetError, match="sruthi-data-models"):
        DashboardData(tmp_path).sessions()


def test_availability_reflects_the_dataset(data, tmp_path):
    assert data.available() is True
    assert DashboardData(tmp_path).available() is False


# --- adapters ----------------------------------------------------------------

def test_absent_feature_is_reported_unavailable_with_an_owner():
    result = get_prediction(session_context=None)
    assert isinstance(result, AdapterResult)
    assert result.available is False
    assert result.value is None
    assert "not_integrated_yet" in result.reason
    assert result.owner == "Member 1"


def test_no_placeholder_value_is_ever_substituted():
    """A fabricated number on screen is worse than a blank one."""
    for result in (get_prediction(None), get_safety(None)):
        assert result.value is None


def test_integration_status_lists_every_expected_feature():
    status = integration_status()
    assert set(status) == {
        "prediction", "behavior", "safety", "optimization", "attention",
    }
    assert all(isinstance(r, AdapterResult) for r in status.values())


def test_adapter_uses_a_feature_once_it_exists(monkeypatch):
    module = types.SimpleNamespace(evaluate_safety=lambda ctx: {"severity": "HIGH"})
    monkeypatch.setitem(sys.modules, "features.safety", module)
    result = get_safety(session_context=None)
    assert result.available is True
    assert result.value == {"severity": "HIGH"}
    assert result.feature == FEATURE_SAFETY


def test_a_real_error_inside_a_feature_is_not_disguised_as_missing(monkeypatch):
    """Only ImportError/AttributeError mean 'not integrated'."""
    def broken(_ctx):
        raise ValueError("model failed to load")

    module = types.SimpleNamespace(predict_task=broken)
    monkeypatch.setitem(sys.modules, "features.prediction", module)
    with pytest.raises(ValueError, match="model failed to load"):
        get_prediction(session_context=None)


def test_resolve_returns_none_for_a_missing_function(monkeypatch):
    monkeypatch.setitem(sys.modules, "features.prediction", types.SimpleNamespace())
    assert resolve(FEATURE_PREDICTION, "predict_task") is None
