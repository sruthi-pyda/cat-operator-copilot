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

def test_an_unavailable_feature_reports_an_owner_and_no_value(monkeypatch):
    """Asserted against a guaranteed-absent feature, so this holds after integration."""
    monkeypatch.setitem(sys.modules, "features.safety", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "features.safety.api", types.SimpleNamespace())
    result = get_safety(session_context=None)
    assert isinstance(result, AdapterResult)
    assert result.available is False
    assert result.value is None
    assert "not_integrated_yet" in result.reason
    assert result.owner == "Member 2"


def test_no_placeholder_value_is_ever_substituted(monkeypatch):
    """A fabricated number on screen is worse than a blank one."""
    for feature in ("features.safety", "features.optimization"):
        monkeypatch.setitem(sys.modules, feature, types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, f"{feature}.api", types.SimpleNamespace())
    assert get_safety(None).value is None


def test_integration_status_lists_every_expected_feature():
    status = integration_status()
    assert set(status) == {
        "prediction", "behavior", "safety", "optimization", "attention",
    }
    assert all(isinstance(r, AdapterResult) for r in status.values())


def test_integration_status_reflects_what_is_actually_importable():
    """Holds whether or not a teammate's module is present -- it asserts the
    adapter tells the truth, not which features happen to exist today."""
    expected = {
        "prediction": "predict_task", "behavior": "analyze_behavior",
        "safety": "evaluate_safety", "optimization": "generate_plan",
        "attention": "route_event",
    }
    for feature, function_name in expected.items():
        assert integration_status()[feature].available == (
            resolve(feature, function_name) is not None
        )


def test_prediction_prefers_the_combined_call(monkeypatch):
    """predict_task zeroes the fuel fields; predict_combined populates both."""
    module = types.SimpleNamespace(
        predict_task=lambda ctx: "eta_only",
        predict_combined=lambda ctx: "both",
    )
    monkeypatch.setitem(sys.modules, "features.prediction", module)
    assert get_prediction(None).value == "both"


def test_prediction_falls_back_when_only_predict_task_exists(monkeypatch):
    module = types.SimpleNamespace(predict_task=lambda ctx: "eta_only")
    monkeypatch.setitem(sys.modules, "features.prediction", module)
    monkeypatch.setitem(sys.modules, "features.prediction.api", types.SimpleNamespace())
    assert get_prediction(None).value == "eta_only"


def test_adapter_uses_a_feature_once_it_exists(monkeypatch):
    module = types.SimpleNamespace(evaluate_safety=lambda ctx: {"severity": "HIGH"})
    monkeypatch.setitem(sys.modules, "features.safety", module)
    result = get_safety(session_context=None)
    assert result.available is True
    assert result.value == {"severity": "HIGH"}
    assert result.feature == FEATURE_SAFETY


def test_a_real_error_inside_a_feature_is_not_disguised_as_missing(monkeypatch):
    """Only ImportError/AttributeError mean 'not integrated'.

    Every candidate module is stubbed, otherwise resolution falls through to the
    real feature and this stops testing what it claims to.
    """
    def broken(_ctx):
        raise ValueError("model failed to load")

    monkeypatch.setitem(
        sys.modules, "features.prediction",
        types.SimpleNamespace(predict_combined=broken, predict_task=broken),
    )
    monkeypatch.setitem(sys.modules, "features.prediction.api", types.SimpleNamespace())
    with pytest.raises(ValueError, match="model failed to load"):
        get_prediction(session_context=None)


def test_resolve_returns_none_for_a_missing_function(monkeypatch):
    monkeypatch.setitem(sys.modules, "features.prediction", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "features.prediction.api", types.SimpleNamespace())
    assert resolve(FEATURE_PREDICTION, "predict_task") is None


def test_a_contract_function_in_the_api_submodule_is_found(monkeypatch):
    """Member 1 put predict_task in features/prediction/api.py with an empty __init__."""
    monkeypatch.setitem(sys.modules, "features.prediction", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "features.prediction.api",
        types.SimpleNamespace(predict_task=lambda ctx: "from_api"),
    )
    assert resolve(FEATURE_PREDICTION, "predict_task")(None) == "from_api"


def test_analyze_behavior_is_found_where_it_actually_lives(monkeypatch):
    """It is implemented in features/prediction/api.py, not features/behavior/."""
    monkeypatch.setitem(sys.modules, "features.behavior", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "features.prediction.api",
        types.SimpleNamespace(analyze_behavior=lambda ctx: "behaviour"),
    )
    assert resolve("behavior", "analyze_behavior")(None) == "behaviour"


def test_a_non_callable_attribute_is_not_mistaken_for_the_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "features.prediction", types.SimpleNamespace(predict_task="not a function"),
    )
    monkeypatch.setitem(sys.modules, "features.prediction.api", types.SimpleNamespace())
    assert resolve(FEATURE_PREDICTION, "predict_task") is None
