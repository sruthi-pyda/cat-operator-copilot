"""Smoke tests for the Streamlit dashboard (Feature 05).

These drive the real app through Streamlit's AppTest harness, so a crash in a
render function fails the suite rather than the demo.

They need the synthetic dataset, which is untracked on this branch (D028), so
they skip when it is absent instead of failing on a teammate's clone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from features.dashboard.data import DashboardData

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "ui" / "dashboard.py"

dataset_required = pytest.mark.skipif(
    not DashboardData.from_settings().available(),
    reason="synthetic dataset not present; see docs/DECISIONS.md D028",
)


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(str(APP), default_timeout=120)
    instance.run()
    return instance


@dataset_required
def test_the_dashboard_renders_without_raising(app):
    assert not app.exception


@dataset_required
def test_the_synthetic_disclaimer_is_always_visible(app):
    captions = " ".join(element.value for element in app.caption)
    assert "All data is synthetic" in captions


@dataset_required
def test_unintegrated_features_are_declared_not_faked(app):
    """Missing features must say so; no placeholder number may appear."""
    notices = " ".join(element.value for element in app.info)
    assert "Not integrated yet" in notices


@dataset_required
def test_every_required_dashboard_section_is_present(app):
    headings = " ".join(element.value for element in app.subheader)
    for section in (
        "Shift status", "Plan", "Prediction", "Conditions",
        "Live operation", "Operating Buddy", "Training gate",
        "End of shift",
    ):
        assert section in headings


@dataset_required
def test_the_training_gate_refuses_a_context_driven_issue(app):
    """The operator-protection rule, exercised through the UI."""
    attribution = next(s for s in app.selectbox if s.label == "Attribution")
    attribution.set_value("context-driven").run()
    notices = " ".join(element.value for element in app.info)
    assert "No training triggered" in notices
    assert "attribution_not_operator_linked" in notices
