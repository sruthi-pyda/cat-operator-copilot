"""Verification for the curated demo scenarios.

A scenario list nobody checks is worse than no list, because it gets trusted.
These tests assert that each session named in `features/dashboard/demo_selector.py`
still shows what it promises, so the data changing breaks a test rather than the
demo.

Two tiers:
  * cheap checks (session exists, severity, authorization) run always
  * plan checks call the optimizer and the prediction models, so they are opt-in
    via RUN_SLOW=1 -- roughly 6 seconds per scenario

Everything skips when the untracked dataset is absent (D028).
"""
from __future__ import annotations

import os

import pytest

from features.dashboard.data import DashboardData
from features.dashboard.demo_selector import (
    CATEGORY_ORDER,
    SCENARIOS,
    all_scenarios,
    scenario_by_key,
    scenario_for_label,
    scenario_labels,
    scenarios_by_category,
)
from features.passport.authorization import check_authorization

dataset_required = pytest.mark.skipif(
    not DashboardData.from_settings().available(),
    reason="synthetic dataset not present; see docs/DECISIONS.md D028",
)
slow = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="calls the optimizer and prediction models; set RUN_SLOW=1 to include",
)

ALL = list(SCENARIOS)
IDS = [s.key for s in ALL]


@pytest.fixture(scope="module")
def data() -> DashboardData:
    return DashboardData.from_settings()


# --- the list itself ---------------------------------------------------------

def test_there_are_twenty_scenarios():
    assert len(SCENARIOS) == 20


def test_keys_are_unique():
    keys = [s.key for s in SCENARIOS]
    assert len(keys) == len(set(keys))


def test_every_scenario_names_a_known_category():
    assert {s.category for s in SCENARIOS} <= set(CATEGORY_ORDER)


def test_every_scenario_says_what_to_show_and_what_to_say():
    for scenario in SCENARIOS:
        assert scenario.show.strip(), f"{scenario.key} has no 'show'"
        assert scenario.say.strip(), f"{scenario.key} has no 'say'"


def test_every_category_is_represented():
    covered = scenarios_by_category()
    assert set(covered) == set(CATEGORY_ORDER)


def test_the_dashboard_opens_on_the_registered_operator():
    """OP1001 is the only registered face. Opening on any other operator would
    contradict the login the audience just watched."""
    from features.dashboard.demo_selector import default_label_index, default_scenario

    assert default_scenario().operator_id == "OP1001"
    assert default_scenario().expect.get("authorized") is True
    # index 0 is the manual-browse entry, so a real scenario must be past it
    assert default_label_index() >= 1


def test_lookup_helpers_round_trip():
    assert scenario_by_key("full-plan").session_id == "S000146"
    assert scenario_by_key("does-not-exist") is None
    labels = scenario_labels()
    assert len(labels) == len(SCENARIOS)
    assert scenario_for_label(labels[0]) is all_scenarios()[0]
    assert scenario_for_label("nonsense") is None


# --- cheap data checks -------------------------------------------------------

@dataset_required
@pytest.mark.parametrize("scenario", ALL, ids=IDS)
def test_session_exists_and_belongs_to_the_named_operator(scenario, data):
    context = data.build_session_context(scenario.session_id)
    assert context.operator_id == scenario.operator_id


@dataset_required
@pytest.mark.parametrize(
    "scenario", [s for s in ALL if "worst_severity" in s.expect],
    ids=[s.key for s in ALL if "worst_severity" in s.expect],
)
def test_worst_severity_is_what_the_scenario_claims(scenario, data):
    events = data.safety_events(scenario.session_id)
    present = set(events.severity) if not events.empty else set()
    worst = next(
        (s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO") if s in present), "NONE"
    )
    assert worst == scenario.expect["worst_severity"]


@dataset_required
@pytest.mark.parametrize(
    "scenario", [s for s in ALL if "authorized" in s.expect],
    ids=[s.key for s in ALL if "authorized" in s.expect],
)
def test_authorization_outcome_is_what_the_scenario_claims(scenario, data):
    context = data.build_session_context(scenario.session_id)
    repository = data.repository()
    result = check_authorization(
        repository.get_operator(context.operator_id),
        repository.get_machine(context.machine_id),
    )
    assert result.authorized is scenario.expect["authorized"]
    if "refusal_reason" in scenario.expect:
        assert scenario.expect["refusal_reason"] in result.reasons


@dataset_required
def test_the_two_refusal_scenarios_fail_for_different_reasons():
    """One is machine clearance, the other a lapsed certificate -- showing both
    is what proves the check is real rather than a single hard-coded branch."""
    reasons = {
        s.expect.get("refusal_reason")
        for s in SCENARIOS if s.expect.get("authorized") is False
    }
    assert reasons == {"machine_type_not_authorized", "certification_expired"}


# --- expensive plan checks ---------------------------------------------------

@slow
@dataset_required
@pytest.mark.parametrize(
    "scenario",
    [s for s in ALL if "min_plan_steps" in s.expect or "plan_empty" in s.expect],
    ids=[s.key for s in ALL if "min_plan_steps" in s.expect or "plan_empty" in s.expect],
)
def test_plan_matches_the_scenario_claim(scenario, data):
    from features.dashboard import adapters

    context = data.build_session_context(scenario.session_id)
    result = adapters.get_plan(tasks=None, session_context=context)
    if not result.available:
        pytest.skip("Optimization is not integrated in this checkout")

    steps = result.value.get("steps") or []
    if scenario.expect.get("plan_empty"):
        assert not steps, f"{scenario.key} promised an empty plan, got {len(steps)} steps"
    if "min_plan_steps" in scenario.expect:
        assert len(steps) >= scenario.expect["min_plan_steps"], (
            f"{scenario.key} promised at least {scenario.expect['min_plan_steps']} steps, "
            f"got {len(steps)}"
        )
    if "deadlines_met" in scenario.expect:
        assert result.value.get("deadlines_met") == scenario.expect["deadlines_met"]
