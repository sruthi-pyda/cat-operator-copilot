"""Tests for telemetry replay (integration Phase 6).

Uses fixture telemetry rather than the untracked dataset (D028). The fixture
session walks safe_idle -> digging -> swinging -> safe_idle, which is the shape
the demo narrative needs: the Buddy must switch off the moment work starts and
come back only when the machine is genuinely idle again.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from features.dashboard.data import DashboardData
from features.dashboard.replay import ReplayStep, TelemetryReplay

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def replay() -> TelemetryReplay:
    return TelemetryReplay(DashboardData(FIXTURES), "S000001")


def test_every_telemetry_row_becomes_one_step(replay):
    steps = list(replay.steps())
    assert len(steps) == 4
    assert [s.index for s in steps] == [0, 1, 2, 3]


def test_steps_are_ordered_by_time(replay):
    timestamps = [s.timestamp for s in replay.steps()]
    assert timestamps == sorted(timestamps)


def test_only_the_requested_session_is_replayed(replay):
    assert all(s.timestamp.startswith("2026-01-01T08") for s in replay.steps())


def test_buddy_follows_the_machine_state(replay):
    availability = [(s.machine_state, s.buddy_available) for s in replay.steps()]
    assert availability == [
        ("safe_idle", True),
        ("digging", False),
        ("swinging", False),
        ("safe_idle", True),
    ]


def test_a_blocked_step_explains_itself(replay):
    digging = list(replay.steps())[1]
    assert "machine_state_not_safe" in digging.safe_state.reasons
    assert "attachment_movement_active" in digging.safe_state.reasons


def test_a_safety_event_attaches_to_the_step_it_happened_during(replay):
    """The 08:01:30 event belongs to the 08:01:00 row, not the 08:02:00 one."""
    steps = list(replay.steps())
    assert [e["event_id"] for e in steps[1].safety_events] == ["SE00001"]
    assert steps[0].safety_events == ()
    assert steps[2].safety_events == ()


def test_an_event_after_the_last_row_still_lands_on_the_last_step(replay):
    last = list(replay.steps())[-1]
    assert [e["event_id"] for e in last.safety_events] == ["SE00002"]


def test_another_sessions_event_never_appears(replay):
    seen = {e["event_id"] for s in replay.steps() for e in s.safety_events}
    assert "SE00003" not in seen


def test_limit_truncates_without_reordering(replay):
    steps = list(replay.steps(limit=2))
    assert len(steps) == 2
    assert steps[-1].machine_state == "digging"


def test_run_collects_the_same_steps(replay):
    assert [s.index for s in replay.run()] == [0, 1, 2, 3]


def test_a_session_with_no_telemetry_yields_nothing():
    empty = TelemetryReplay(DashboardData(FIXTURES), "S999999")
    assert list(empty.steps()) == []


def test_safety_is_always_reported_never_silently_skipped(replay):
    """The replay must never imply a decision no feature actually made.

    Integration-agnostic on purpose: before Safety lands the step carries an
    unavailable result naming its owner; after, it carries the real evaluation.
    What must never happen is a step with no safety information at all.
    """
    step = next(iter(replay.steps()))
    evaluation = step.safety_evaluation
    assert evaluation is not None
    assert evaluation.owner == "Member 2"
    if evaluation.available:
        assert evaluation.value is not None
    else:
        assert evaluation.value is None
        assert "not_integrated_yet" in evaluation.reason


# --- presentation ------------------------------------------------------------

def test_summary_shows_time_state_and_buddy(replay):
    line = next(iter(replay.steps())).summary()
    assert "08:00:00" in line
    assert "safe_idle" in line
    assert "buddy:on" in line


def test_summary_flags_a_safety_event(replay):
    line = list(replay.steps())[1].summary()
    assert "CRITICAL" in line
    assert "worker_inside_swing_envelope_with_closing_motion" in line


def test_payload_round_trips(replay):
    payload = list(replay.steps())[1].to_dict()
    assert payload["machine_state"] == "digging"
    assert payload["buddy_available"] is False
    assert payload["buddy_reasons"]
    assert payload["safety_events"][0]["severity"] == "CRITICAL"


def test_step_is_immutable(replay):
    step = next(iter(replay.steps()))
    assert isinstance(step, ReplayStep)
    with pytest.raises(AttributeError):
        step.machine_state = "tampered"
