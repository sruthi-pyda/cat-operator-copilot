"""Tests for Attention Manager candidate events produced by this slice.

These check that candidates carry the metadata the arbiter needs and that they
stop short of arbitrating. The decision itself is Member 2's; a candidate that
already decided would give the system two attention owners.

The final test is this slice's half of demo Test 3 — a critical safety event
arriving while training is waiting. Proving safety actually wins needs the
Attention Manager; what is provable here is that the two candidates reach it
correctly ranked and that training is not marked urgent.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from features.dashboard.attention_candidates import (
    ACTIONABILITY_INFORMATIONAL,
    ACTIONABILITY_NOW,
    ACTIONABILITY_WHEN_STOPPED,
    DECISION_PENDING,
    EVENT_BUDDY,
    EVENT_REPLAN,
    EVENT_SAFETY,
    EVENT_TRAINING,
    URGENCY_DEFERRABLE,
    URGENCY_IMMEDIATE,
    buddy_candidate,
    operator_state_from,
    replan_candidate,
    safety_candidate,
    training_candidate,
)

TRIGGER = SimpleNamespace(confidence=0.82, lesson_id="L003", escalation_state="first_trigger")
ANSWERED = SimpleNamespace(answered=True)
CRITICAL_EVENT = {
    "event_id": "SE00123",
    "severity": "CRITICAL",
    "timestamp": "2026-01-01T08:01:30",
    "confidence": 0.94,
    "machine_state": "digging",
}


@pytest.mark.parametrize(
    "machine_state,expected",
    [("digging", "active"), ("loading", "active"), ("swinging", "active"),
     ("traveling", "active"), ("grading", "active"),
     ("safe_idle", "available"), ("parked", "available"), (None, "unknown")],
)
def test_operator_state_reflects_whether_the_operator_is_working(machine_state, expected):
    assert operator_state_from(machine_state) == expected


def test_no_candidate_decides_its_own_fate():
    """Deciding here would create a second attention owner."""
    candidates = [
        training_candidate(TRIGGER, "safe_idle"),
        buddy_candidate(ANSWERED, "safe_idle"),
        replan_candidate([1, 2], "safe_idle"),
        safety_candidate(CRITICAL_EVENT),
    ]
    assert all(c.decision == DECISION_PENDING for c in candidates)
    assert all(c.reason.startswith("awaiting_") for c in candidates)


def test_training_is_deferrable_and_only_actionable_when_stopped():
    candidate = training_candidate(TRIGGER, "digging")
    assert candidate.event_type == EVENT_TRAINING
    assert candidate.urgency == URGENCY_DEFERRABLE
    assert candidate.actionability == ACTIONABILITY_WHEN_STOPPED
    assert candidate.operator_state == "active"
    assert candidate.confidence == 0.82


def test_training_never_claims_urgency_even_when_the_operator_is_free():
    assert training_candidate(TRIGGER, "safe_idle").urgency == URGENCY_DEFERRABLE


def test_buddy_is_deferrable_and_waits_for_a_stop():
    candidate = buddy_candidate(ANSWERED, "digging")
    assert candidate.event_type == EVENT_BUDDY
    assert candidate.urgency == URGENCY_DEFERRABLE
    assert candidate.actionability == ACTIONABILITY_WHEN_STOPPED


def test_an_unanswered_buddy_response_carries_no_confidence():
    assert buddy_candidate(SimpleNamespace(answered=False), "parked").confidence == 0.0


def test_replan_is_shown_soon_but_is_informational():
    candidate = replan_candidate(["rain increased"], "digging")
    assert candidate.event_type == EVENT_REPLAN
    assert candidate.actionability == ACTIONABILITY_INFORMATIONAL
    assert candidate.severity == "MEDIUM"


def test_a_replan_with_no_changes_is_only_informational():
    candidate = replan_candidate([], "safe_idle")
    assert candidate.severity == "INFO"
    assert candidate.confidence == 0.0


def test_a_critical_safety_event_is_immediate_and_actionable_now():
    candidate = safety_candidate(CRITICAL_EVENT)
    assert candidate.event_type == EVENT_SAFETY
    assert candidate.severity == "CRITICAL"
    assert candidate.urgency == URGENCY_IMMEDIATE
    assert candidate.actionability == ACTIONABILITY_NOW
    assert candidate.event_id == "SE00123"


def test_a_low_severity_safety_event_is_not_immediate():
    candidate = safety_candidate({"severity": "LOW", "machine_state": "safe_idle"})
    assert candidate.urgency != URGENCY_IMMEDIATE


def test_demo_test_3_candidates_reach_the_arbiter_correctly_ranked():
    """Critical safety arriving while training waits.

    Safety must be the only immediate, actionable-now candidate; training must
    stay deferrable so the arbiter can hold rather than drop it.
    """
    training = training_candidate(TRIGGER, "digging")
    safety = safety_candidate(CRITICAL_EVENT, "digging")

    assert safety.urgency == URGENCY_IMMEDIATE
    assert safety.actionability == ACTIONABILITY_NOW
    assert training.urgency == URGENCY_DEFERRABLE
    assert training.actionability == ACTIONABILITY_WHEN_STOPPED
    # Neither has pre-empted the other: the arbiter still has both to weigh.
    assert training.decision == safety.decision == DECISION_PENDING


def test_candidates_carry_every_field_the_arbiter_stores():
    candidate = training_candidate(TRIGGER, "safe_idle")
    for field in (
        "event_id", "event_type", "severity", "urgency", "actionability",
        "confidence", "operator_state", "decision", "reason", "timestamp",
    ):
        assert getattr(candidate, field) not in (None, "")
