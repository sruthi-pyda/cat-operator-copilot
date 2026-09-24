"""Tests for Buddy evidence retrieval and the approved manual (Feature 07).

The manual is the source that makes a safety question answerable at all:
`buddy.ask()` discards every non-safety source once a question is safety-critical,
so these tests check the end-to-end path, not just the router.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from features.buddy.buddy import STATUS_ANSWERED, STATUS_DEFERRED_SAFETY_CRITICAL, ask
from features.buddy.evidence import (
    SOURCE_MACHINE_MANUAL,
    SOURCE_PASSPORT,
    SOURCE_PREDICTION,
    SOURCE_SAFETY_INCIDENT,
    SOURCE_SESSION,
    SOURCE_TELEMETRY,
    SOURCE_TRAINING_STATE,
)
from features.buddy.retrieval import load_manual, match_snippets, retrieve
from features.buddy.safe_state import MachineStateSnapshot

NOW = datetime(2026, 1, 1, 8, 0, 0)
SAFE_STATES = frozenset({"parked", "safe_idle"})
SETTINGS = {"buddy": {"no_attachment_movement_required": True, "max_evidence_age_min": 60}}

PARKED = MachineStateSnapshot(
    machine_state="safe_idle", attachment_movement=False,
    arm_speed=0.0, bucket_state="closed", machine_speed_kmh=0.0,
)

SESSION = SimpleNamespace(
    session_id="S000001",
    operator_id="OP1003",
    task_id="T00301",
    timestamp=NOW.isoformat(),
    operator=SimpleNamespace(machine_skill="advanced", authorization="authorized"),
    task=SimpleNamespace(task_type="grading"),
)

TELEMETRY = {
    "timestamp": NOW.isoformat(),
    "fuel_level_pct": 62.5,
    "machine_state": "safe_idle",
}

CRITICAL_EVENT = {
    "severity": "CRITICAL",
    "trigger_reason": "worker_inside_swing_envelope_with_closing_motion",
    "required_action": "stop_or_safe_action",
    "timestamp": NOW.isoformat(),
    "confidence": 0.94,
    "synthetic_flag": True,
}


# --- manual ------------------------------------------------------------------

def test_the_manual_loads_and_is_marked_synthetic():
    snippets = load_manual()
    assert len(snippets) >= 5
    assert all(s.synthetic_flag for s in snippets)
    assert all(s.snippet_id and s.content for s in snippets)


def test_no_snippet_claims_to_be_real_manufacturer_documentation():
    """Synthetic guidance must never pose as a real machine manual."""
    for snippet in load_manual():
        lowered = snippet.content.lower()
        assert "caterpillar" not in lowered
        assert "cat manual" not in lowered


@pytest.mark.parametrize(
    "question,expected_topic",
    [
        ("is it safe to swing right now", "swing_zone"),
        ("do I need my seatbelt", "seatbelt"),
        ("how do I park the machine", "safe_shutdown"),
        ("what about working on a slope", "slope_travel"),
        ("someone may be injured", "emergency_stop"),
    ],
)
def test_questions_match_the_right_snippet(question, expected_topic):
    topics = {s.topic for s in match_snippets(question)}
    assert expected_topic in topics


def test_an_unrelated_question_matches_nothing():
    assert match_snippets("what is the weather forecast for tuesday") == ()


def test_an_empty_question_matches_nothing():
    assert match_snippets("") == ()


# --- routing -----------------------------------------------------------------

def test_fuel_question_reaches_telemetry():
    evidence = retrieve("how much fuel is left", session_context=SESSION, telemetry_row=TELEMETRY)
    assert any(e.source == SOURCE_TELEMETRY and e.value == 62.5 for e in evidence)


def test_task_question_falls_back_to_the_session_when_no_plan_exists():
    evidence = retrieve("what is my next task", session_context=SESSION)
    assert any(e.source == SOURCE_SESSION and e.value == "T00301" for e in evidence)


def test_operator_question_reaches_the_passport():
    evidence = retrieve("am I certified for this", session_context=SESSION)
    assert any(e.source == SOURCE_PASSPORT for e in evidence)


def test_prediction_is_used_only_when_the_question_is_about_time_or_fuel():
    prediction = SimpleNamespace(eta_p50=31.0, eta_p10=25.0, eta_p90=41.0, confidence=0.8)
    asked = retrieve("how long until I finish", session_context=SESSION, prediction=prediction)
    unrelated = retrieve("am I certified for this", session_context=SESSION, prediction=prediction)
    assert any(e.source == SOURCE_PREDICTION for e in asked)
    assert not any(e.source == SOURCE_PREDICTION for e in unrelated)


def test_training_question_reaches_training_state():
    evidence = retrieve("do I have a lesson outstanding", session_context=SESSION,
                        training_state="idle_reduction assigned")
    assert any(e.source == SOURCE_TRAINING_STATE for e in evidence)


def test_evidence_is_ordered_most_authoritative_first():
    evidence = retrieve(
        "is it safe to swing", session_context=SESSION,
        telemetry_row=TELEMETRY, safety_events=[CRITICAL_EVENT],
    )
    authorities = [e.authority for e in evidence]
    assert authorities == sorted(authorities, reverse=True)
    assert evidence[0].source == SOURCE_SAFETY_INCIDENT


def test_nothing_is_retrieved_from_sources_that_were_not_supplied():
    evidence = retrieve("how much fuel is left")
    assert all(e.source == SOURCE_MACHINE_MANUAL for e in evidence)


# --- end to end through the Buddy --------------------------------------------

def _ask(question, evidence):
    return ask(question, PARKED, evidence, as_of=NOW, settings=SETTINGS, safe_states=SAFE_STATES)


def test_a_safety_question_is_now_answerable_from_the_manual():
    """Before the manual existed this could only ever defer."""
    evidence = retrieve("is it safe to swing right now", session_context=SESSION,
                        telemetry_row=TELEMETRY)
    response = _ask("is it safe to swing right now", evidence)
    assert response.answered is True
    assert response.status == STATUS_ANSWERED
    assert SOURCE_MACHINE_MANUAL in response.answer
    assert "tail swing" in response.answer


def test_a_recorded_incident_outranks_the_manual():
    evidence = retrieve("why did the safety alert trigger", session_context=SESSION,
                        safety_events=[CRITICAL_EVENT])
    response = _ask("why did the safety alert trigger", evidence)
    assert response.answered is True
    assert "worker_inside_swing_envelope_with_closing_motion" in response.answer


def test_a_safety_question_with_no_manual_coverage_still_defers():
    evidence = retrieve("is it dangerous to work near the river", session_context=SESSION,
                        telemetry_row=TELEMETRY)
    response = _ask("is it dangerous to work near the river", evidence)
    assert response.answered is False
    assert response.status == STATUS_DEFERRED_SAFETY_CRITICAL


def test_an_operational_question_is_answered_without_the_manual():
    evidence = retrieve("what is my next task", session_context=SESSION)
    response = _ask("what is my next task", evidence)
    assert response.answered is True
    assert "T00301" in response.answer
