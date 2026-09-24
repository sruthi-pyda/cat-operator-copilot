"""Tests for the Grounded AI Operating Buddy (Feature 07).

Covers Tests 4, 5 and 6 from the integration checklist:
  4. machine moving + attachment active -> interaction blocked
  5. machine parked + safe idle + no attachment movement -> Buddy enabled
  6. two evidence sources conflict -> conflict reported, authoritative source
     used, or deferral

Safe states are passed explicitly so these tests pin Buddy behaviour rather than
tracking edits to the Safety Guardian's config file.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from features.buddy.buddy import (
    DEFER_SAFETY_GUARDIAN,
    STATUS_ANSWERED,
    STATUS_BLOCKED_UNSAFE_STATE,
    STATUS_DEFERRED_CONFLICT,
    STATUS_DEFERRED_NO_EVIDENCE,
    STATUS_DEFERRED_SAFETY_CRITICAL,
    STATUS_DEFERRED_STALE_EVIDENCE,
    ask,
    is_safety_critical,
)
from features.buddy.conflict import (
    RESOLUTION_BY_AUTHORITY,
    RESOLUTION_DEFER,
    RESOLUTION_NO_CONFLICT,
    detect_conflict,
)
from features.buddy.evidence import (
    SOURCE_MACHINE_MANUAL,
    SOURCE_PREDICTION,
    SOURCE_SAFETY_INCIDENT,
    SOURCE_TASK_PLAN,
    SOURCE_TELEMETRY,
    Evidence,
    filter_approved,
)
from features.buddy.safe_state import (
    REASON_ATTACHMENT_MOVING,
    REASON_ATTACHMENT_UNKNOWN,
    REASON_BUCKET_MOVING,
    REASON_MACHINE_MOVING,
    REASON_STATE_NOT_SAFE,
    REASON_STATE_UNKNOWN,
    MachineStateSnapshot,
    evaluate_safe_state,
    load_safe_states,
)

NOW = datetime(2026, 9, 23, 10, 0, 0)
SAFE_STATES = frozenset({"parked", "safe_idle"})
SETTINGS = {"buddy": {"no_attachment_movement_required": True, "max_evidence_age_min": 60}}

PARKED = MachineStateSnapshot(
    machine_state="parked",
    attachment_movement="none",
    arm_speed=0.0,
    bucket_state="idle",
    machine_speed_kmh=0.0,
)


def _evidence(source, value, content="", minutes_old=1, confidence=0.9):
    return Evidence(
        source=source,
        value=value,
        content=content or f"{source} says {value}",
        timestamp=(NOW - timedelta(minutes=minutes_old)).isoformat(),
        synthetic_flag=True,
        confidence=confidence,
    )


def _gate(snapshot):
    return evaluate_safe_state(snapshot, safe_states=SAFE_STATES, settings=SETTINGS)


# --- safe-state gate ---------------------------------------------------------

def test_5_parked_safe_idle_no_attachment_movement_enables_buddy():
    assert _gate(PARKED).allowed is True
    safe_idle = MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement="idle", arm_speed=0.0,
        bucket_state="idle", machine_speed_kmh=0.0,
    )
    assert _gate(safe_idle).allowed is True


def test_4_moving_machine_with_active_attachment_is_blocked():
    moving = MachineStateSnapshot(
        machine_state="traveling", attachment_movement="swinging", arm_speed=0.6,
        bucket_state="carrying", machine_speed_kmh=7.5,
    )
    result = _gate(moving)
    assert result.allowed is False
    assert REASON_STATE_NOT_SAFE in result.reasons
    assert REASON_ATTACHMENT_MOVING in result.reasons
    assert REASON_MACHINE_MOVING in result.reasons


def test_stationary_machine_with_swinging_arm_is_still_blocked():
    """'Not travelling' is explicitly not sufficient."""
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement="swinging",
        arm_speed=0.4, bucket_state="idle", machine_speed_kmh=0.0,
    ))
    assert result.allowed is False
    assert REASON_ATTACHMENT_MOVING in result.reasons


def test_unknown_state_denies_rather_than_assuming_safe():
    unknown = MachineStateSnapshot()
    result = _gate(unknown)
    assert result.allowed is False
    assert REASON_STATE_UNKNOWN in result.reasons
    assert REASON_ATTACHMENT_UNKNOWN in result.reasons


def test_safe_states_come_from_the_safety_guardian_config():
    """The Buddy must not keep a private definition of 'safe'."""
    assert load_safe_states() == SAFE_STATES


# --- real telemetry vocabulary -----------------------------------------------
# The generated telemetry reports attachment_movement as a boolean and
# bucket_state as a configuration (closed / open / loading). These pin that
# shape: an earlier version read False as "moving" and the Buddy never enabled.

@pytest.mark.parametrize("moving_value", [False, "False", "false", 0, "no"])
def test_boolean_attachment_movement_false_means_stationary(moving_value):
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement=moving_value,
        arm_speed=0.0, bucket_state="closed", machine_speed_kmh=0.0,
    ))
    assert result.allowed is True


@pytest.mark.parametrize("moving_value", [True, "True", "true", 1])
def test_boolean_attachment_movement_true_blocks(moving_value):
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement=moving_value,
        arm_speed=0.0, bucket_state="closed", machine_speed_kmh=0.0,
    ))
    assert result.allowed is False
    assert REASON_ATTACHMENT_MOVING in result.reasons


@pytest.mark.parametrize("bucket_state", ["closed", "open", "loading"])
def test_bucket_configuration_is_not_treated_as_motion(bucket_state):
    """closed/open/loading describe the bucket, not movement of it."""
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement=False,
        arm_speed=0.0, bucket_state=bucket_state, machine_speed_kmh=0.0,
    ))
    assert result.allowed is True


@pytest.mark.parametrize("bucket_state", ["digging", "dumping", "swinging"])
def test_bucket_in_actual_motion_still_blocks(bucket_state):
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement=False,
        arm_speed=0.0, bucket_state=bucket_state, machine_speed_kmh=0.0,
    ))
    assert result.allowed is False
    assert REASON_BUCKET_MOVING in result.reasons


def test_a_real_safe_idle_telemetry_row_enables_the_buddy():
    """Exact field shape of a safe_idle row in data/synthetic/telemetry.csv."""
    result = _gate(MachineStateSnapshot(
        machine_state="safe_idle", attachment_movement=False,
        arm_speed=0.0, bucket_state="loading", machine_speed_kmh=0.0,
    ))
    assert result.allowed is True, result.reasons


def test_a_real_digging_telemetry_row_blocks_the_buddy():
    result = _gate(MachineStateSnapshot(
        machine_state="digging", attachment_movement=True,
        arm_speed=0.7, bucket_state="loading", machine_speed_kmh=0.0,
    ))
    assert result.allowed is False


# --- conflict ----------------------------------------------------------------

def test_agreeing_sources_are_not_a_conflict():
    result = detect_conflict([
        _evidence(SOURCE_TELEMETRY, 42.0), _evidence(SOURCE_PREDICTION, 42.0),
    ])
    assert result.conflicted is False
    assert result.resolution == RESOLUTION_NO_CONFLICT


def test_6_conflict_is_resolved_by_the_authoritative_source():
    result = detect_conflict([
        _evidence(SOURCE_PREDICTION, 12.0), _evidence(SOURCE_TELEMETRY, 48.0),
    ])
    assert result.conflicted is True
    assert result.resolution == RESOLUTION_BY_AUTHORITY
    assert result.winner.source == SOURCE_TELEMETRY


def test_6_equal_authority_conflict_defers_instead_of_guessing():
    result = detect_conflict([
        Evidence(source=SOURCE_TELEMETRY, value=10.0, confidence=0.9),
        Evidence(source=SOURCE_TELEMETRY, value=90.0, confidence=0.9),
    ])
    assert result.conflicted is True
    assert result.resolution == RESOLUTION_DEFER
    assert result.winner is None


def test_numeric_tolerance_does_not_flag_rounding_as_conflict():
    result = detect_conflict([
        _evidence(SOURCE_TELEMETRY, 100.0), _evidence(SOURCE_TASK_PLAN, 100.5),
    ])
    assert result.conflicted is False


def test_untrusted_sources_are_dropped_before_answering():
    items = [_evidence(SOURCE_TELEMETRY, 1), Evidence(source="random_blog", value=1)]
    assert tuple(e.source for e in filter_approved(items)) == (SOURCE_TELEMETRY,)


# --- end-to-end --------------------------------------------------------------

def _ask(question, snapshot=PARKED, evidence=()):
    return ask(question, snapshot, evidence, as_of=NOW, settings=SETTINGS, safe_states=SAFE_STATES)


def test_blocked_state_returns_no_answer_at_all():
    response = _ask("what is my fuel level", MachineStateSnapshot(
        machine_state="digging", attachment_movement="lifting", machine_speed_kmh=3.0,
    ), [_evidence(SOURCE_TELEMETRY, 60.0)])
    assert response.answered is False
    assert response.status == STATUS_BLOCKED_UNSAFE_STATE
    assert response.answer is None


def test_answers_from_trusted_evidence_when_parked():
    response = _ask("what is my fuel level", evidence=[
        _evidence(SOURCE_TELEMETRY, 60.0, content="Fuel level is 60 percent."),
    ])
    assert response.answered is True
    assert response.status == STATUS_ANSWERED
    assert "60" in response.answer
    assert SOURCE_TELEMETRY in response.answer


def test_answer_cites_its_source_and_never_stands_alone():
    response = _ask("what is my next task", evidence=[
        _evidence(SOURCE_TASK_PLAN, "T00042", content="Next task is T00042."),
    ])
    assert "source: " + SOURCE_TASK_PLAN in response.answer
    assert response.evidence


def test_no_trusted_evidence_defers_instead_of_inventing_an_answer():
    response = _ask("what is my next task", evidence=[])
    assert response.answered is False
    assert response.status == STATUS_DEFERRED_NO_EVIDENCE
    assert response.answer is None


def test_safety_question_defers_when_only_non_safety_sources_are_available():
    response = _ask("is it safe to swing right now", evidence=[
        _evidence(SOURCE_PREDICTION, "probably", content="Model suggests it is fine."),
        _evidence(SOURCE_TELEMETRY, "clear"),
    ])
    assert response.answered is False
    assert response.status == STATUS_DEFERRED_SAFETY_CRITICAL
    assert response.deferral_target == DEFER_SAFETY_GUARDIAN


def test_safety_question_may_be_answered_from_the_safety_guardian():
    """The Buddy may *explain* a safety event -- it just may not decide one."""
    response = _ask("why did the safety alert trigger", evidence=[
        _evidence(SOURCE_SAFETY_INCIDENT, "HIGH",
                  content="Worker entered the swing envelope with closing motion."),
    ])
    assert response.answered is True
    assert "swing envelope" in response.answer


def test_manual_outranks_prediction_on_a_safety_question():
    response = _ask("what is the safe stopping procedure", evidence=[
        _evidence(SOURCE_MACHINE_MANUAL, "lower_and_park",
                  content="Lower the attachment, neutralise controls, then park."),
        _evidence(SOURCE_PREDICTION, "keep_going", content="Continue the cycle."),
    ])
    assert response.answered is True
    assert SOURCE_MACHINE_MANUAL in response.answer
    assert "Continue the cycle" not in response.answer


def test_6_unresolvable_conflict_is_reported_and_deferred():
    response = _ask("what is my next task", evidence=[
        Evidence(source=SOURCE_TASK_PLAN, value="T1", content="Next is T1",
                 timestamp=NOW.isoformat(), confidence=0.8),
        Evidence(source=SOURCE_TASK_PLAN, value="T2", content="Next is T2",
                 timestamp=NOW.isoformat(), confidence=0.8),
    ])
    assert response.answered is False
    assert response.status == STATUS_DEFERRED_CONFLICT
    assert response.conflict.conflicted is True


def test_resolved_conflict_still_states_the_disagreement():
    response = _ask("how much fuel is left", evidence=[
        _evidence(SOURCE_PREDICTION, 10.0, content="About 10 percent."),
        _evidence(SOURCE_TELEMETRY, 55.0, content="Fuel level is 55 percent."),
    ])
    assert response.answered is True
    assert "disagree" in response.answer.lower()
    assert "55" in response.answer


def test_stale_evidence_defers():
    response = _ask("what is my next task", evidence=[
        _evidence(SOURCE_TASK_PLAN, "T1", minutes_old=600),
    ])
    assert response.answered is False
    assert response.status == STATUS_DEFERRED_STALE_EVIDENCE


def test_undated_evidence_is_usable_but_reported_as_unknown_freshness():
    response = _ask("what is my next task", evidence=[
        Evidence(source=SOURCE_TASK_PLAN, value="T1", content="Next is T1", confidence=0.7),
    ])
    assert response.answered is True
    payload = response.to_dict(as_of=NOW)
    assert payload["evidence"][0]["freshness"] == "unknown"


@pytest.mark.parametrize("question", [
    "is it safe to swing", "any danger nearby", "where is the worker",
    "could this injure someone", "why did the incident happen", "seatbelt rules",
])
def test_safety_questions_are_recognised(question):
    assert is_safety_critical(question) is True


@pytest.mark.parametrize("question", [
    "what is my next task", "how much fuel is left", "what is my eta",
])
def test_operational_questions_are_not_treated_as_safety_critical(question):
    assert is_safety_critical(question) is False


def test_response_payload_marks_data_as_synthetic():
    response = _ask("what is my next task", evidence=[_evidence(SOURCE_TASK_PLAN, "T1")])
    assert response.to_dict(as_of=NOW)["synthetic_flag"] is True
