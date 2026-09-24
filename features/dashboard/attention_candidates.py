"""Turns this slice's outputs into Attention Manager candidate events.

Every subsystem may *propose*; only the Attention Manager decides what reaches
the operator. So these functions deliberately stop at proposing: each candidate
carries the severity, urgency, actionability, confidence and operator state the
arbiter needs, and leaves `decision` as `pending` for it to fill in.

Nothing here ranks or suppresses anything. Re-implementing the arbitration would
give the system two competing attention owners, which is exactly what the
architecture forbids.

Priority order the arbiter applies (architecture section 25), for reference:

    1 critical safety   always wins
    2 high safety       interrupt when operator state allows
    3 task replanning   next safe glance
    4 behavior nudge    queue or bundle during active work
    5 training          queue while moving, surface when parked or safe idle
    6 buddy             only when parked or safe idle with no attachment movement
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from shared.schemas import AttentionEvent

EVENT_SAFETY = "safety"
EVENT_REPLAN = "task_replanning"
EVENT_TRAINING = "training"
EVENT_BUDDY = "buddy"

URGENCY_IMMEDIATE = "immediate"
URGENCY_SOON = "soon"
URGENCY_DEFERRABLE = "deferrable"

ACTIONABILITY_NOW = "actionable_now"
ACTIONABILITY_WHEN_STOPPED = "actionable_when_stopped"
ACTIONABILITY_INFORMATIONAL = "informational"

DECISION_PENDING = "pending"
REASON_PENDING = "awaiting_attention_manager_arbitration"

# Machine states in which the operator is working and should not be interrupted
# by anything below high safety.
ACTIVE_STATES = frozenset({"digging", "loading", "swinging", "traveling", "grading"})


def operator_state_from(machine_state: Optional[str]) -> str:
    """`active` while working, `available` when parked or safe idle."""
    if machine_state is None:
        return "unknown"
    return "active" if str(machine_state).strip().lower() in ACTIVE_STATES else "available"


def _event_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().int % 100000:05d}"


def _timestamp(value: Optional[str]) -> str:
    return value or datetime.now().isoformat(timespec="seconds")


def _candidate(
    event_id: str,
    event_type: str,
    severity: str,
    urgency: str,
    actionability: str,
    confidence: float,
    operator_state: str,
    timestamp: Optional[str],
) -> AttentionEvent:
    return AttentionEvent(
        event_id=event_id,
        event_type=event_type,
        severity=severity,
        urgency=urgency,
        actionability=actionability,
        confidence=confidence,
        operator_state=operator_state,
        decision=DECISION_PENDING,
        reason=REASON_PENDING,
        timestamp=_timestamp(timestamp),
    )


def training_candidate(trigger: Any, machine_state: Optional[str] = None) -> AttentionEvent:
    """A fired training trigger, proposed for the operator's attention.

    Training is never urgent: it is deferrable and only actionable once the
    machine is stopped, so the arbiter can hold it through an entire shift
    without losing it.
    """
    return _candidate(
        event_id=_event_id("AE"),
        event_type=EVENT_TRAINING,
        severity="LOW",
        urgency=URGENCY_DEFERRABLE,
        actionability=ACTIONABILITY_WHEN_STOPPED,
        confidence=float(getattr(trigger, "confidence", 0.0) or 0.0),
        operator_state=operator_state_from(machine_state),
        timestamp=None,
    )


def buddy_candidate(response: Any, machine_state: Optional[str] = None) -> AttentionEvent:
    """A Buddy answer waiting to be delivered.

    The Buddy's own safe-state gate already decides whether it may speak; this
    only tells the arbiter that something is waiting and that it can wait.
    """
    return _candidate(
        event_id=_event_id("AE"),
        event_type=EVENT_BUDDY,
        severity="INFO",
        urgency=URGENCY_DEFERRABLE,
        actionability=ACTIONABILITY_WHEN_STOPPED,
        confidence=1.0 if getattr(response, "answered", False) else 0.0,
        operator_state=operator_state_from(machine_state),
        timestamp=None,
    )


def replan_candidate(
    changes: Any, machine_state: Optional[str] = None, timestamp: Optional[str] = None
) -> AttentionEvent:
    """A context change the operator should see at the next safe glance."""
    count = len(changes) if changes is not None else 0
    return _candidate(
        event_id=_event_id("AE"),
        event_type=EVENT_REPLAN,
        severity="MEDIUM" if count else "INFO",
        urgency=URGENCY_SOON,
        actionability=ACTIONABILITY_INFORMATIONAL,
        confidence=1.0 if count else 0.0,
        operator_state=operator_state_from(machine_state),
        timestamp=timestamp,
    )


def safety_candidate(event: dict[str, Any], machine_state: Optional[str] = None) -> AttentionEvent:
    """A candidate built from a recorded safety event.

    This is a dataset row, not a Safety Guardian decision. Live evaluation
    belongs to the Safety Guardian; this exists so the attention queue can be
    exercised end to end before that feature is integrated.
    """
    severity = str(event.get("severity", "INFO")).upper()
    critical = severity in {"CRITICAL", "HIGH"}
    return _candidate(
        event_id=str(event.get("event_id") or _event_id("AE")),
        event_type=EVENT_SAFETY,
        severity=severity,
        urgency=URGENCY_IMMEDIATE if critical else URGENCY_SOON,
        actionability=ACTIONABILITY_NOW if critical else ACTIONABILITY_INFORMATIONAL,
        confidence=float(event.get("confidence", 0.9) or 0.9),
        operator_state=operator_state_from(machine_state or event.get("machine_state")),
        timestamp=event.get("timestamp"),
    )
