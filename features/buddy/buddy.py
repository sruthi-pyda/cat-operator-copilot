"""Grounded AI Operating Buddy (Feature 07).

    question
      -> safe-state gate
      -> retrieve evidence
      -> source authority check
      -> freshness check
      -> conflict check
      -> answer OR safe deferral

The Buddy never makes a safety decision and never invents an operating
instruction. An answer is assembled from the text a trusted source already
carries; if no trusted source says it, the Buddy defers instead of filling the
gap. Questions that are safety-critical may only be answered from a Safety
Guardian incident or an approved manual snippet -- every other source defers to
the Safety Guardian.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from features.buddy.conflict import (
    RESOLUTION_BY_AUTHORITY,
    RESOLUTION_DEFER,
    ConflictResult,
    detect_conflict,
)
from features.buddy.evidence import (
    SOURCE_MACHINE_MANUAL,
    SOURCE_SAFETY_INCIDENT,
    Evidence,
    filter_approved,
    max_evidence_age_min,
    partition_by_freshness,
)
from features.buddy.safe_state import MachineStateSnapshot, SafeStateResult, evaluate_safe_state

STATUS_ANSWERED = "answered"
STATUS_BLOCKED_UNSAFE_STATE = "blocked_unsafe_state"
STATUS_DEFERRED_NO_EVIDENCE = "deferred_no_evidence"
STATUS_DEFERRED_STALE_EVIDENCE = "deferred_stale_evidence"
STATUS_DEFERRED_CONFLICT = "deferred_conflict"
STATUS_DEFERRED_SAFETY_CRITICAL = "deferred_safety_critical"

DEFER_SAFETY_GUARDIAN = "safety_guardian"
DEFER_APPROVED_MANUAL = "approved_manual"

# Deliberately broad. A false positive costs one deferral; a false negative would
# let the Buddy answer a safety question on its own authority.
SAFETY_KEYWORDS = (
    "safe", "safety", "danger", "hazard", "risk", "worker", "proximity", "swing",
    "emergency", "stop", "injur", "accident", "incident", "collision", "seatbelt",
    "evacuat", "rollover", "tip over", "blind spot", "lockout",
)

AUTHORITATIVE_SAFETY_SOURCES = frozenset({SOURCE_SAFETY_INCIDENT, SOURCE_MACHINE_MANUAL})


@dataclass(frozen=True)
class BuddyResponse:
    answered: bool
    status: str
    reason: str
    answer: Optional[str] = None
    safe_state: Optional[SafeStateResult] = None
    evidence: tuple[Evidence, ...] = ()
    conflict: Optional[ConflictResult] = None
    deferral_target: Optional[str] = None
    synthetic_flag: bool = True

    def to_dict(self, as_of: Optional[datetime] = None, max_age_min: float = 60.0) -> dict[str, Any]:
        as_of = as_of or datetime.now()
        return {
            "answered": self.answered,
            "status": self.status,
            "reason": self.reason,
            "answer": self.answer,
            "safe_state": self.safe_state.to_dict() if self.safe_state else None,
            "evidence": [item.to_dict(as_of, max_age_min) for item in self.evidence],
            "conflict": self.conflict.to_dict() if self.conflict else None,
            "deferral_target": self.deferral_target,
            "synthetic_flag": self.synthetic_flag,
        }


def is_safety_critical(question: str) -> bool:
    text = (question or "").lower()
    return any(keyword in text for keyword in SAFETY_KEYWORDS)


def _compose_answer(winner: Evidence, conflict: ConflictResult) -> str:
    """Answers are quoted from evidence, never generated."""
    body = winner.content or str(winner.value)
    answer = f"{body} (source: {winner.source}"
    if winner.timestamp:
        answer += f", recorded {winner.timestamp}"
    answer += ")"
    if conflict.conflicted:
        others = ", ".join(s for s in conflict.sources if s != winner.source)
        answer = (
            f"Sources disagree ({others} report differently); "
            f"using the authoritative source. {answer}"
        )
    return answer


def ask(
    question: str,
    machine_state: MachineStateSnapshot,
    evidence: Sequence[Evidence],
    as_of: Optional[datetime] = None,
    settings: Optional[dict] = None,
    safe_states: Optional[frozenset[str]] = None,
) -> BuddyResponse:
    as_of = as_of or datetime.now()
    max_age = max_evidence_age_min(settings)

    gate = evaluate_safe_state(machine_state, safe_states=safe_states, settings=settings)
    if not gate.allowed:
        return BuddyResponse(
            answered=False,
            status=STATUS_BLOCKED_UNSAFE_STATE,
            reason="Buddy interaction is only available when the machine is parked "
            "or in verified safe idle with no attachment movement.",
            safe_state=gate,
            synthetic_flag=True,
        )

    approved = filter_approved(evidence)
    safety_critical = is_safety_critical(question)
    if safety_critical:
        approved = tuple(e for e in approved if e.source in AUTHORITATIVE_SAFETY_SOURCES)

    if not approved:
        return BuddyResponse(
            answered=False,
            status=STATUS_DEFERRED_SAFETY_CRITICAL if safety_critical else STATUS_DEFERRED_NO_EVIDENCE,
            reason="Safety-critical question with no Safety Guardian or approved manual "
            "evidence; deferring." if safety_critical
            else "No trusted source covers this question.",
            safe_state=gate,
            deferral_target=DEFER_SAFETY_GUARDIAN if safety_critical else None,
            synthetic_flag=True,
        )

    usable, stale = partition_by_freshness(approved, as_of, max_age)
    if not usable:
        return BuddyResponse(
            answered=False,
            status=STATUS_DEFERRED_STALE_EVIDENCE,
            reason=f"All supporting evidence is older than {max_age:g} minutes.",
            safe_state=gate,
            evidence=stale,
            deferral_target=DEFER_SAFETY_GUARDIAN if safety_critical else None,
            synthetic_flag=any(e.synthetic_flag for e in stale),
        )

    conflict = detect_conflict(usable)
    synthetic = any(e.synthetic_flag for e in usable)

    if conflict.conflicted and conflict.resolution == RESOLUTION_DEFER:
        return BuddyResponse(
            answered=False,
            status=STATUS_DEFERRED_CONFLICT,
            reason="Trusted sources disagree and none outranks the other.",
            safe_state=gate,
            evidence=usable,
            conflict=conflict,
            deferral_target=DEFER_APPROVED_MANUAL if safety_critical else DEFER_SAFETY_GUARDIAN,
            synthetic_flag=synthetic,
        )

    winner = conflict.winner
    return BuddyResponse(
        answered=True,
        status=STATUS_ANSWERED,
        reason="Answered from the highest-authority fresh evidence."
        if conflict.resolution == RESOLUTION_BY_AUTHORITY
        else "Sources agree.",
        answer=_compose_answer(winner, conflict),
        safe_state=gate,
        evidence=usable,
        conflict=conflict,
        synthetic_flag=synthetic,
    )
