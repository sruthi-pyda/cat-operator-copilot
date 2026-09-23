"""Decides whether an operator's behaviour history justifies coaching.

This is the gate in the Safety -> Behavior -> Training -> updated context loop,
and its job is mostly to say *no*. Training must not fire on every anomaly: an
operator working difficult ground is not an operator who needs a lesson.

Three conditions must all hold before a trigger is produced:

  1. REPEATED        the issue occurs at least `min_occurrences` times inside the
                     recent history window -- a one-off is never coached;
  2. CONFIDENT       those occurrences carry attribution confidence at or above
                     `min_confidence`;
  3. OPERATOR-LINKED the attribution is operator-driven or mixed, and the share
                     of the gap explained by site/machine/environment context is
                     at or below `max_context_share`.

Condition 3 is the one that protects the operator. `context_explained_component`
is how much of the observed-versus-expected gap the context already accounts
for; when it dominates, the finding belongs to planning, not to coaching.

All thresholds come from the `training:` block of config/settings.yaml. Nothing
here decides that an operator is a poor operator -- the only claim available is
that observed behaviour differs from what was expected under the current
operating context, repeatedly, in a way the context does not explain.

Worked example:

    >>> history = [op_driven(0.82), op_driven(0.79)]      # both low context share
    >>> decision = evaluate_training_gate(history, operator_id="OP1001",
    ...                                   issue_type="idle_reduction")
    >>> decision.triggered, decision.reasons
    (True, ())

Swap either result for a context-driven one and the gate returns
`triggered=False` with `REASON_CONTEXT_DRIVEN_ATTRIBUTION` -- inspectable by the
dashboard, which shows the operator *why* nothing fired.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from features.training.lessons import TrainingContent, load_content
from shared.config import load_settings
from shared.schemas import BehaviorResult, TrainingTrigger

# Attributions that can be coached at all. "mixed" is included because part of
# the gap is operator-linked; the context-share check then decides how much.
COACHABLE_ATTRIBUTIONS = frozenset({"operator-driven", "mixed"})
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Reason codes. Stable strings; the dashboard displays them, so do not rename.
REASON_NO_HISTORY = "no_behavior_history"
REASON_INSUFFICIENT_EVIDENCE = "attribution_insufficient_evidence"
REASON_NOT_REPEATED = "issue_not_repeated_enough"
REASON_LOW_CONFIDENCE = "attribution_confidence_below_threshold"
REASON_CONTEXT_DRIVEN_ATTRIBUTION = "attribution_not_operator_linked"
REASON_CONTEXT_EXPLAINED = "gap_primarily_explained_by_context"
REASON_NO_LESSON_FOR_ISSUE = "no_lesson_mapped_for_issue_type"

# Emitted in this order so the dashboard always renders reasons consistently.
_REASON_ORDER = (
    REASON_NO_HISTORY,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_CONTEXT_DRIVEN_ATTRIBUTION,
    REASON_CONTEXT_EXPLAINED,
    REASON_LOW_CONFIDENCE,
    REASON_NOT_REPEATED,
    REASON_NO_LESSON_FOR_ISSUE,
)

# Escalation states, in order. `prior_trigger_count` is how many times this
# operator has already been triggered for this same issue_type.
#   first_trigger  no previous trigger for this issue        -> lesson offered
#   repeat         triggered before, issue came back         -> lesson + supervisor visibility
#   escalated      triggered `escalate_after_triggers` times -> in-person coaching
#   resolved       set by progress.py when the follow-up measurement improves
ESCALATION_FIRST = "first_trigger"
ESCALATION_REPEAT = "repeat"
ESCALATION_ESCALATED = "escalated"
ESCALATION_RESOLVED = "resolved"
ESCALATION_STATES = (ESCALATION_FIRST, ESCALATION_REPEAT, ESCALATION_ESCALATED, ESCALATION_RESOLVED)

_DEFAULT_THRESHOLDS = {
    "history_window": 10,
    "min_occurrences": 2,
    "min_confidence": 0.70,
    "max_context_share": 0.50,
    "escalate_after_triggers": 2,
}


@dataclass(frozen=True)
class TrainingThresholds:
    """The `training:` block of settings.yaml, with defensive defaults."""

    history_window: int = _DEFAULT_THRESHOLDS["history_window"]
    min_occurrences: int = _DEFAULT_THRESHOLDS["min_occurrences"]
    min_confidence: float = _DEFAULT_THRESHOLDS["min_confidence"]
    max_context_share: float = _DEFAULT_THRESHOLDS["max_context_share"]
    escalate_after_triggers: int = _DEFAULT_THRESHOLDS["escalate_after_triggers"]

    @classmethod
    def from_settings(cls, settings: Optional[dict[str, Any]] = None) -> "TrainingThresholds":
        block = ((settings if settings is not None else load_settings()).get("training") or {})
        return cls(
            history_window=int(block.get("history_window", _DEFAULT_THRESHOLDS["history_window"])),
            min_occurrences=int(block.get("min_occurrences", _DEFAULT_THRESHOLDS["min_occurrences"])),
            min_confidence=float(block.get("min_confidence", _DEFAULT_THRESHOLDS["min_confidence"])),
            max_context_share=float(
                block.get("max_context_share", _DEFAULT_THRESHOLDS["max_context_share"])
            ),
            escalate_after_triggers=int(
                block.get("escalate_after_triggers", _DEFAULT_THRESHOLDS["escalate_after_triggers"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_window": self.history_window,
            "min_occurrences": self.min_occurrences,
            "min_confidence": self.min_confidence,
            "max_context_share": self.max_context_share,
            "escalate_after_triggers": self.escalate_after_triggers,
        }


@dataclass(frozen=True)
class OccurrenceCheck:
    """Why one BehaviorResult did or did not count towards the gate."""

    session_id: str
    attribution: str
    confidence: float
    context_share: float
    qualifies: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attribution": self.attribution,
            "confidence": self.confidence,
            "context_share": self.context_share,
            "qualifies": self.qualifies,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class TriggerDecision:
    """Full, inspectable outcome of the gate -- including when nothing fired.

    `trigger` is None whenever `triggered` is False.
    """

    triggered: bool
    operator_id: str
    issue_type: str
    reasons: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)
    occurrences: tuple[OccurrenceCheck, ...] = ()
    qualifying_occurrences: int = 0
    considered: int = 0
    thresholds: TrainingThresholds = TrainingThresholds()
    trigger: Optional[TrainingTrigger] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "operator_id": self.operator_id,
            "issue_type": self.issue_type,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "occurrences": [o.to_dict() for o in self.occurrences],
            "qualifying_occurrences": self.qualifying_occurrences,
            "considered": self.considered,
            "thresholds": self.thresholds.to_dict(),
            "trigger": _trigger_to_dict(self.trigger),
        }


def _trigger_to_dict(trigger: Optional[TrainingTrigger]) -> Optional[dict[str, Any]]:
    if trigger is None:
        return None
    return {
        "trigger_event_id": trigger.trigger_event_id,
        "operator_id": trigger.operator_id,
        "issue_type": trigger.issue_type,
        "attribution_type": trigger.attribution_type,
        "confidence": trigger.confidence,
        "lesson_id": trigger.lesson_id,
        "escalation_state": trigger.escalation_state,
        "synthetic_flag": trigger.synthetic_flag,
    }


def context_share(result: BehaviorResult) -> float:
    """Share of the observed-versus-expected gap that context already explains.

    Magnitudes are used, so a residual of either sign counts as operator-linked
    signal. When neither component carries any magnitude there is nothing to
    attribute to the operator, so the share is 1.0 -- the gate fails closed, in
    the operator's favour.
    """
    operator_part = abs(result.operator_residual or 0.0)
    context_part = abs(result.context_explained_component or 0.0)
    total = operator_part + context_part
    if total <= 0.0:
        return 1.0
    return context_part / total


def escalation_state_for(prior_trigger_count: int, thresholds: TrainingThresholds) -> str:
    """first_trigger -> repeat -> escalated, by how often this issue already fired."""
    if prior_trigger_count <= 0:
        return ESCALATION_FIRST
    if prior_trigger_count < max(1, thresholds.escalate_after_triggers):
        return ESCALATION_REPEAT
    return ESCALATION_ESCALATED


def generate_trigger_event_id() -> str:
    return f"TR{uuid.uuid4().int % 100_000:05d}"


def _check_occurrence(result: BehaviorResult, thresholds: TrainingThresholds) -> OccurrenceCheck:
    reasons: list[str] = []
    share = context_share(result)
    attribution = result.attribution

    if attribution == INSUFFICIENT_EVIDENCE:
        reasons.append(REASON_INSUFFICIENT_EVIDENCE)
    elif attribution not in COACHABLE_ATTRIBUTIONS:
        reasons.append(REASON_CONTEXT_DRIVEN_ATTRIBUTION)
    if (result.confidence or 0.0) < thresholds.min_confidence:
        reasons.append(REASON_LOW_CONFIDENCE)
    if share > thresholds.max_context_share:
        reasons.append(REASON_CONTEXT_EXPLAINED)

    return OccurrenceCheck(
        session_id=result.session_id,
        attribution=attribution,
        confidence=float(result.confidence or 0.0),
        context_share=round(share, 4),
        qualifies=not reasons,
        reasons=tuple(reasons),
    )


def _attribution_type(qualifying: Sequence[OccurrenceCheck]) -> str:
    """operator-driven only when every qualifying occurrence was; otherwise mixed."""
    if qualifying and all(o.attribution == "operator-driven" for o in qualifying):
        return "operator-driven"
    return "mixed"


def evaluate_training_gate(
    behavior_history: Iterable[BehaviorResult],
    operator_id: str,
    issue_type: str,
    prior_trigger_count: int = 0,
    settings: Optional[dict[str, Any]] = None,
    content: Optional[TrainingContent] = None,
    trigger_event_id: Optional[str] = None,
) -> TriggerDecision:
    """Run the three-part gate and report the outcome with its reasons.

    `behavior_history` is the sequence of BehaviorResult for one operator and one
    issue_type, oldest first; only the most recent `history_window` are weighed.
    """
    thresholds = TrainingThresholds.from_settings(settings)
    history = list(behavior_history or [])
    window = history[-thresholds.history_window:] if thresholds.history_window > 0 else history

    occurrences = tuple(_check_occurrence(result, thresholds) for result in window)
    qualifying = tuple(o for o in occurrences if o.qualifies)
    usable = tuple(o for o in occurrences if o.attribution != INSUFFICIENT_EVIDENCE)

    # Each check counts, over the window, how many occurrences satisfy that one
    # part of the gate. They are reported separately so the dashboard can say
    # which part held, but only `gate_met` -- occurrences satisfying all parts at
    # once -- produces a trigger.
    needed = thresholds.min_occurrences
    checks = {
        "history_present": bool(window),
        "evidence_usable": bool(usable),
        "issue_repeated": len(usable) >= needed,
        "confidence_sufficient": sum(o.confidence >= thresholds.min_confidence for o in usable) >= needed,
        "operator_linked": sum(o.attribution in COACHABLE_ATTRIBUTIONS for o in usable) >= needed,
        "context_not_dominant": sum(
            o.context_share <= thresholds.max_context_share for o in usable
        ) >= needed,
        "gate_met": len(qualifying) >= needed,
    }

    reasons: set[str] = set()
    if not window:
        reasons.add(REASON_NO_HISTORY)
    elif not usable:
        reasons.add(REASON_INSUFFICIENT_EVIDENCE)
    elif len(qualifying) < thresholds.min_occurrences:
        # Report every distinct blocker seen in the window, then say that too few
        # occurrences survived. Both are true and both are useful to display.
        for occurrence in occurrences:
            reasons.update(occurrence.reasons)
        reasons.add(REASON_NOT_REPEATED)

    lesson_id: Optional[str] = None
    if not reasons:
        lesson_id = (content or load_content()).lesson_id_for_issue(issue_type)
        if lesson_id is None:
            reasons.add(REASON_NO_LESSON_FOR_ISSUE)
    checks["lesson_available"] = lesson_id is not None

    ordered = tuple(r for r in _REASON_ORDER if r in reasons)
    if reasons or lesson_id is None:
        return TriggerDecision(
            triggered=False,
            operator_id=operator_id,
            issue_type=issue_type,
            reasons=ordered,
            checks=checks,
            occurrences=occurrences,
            qualifying_occurrences=len(qualifying),
            considered=len(window),
            thresholds=thresholds,
        )

    mean_confidence = sum(o.confidence for o in qualifying) / len(qualifying)
    trigger = TrainingTrigger(
        trigger_event_id=trigger_event_id or generate_trigger_event_id(),
        operator_id=operator_id,
        issue_type=issue_type,
        attribution_type=_attribution_type(qualifying),
        confidence=round(mean_confidence, 4),
        lesson_id=lesson_id,
        escalation_state=escalation_state_for(prior_trigger_count, thresholds),
        # Any synthetic input makes the trigger synthetic; this demo is all synthetic.
        synthetic_flag=any(r.synthetic_flag for r in window),
    )
    return TriggerDecision(
        triggered=True,
        operator_id=operator_id,
        issue_type=issue_type,
        reasons=(),
        checks=checks,
        occurrences=occurrences,
        qualifying_occurrences=len(qualifying),
        considered=len(window),
        thresholds=thresholds,
        trigger=trigger,
    )


def check_training_trigger(
    behavior_history: Iterable[BehaviorResult],
    operator_id: str,
    issue_type: str,
    prior_trigger_count: int = 0,
    settings: Optional[dict[str, Any]] = None,
    content: Optional[TrainingContent] = None,
    trigger_event_id: Optional[str] = None,
) -> Optional[TrainingTrigger]:
    """The API-contract entry point: a TrainingTrigger, or None when the gate holds.

    Use `evaluate_training_gate` when the reasons matter (they do, on screen).
    """
    return evaluate_training_gate(
        behavior_history,
        operator_id=operator_id,
        issue_type=issue_type,
        prior_trigger_count=prior_trigger_count,
        settings=settings,
        content=content,
        trigger_event_id=trigger_event_id,
    ).trigger
