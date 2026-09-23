"""Measures what a training cycle actually changed.

A trigger is only the start of the loop:

    trigger -> lesson -> quiz -> follow-up measurement -> escalation state

This module scores the quiz, compares a before metric against an after metric,
and advances the escalation state. It never re-decides whether coaching was
warranted -- that is trigger.py's job -- and it never concludes anything about
the operator beyond the two things it can actually measure: what they answered
and what the follow-up sessions recorded.

Metric direction matters and is explicit. Idle ratio, cycle time and fuel are
`lower_is_better=True` (the default); throughput-style metrics pass False.

Division by zero is a real case, not a defensive nicety: a before metric of 0
(zero recorded idle, zero events) has no percentage to improve on. Those
comparisons return `improvement_pct=0.0` with `comparable=False` and the reason
code `REASON_BASELINE_ZERO`, so a dashboard shows "no baseline to compare"
rather than an invented number or a crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from features.training.lessons import Quiz
from features.training.trigger import (
    ESCALATION_ESCALATED,
    ESCALATION_FIRST,
    ESCALATION_REPEAT,
    ESCALATION_RESOLVED,
    ESCALATION_STATES,
)
from shared.config import load_settings
from shared.schemas import TrainingTrigger

DEFAULT_QUIZ_PASS_SCORE = 0.80
DEFAULT_IMPROVEMENT_TARGET_PCT = 10.0

REASON_BASELINE_ZERO = "baseline_metric_is_zero"
REASON_TARGET_MET = "improvement_target_met"
REASON_TARGET_NOT_MET = "improvement_target_not_met"

# `resolved` is terminal: a cycle that measurably worked does not escalate.
_ESCALATION_ORDER = (ESCALATION_FIRST, ESCALATION_REPEAT, ESCALATION_ESCALATED)


def quiz_pass_score(settings: Optional[dict[str, Any]] = None) -> float:
    block = ((settings if settings is not None else load_settings()).get("training") or {})
    return float(block.get("quiz_pass_score", DEFAULT_QUIZ_PASS_SCORE))


def improvement_target_pct(settings: Optional[dict[str, Any]] = None) -> float:
    block = ((settings if settings is not None else load_settings()).get("training") or {})
    return float(block.get("improvement_target_pct", DEFAULT_IMPROVEMENT_TARGET_PCT))


@dataclass(frozen=True)
class AnswerResult:
    question_id: str
    given_answer_id: Optional[str]
    correct_answer_id: str
    correct: bool
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "given_answer_id": self.given_answer_id,
            "correct_answer_id": self.correct_answer_id,
            "correct": self.correct,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class QuizResult:
    """Score for one attempt. Unanswered questions count as incorrect, not absent."""

    quiz_id: str
    correct: int
    total: int
    answered: int
    score: float
    passed: bool
    pass_score: float
    answers: tuple[AnswerResult, ...] = ()
    synthetic_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "quiz_id": self.quiz_id,
            "correct": self.correct,
            "total": self.total,
            "answered": self.answered,
            "score": self.score,
            "score_pct": round(self.score * 100.0, 2),
            "passed": self.passed,
            "pass_score": self.pass_score,
            "answers": [a.to_dict() for a in self.answers],
            "synthetic_flag": self.synthetic_flag,
        }


@dataclass(frozen=True)
class ImprovementResult:
    """Before/after comparison of one metric.

    `improvement_pct` is positive when the metric moved in the desired
    direction, negative when it moved the wrong way, and 0.0 when there is no
    usable baseline (`comparable` is then False).
    """

    metric_name: str
    before: float
    after: float
    lower_is_better: bool
    improvement_pct: float
    improved: bool
    target_pct: float
    target_met: bool
    comparable: bool
    reasons: tuple[str, ...] = ()
    synthetic_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "before": self.before,
            "after": self.after,
            "lower_is_better": self.lower_is_better,
            "improvement_pct": self.improvement_pct,
            "improved": self.improved,
            "target_pct": self.target_pct,
            "target_met": self.target_met,
            "comparable": self.comparable,
            "reasons": list(self.reasons),
            "synthetic_flag": self.synthetic_flag,
        }


@dataclass(frozen=True)
class TrainingOutcome:
    """One completed cycle: what fired, what was scored, what changed next."""

    trigger_event_id: str
    operator_id: str
    issue_type: str
    lesson_id: str
    quiz_result: Optional[QuizResult]
    improvement: Optional[ImprovementResult]
    escalation_state: str
    next_escalation_state: str
    synthetic_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_event_id": self.trigger_event_id,
            "operator_id": self.operator_id,
            "issue_type": self.issue_type,
            "lesson_id": self.lesson_id,
            "quiz_result": self.quiz_result.to_dict() if self.quiz_result else None,
            "improvement": self.improvement.to_dict() if self.improvement else None,
            "escalation_state": self.escalation_state,
            "next_escalation_state": self.next_escalation_state,
            "synthetic_flag": self.synthetic_flag,
        }


def score_quiz(
    quiz: Quiz,
    answers: Mapping[str, Optional[str]],
    settings: Optional[dict[str, Any]] = None,
    pass_score: Optional[float] = None,
) -> QuizResult:
    """Score `answers` (question_id -> chosen option_id) against `quiz`.

    A missing or unknown question_id is scored as incorrect; answers for
    questions that are not in the quiz are ignored rather than credited.
    """
    threshold = pass_score if pass_score is not None else quiz_pass_score(settings)
    answers = answers or {}

    results = []
    for question in quiz.questions:
        given = answers.get(question.question_id)
        results.append(
            AnswerResult(
                question_id=question.question_id,
                given_answer_id=given,
                correct_answer_id=question.answer_id,
                correct=question.is_correct(given),
                explanation=question.explanation,
            )
        )

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    answered = sum(1 for r in results if r.given_answer_id is not None)
    score = (correct / total) if total else 0.0

    return QuizResult(
        quiz_id=quiz.quiz_id,
        correct=correct,
        total=total,
        answered=answered,
        score=round(score, 4),
        passed=bool(total) and score >= threshold,
        pass_score=threshold,
        answers=tuple(results),
        synthetic_flag=quiz.synthetic_flag,
    )


def compare_metrics(
    metric_name: str,
    before: float,
    after: float,
    lower_is_better: bool = True,
    settings: Optional[dict[str, Any]] = None,
    target_pct: Optional[float] = None,
    synthetic_flag: bool = True,
) -> ImprovementResult:
    """Percentage improvement from `before` to `after`, safe when `before` is 0."""
    target = target_pct if target_pct is not None else improvement_target_pct(settings)
    before = float(before)
    after = float(after)

    if before == 0.0:
        return ImprovementResult(
            metric_name=metric_name,
            before=before,
            after=after,
            lower_is_better=lower_is_better,
            improvement_pct=0.0,
            improved=False,
            target_pct=target,
            target_met=False,
            comparable=False,
            reasons=(REASON_BASELINE_ZERO,),
            synthetic_flag=synthetic_flag,
        )

    delta = (before - after) if lower_is_better else (after - before)
    improvement_pct = round(delta / abs(before) * 100.0, 4)
    target_met = improvement_pct >= target
    return ImprovementResult(
        metric_name=metric_name,
        before=before,
        after=after,
        lower_is_better=lower_is_better,
        improvement_pct=improvement_pct,
        improved=improvement_pct > 0.0,
        target_pct=target,
        target_met=target_met,
        comparable=True,
        reasons=(REASON_TARGET_MET,) if target_met else (REASON_TARGET_NOT_MET,),
        synthetic_flag=synthetic_flag,
    )


def next_escalation_state(current_state: str, target_met: bool) -> str:
    """Advance the escalation state after a measured follow-up.

    A cycle whose follow-up met the improvement target ends in `resolved`;
    otherwise the state advances one step and stops at `escalated`. `resolved`
    is terminal -- a new trigger for the same issue starts the ladder again from
    `prior_trigger_count` (see trigger.escalation_state_for).
    """
    if current_state not in ESCALATION_STATES:
        raise ValueError(f"Unknown escalation state: {current_state!r}")
    if target_met:
        return ESCALATION_RESOLVED
    if current_state == ESCALATION_RESOLVED:
        return ESCALATION_RESOLVED
    index = _ESCALATION_ORDER.index(current_state)
    return _ESCALATION_ORDER[min(index + 1, len(_ESCALATION_ORDER) - 1)]


def complete_cycle(
    trigger: TrainingTrigger,
    quiz_result: Optional[QuizResult] = None,
    improvement: Optional[ImprovementResult] = None,
) -> TrainingOutcome:
    """Bundle a finished cycle for storage and display.

    The loop is only closed by measurement: passing the quiz alone does not
    resolve an issue, so `next_escalation_state` keys off the follow-up metric.
    """
    target_met = bool(improvement and improvement.comparable and improvement.target_met)
    return TrainingOutcome(
        trigger_event_id=trigger.trigger_event_id,
        operator_id=trigger.operator_id,
        issue_type=trigger.issue_type,
        lesson_id=trigger.lesson_id,
        quiz_result=quiz_result,
        improvement=improvement,
        escalation_state=trigger.escalation_state,
        next_escalation_state=next_escalation_state(trigger.escalation_state, target_met),
        synthetic_flag=trigger.synthetic_flag,
    )
