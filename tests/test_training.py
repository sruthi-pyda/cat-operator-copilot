"""Tests for the Operator Training Hub (Feature 06).

The gate is the point of the feature, so most of these tests are about training
*not* firing. `behavior()` builds a result that would pass the gate, and each
test breaks exactly one condition, so a test that fails tells you which part of
the gate moved.

Thresholds come from config/settings.yaml. `THRESHOLDS` below is read from the
real file rather than restated, so a config change surfaces as a failure here
instead of silently changing production behaviour.
"""
from __future__ import annotations

import pytest

from features.training.lessons import (
    ContentError,
    MAX_QUIZ_QUESTIONS,
    MIN_QUIZ_QUESTIONS,
    clear_content_cache,
    lesson_id_for_issue,
    load_content,
)
from features.training.progress import (
    REASON_BASELINE_ZERO,
    ImprovementResult,
    compare_metrics,
    complete_cycle,
    next_escalation_state,
    score_quiz,
)
from features.training.trigger import (
    COACHABLE_ATTRIBUTIONS,
    ESCALATION_ESCALATED,
    ESCALATION_FIRST,
    ESCALATION_REPEAT,
    ESCALATION_RESOLVED,
    REASON_CONTEXT_DRIVEN_ATTRIBUTION,
    REASON_CONTEXT_EXPLAINED,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_LOW_CONFIDENCE,
    REASON_NOT_REPEATED,
    REASON_NO_HISTORY,
    REASON_NO_LESSON_FOR_ISSUE,
    REASON_RESIDUAL_FAVOURABLE,
    TrainingThresholds,
    check_training_trigger,
    context_share,
    escalation_state_for,
    evaluate_training_gate,
)
from shared.constants import ATTRIBUTION_TYPES
from shared.schemas import BehaviorResult

THRESHOLDS = TrainingThresholds.from_settings()
ISSUE = "idle_reduction"
OPERATOR = "OP1001"


@pytest.fixture
def content():
    return load_content()


def behavior(
    attribution: str = "operator-driven",
    confidence: float = 0.85,
    operator_residual: float = 0.09,
    context_explained_component: float = 0.01,
    session_id: str = "S000001",
) -> BehaviorResult:
    """A result that passes every part of the gate unless a caller breaks one."""
    return BehaviorResult(
        session_id=session_id,
        attribution=attribution,
        confidence=confidence,
        observed_value=0.26,
        expected_value=0.16,
        operator_residual=operator_residual,
        context_explained_component=context_explained_component,
        coaching_eligible=attribution in COACHABLE_ATTRIBUTIONS,
        synthetic_flag=True,
    )


def repeated(count: int, **kwargs) -> list[BehaviorResult]:
    return [behavior(session_id=f"S{i:06d}", **kwargs) for i in range(count)]


# --- thresholds come from config, not from Python ----------------------------

def test_thresholds_are_read_from_settings_yaml():
    assert THRESHOLDS.min_occurrences >= 2
    assert 0.0 < THRESHOLDS.min_confidence <= 1.0
    assert 0.0 <= THRESHOLDS.max_context_share <= 1.0
    assert THRESHOLDS.history_window >= THRESHOLDS.min_occurrences


def test_thresholds_fall_back_to_defaults_when_block_is_absent():
    fallback = TrainingThresholds.from_settings({})
    assert fallback.min_occurrences == 2
    assert fallback.min_confidence == 0.70


# --- the gate: refusals ------------------------------------------------------

def test_single_occurrence_does_not_trigger():
    decision = evaluate_training_gate(repeated(1), operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert decision.trigger is None
    assert decision.reasons == (REASON_NOT_REPEATED,)
    assert decision.qualifying_occurrences == 1


def test_empty_history_does_not_trigger():
    decision = evaluate_training_gate([], operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert decision.reasons == (REASON_NO_HISTORY,)


def test_context_driven_does_not_trigger_even_when_repeated():
    history = repeated(5, attribution="context-driven")
    decision = evaluate_training_gate(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert REASON_CONTEXT_DRIVEN_ATTRIBUTION in decision.reasons
    assert decision.checks["issue_repeated"] is True  # it did repeat; it is just not ours to coach
    assert decision.checks["operator_linked"] is False


def test_low_confidence_does_not_trigger_even_when_repeated():
    low = THRESHOLDS.min_confidence - 0.01
    decision = evaluate_training_gate(
        repeated(5, confidence=low), operator_id=OPERATOR, issue_type=ISSUE
    )
    assert decision.triggered is False
    assert REASON_LOW_CONFIDENCE in decision.reasons
    assert decision.checks["confidence_sufficient"] is False


def test_confidence_exactly_at_threshold_is_accepted():
    """The floor is inclusive; a boundary flip should fail a test, not surprise a user."""
    history = repeated(THRESHOLDS.min_occurrences, confidence=THRESHOLDS.min_confidence)
    assert check_training_trigger(history, operator_id=OPERATOR, issue_type=ISSUE) is not None


def test_insufficient_evidence_never_triggers():
    history = repeated(6, attribution="insufficient_evidence")
    decision = evaluate_training_gate(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert decision.reasons == (REASON_INSUFFICIENT_EVIDENCE,)
    assert decision.checks["evidence_usable"] is False


def test_hard_site_does_not_get_the_operator_coached():
    """Operator-driven attribution, high confidence, but context explains the gap."""
    history = repeated(6, operator_residual=0.01, context_explained_component=0.09)
    decision = evaluate_training_gate(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert REASON_CONTEXT_EXPLAINED in decision.reasons
    assert decision.checks["context_not_dominant"] is False


def test_mixed_attribution_triggers_only_when_context_is_not_dominant():
    dominated = repeated(4, attribution="mixed", operator_residual=0.02, context_explained_component=0.08)
    assert check_training_trigger(dominated, operator_id=OPERATOR, issue_type=ISSUE) is None

    operator_led = repeated(4, attribution="mixed", operator_residual=0.08, context_explained_component=0.02)
    trigger = check_training_trigger(operator_led, operator_id=OPERATOR, issue_type=ISSUE)
    assert trigger is not None
    assert trigger.attribution_type == "mixed"


def test_occurrences_that_fail_different_checks_do_not_add_up_to_a_trigger():
    """One low-confidence plus one context-driven is not two qualifying occurrences."""
    history = [
        behavior(confidence=0.2, session_id="S000001"),
        behavior(attribution="context-driven", session_id="S000002"),
    ]
    decision = evaluate_training_gate(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.triggered is False
    assert decision.qualifying_occurrences == 0
    assert {REASON_LOW_CONFIDENCE, REASON_CONTEXT_DRIVEN_ATTRIBUTION} <= set(decision.reasons)


def test_unmapped_issue_type_does_not_trigger():
    decision = evaluate_training_gate(
        repeated(4), operator_id=OPERATOR, issue_type="totally_unknown_issue"
    )
    assert decision.triggered is False
    assert decision.reasons == (REASON_NO_LESSON_FOR_ISSUE,)


def test_only_the_recent_window_is_weighed():
    """Old qualifying sessions must not keep an operator permanently triggerable."""
    stale = repeated(THRESHOLDS.min_occurrences)
    recent = repeated(THRESHOLDS.history_window, attribution="context-driven")
    decision = evaluate_training_gate(stale + recent, operator_id=OPERATOR, issue_type=ISSUE)
    assert decision.considered == THRESHOLDS.history_window
    assert decision.triggered is False


# --- the gate: firing --------------------------------------------------------

def test_repeated_confident_operator_driven_does_trigger():
    history = repeated(THRESHOLDS.min_occurrences)
    trigger = check_training_trigger(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert trigger is not None
    assert trigger.operator_id == OPERATOR
    assert trigger.issue_type == ISSUE
    assert trigger.attribution_type == "operator-driven"
    assert trigger.attribution_type in ATTRIBUTION_TYPES
    assert trigger.lesson_id == lesson_id_for_issue(ISSUE)
    assert trigger.escalation_state == ESCALATION_FIRST
    assert trigger.synthetic_flag is True
    assert trigger.trigger_event_id.startswith("TR")


def test_trigger_confidence_is_the_mean_of_qualifying_occurrences():
    history = [behavior(confidence=0.80, session_id="S1"), behavior(confidence=0.90, session_id="S2")]
    trigger = check_training_trigger(history, operator_id=OPERATOR, issue_type=ISSUE)
    assert trigger.confidence == pytest.approx(0.85)


def test_a_firing_gate_reports_no_blocking_reasons():
    decision = evaluate_training_gate(
        repeated(THRESHOLDS.min_occurrences), operator_id=OPERATOR, issue_type=ISSUE
    )
    assert decision.triggered is True
    assert decision.reasons == ()
    assert all(decision.checks.values())
    assert decision.to_dict()["trigger"]["issue_type"] == ISSUE


def test_trigger_event_id_can_be_supplied_for_deterministic_demos():
    trigger = check_training_trigger(
        repeated(3), operator_id=OPERATOR, issue_type=ISSUE, trigger_event_id="TR00023"
    )
    assert trigger.trigger_event_id == "TR00023"


# --- context share -----------------------------------------------------------

def test_context_share_is_a_proportion_of_the_explained_gap():
    assert context_share(behavior(operator_residual=0.09, context_explained_component=0.01)) == pytest.approx(0.1)
    assert context_share(behavior(operator_residual=0.05, context_explained_component=0.05)) == pytest.approx(0.5)


def test_context_share_of_a_zero_gap_fails_closed():
    """No measurable gap means nothing to attribute to the operator."""
    assert context_share(behavior(operator_residual=0.0, context_explained_component=0.0)) == 1.0
    assert check_training_trigger(
        repeated(4, operator_residual=0.0, context_explained_component=0.0),
        operator_id=OPERATOR,
        issue_type=ISSUE,
    ) is None


def test_zero_gap_agrees_with_the_shared_helper_contract():
    """`BehaviorResult.context_share()` in shared/schemas.py must also return 1.0
    for a zero gap.

    It briefly returned 0.0, which passes the "context is not dominant" check and
    would make a session with no measurable deviation coachable — the opposite of
    the intended protection. Member 1 corrected it. This test pins the agreed
    contract so a future change to either implementation fails loudly rather than
    silently inverting the gate.
    """
    zero_gap = behavior(operator_residual=0.0, context_explained_component=0.0)
    assert context_share(zero_gap) == 1.0
    assert check_training_trigger(
        repeated(4, operator_residual=0.0, context_explained_component=0.0),
        operator_id=OPERATOR,
        issue_type=ISSUE,
    ) is None


def test_an_operator_who_beats_expectation_is_never_coached():
    """Found in integration: the real model returns a negative residual when the
    operator outperforms, and `context_share` uses magnitudes, so outperforming
    looked identical to falling short. The gate assigned an idle-reduction lesson
    to an operator who idled *less* than expected.
    """
    decision = evaluate_training_gate(
        repeated(4, operator_residual=-0.09, context_explained_component=0.01),
        operator_id=OPERATOR,
        issue_type=ISSUE,
    )
    assert decision.triggered is False
    assert REASON_RESIDUAL_FAVOURABLE in decision.reasons


def test_an_operator_who_falls_short_is_still_coached():
    """The direction check must not disable the gate entirely."""
    assert check_training_trigger(
        repeated(4, operator_residual=0.09, context_explained_component=0.01),
        operator_id=OPERATOR,
        issue_type=ISSUE,
    ) is not None


def test_a_zero_residual_is_not_coachable():
    decision = evaluate_training_gate(
        repeated(4, operator_residual=0.0, context_explained_component=0.01),
        operator_id=OPERATOR,
        issue_type=ISSUE,
    )
    assert decision.triggered is False
    assert REASON_RESIDUAL_FAVOURABLE in decision.reasons


def test_direction_can_be_inverted_for_a_higher_is_better_metric():
    """For a metric where higher is better, a negative residual is the shortfall."""
    assert check_training_trigger(
        repeated(4, operator_residual=-0.09, context_explained_component=0.01),
        operator_id=OPERATOR,
        issue_type=ISSUE,
        higher_is_worse=False,
    ) is not None
    assert check_training_trigger(
        repeated(4, operator_residual=0.09, context_explained_component=0.01),
        operator_id=OPERATOR,
        issue_type=ISSUE,
        higher_is_worse=False,
    ) is None


def test_negative_residual_still_counts_as_operator_signal():
    assert context_share(behavior(operator_residual=-0.09, context_explained_component=0.01)) == pytest.approx(0.1)


# --- escalation --------------------------------------------------------------

def test_escalation_state_reflects_prior_triggers_for_the_same_issue():
    assert escalation_state_for(0, THRESHOLDS) == ESCALATION_FIRST
    assert escalation_state_for(1, THRESHOLDS) == ESCALATION_REPEAT
    assert escalation_state_for(THRESHOLDS.escalate_after_triggers, THRESHOLDS) == ESCALATION_ESCALATED
    assert escalation_state_for(99, THRESHOLDS) == ESCALATION_ESCALATED


def test_prior_triggers_are_carried_onto_the_trigger_record():
    trigger = check_training_trigger(
        repeated(3), operator_id=OPERATOR, issue_type=ISSUE, prior_trigger_count=5
    )
    assert trigger.escalation_state == ESCALATION_ESCALATED


def test_escalation_advances_when_the_follow_up_does_not_improve():
    assert next_escalation_state(ESCALATION_FIRST, target_met=False) == ESCALATION_REPEAT
    assert next_escalation_state(ESCALATION_REPEAT, target_met=False) == ESCALATION_ESCALATED
    assert next_escalation_state(ESCALATION_ESCALATED, target_met=False) == ESCALATION_ESCALATED


def test_meeting_the_improvement_target_resolves_the_cycle():
    assert next_escalation_state(ESCALATION_FIRST, target_met=True) == ESCALATION_RESOLVED
    assert next_escalation_state(ESCALATION_ESCALATED, target_met=True) == ESCALATION_RESOLVED
    assert next_escalation_state(ESCALATION_RESOLVED, target_met=False) == ESCALATION_RESOLVED


def test_unknown_escalation_state_is_rejected():
    with pytest.raises(ValueError):
        next_escalation_state("promoted", target_met=False)


# --- content pack ------------------------------------------------------------

def test_every_content_file_loads(content):
    assert 3 <= len(content.lessons) <= 5
    assert len(content.scenarios) == 3
    assert content.quizzes


def test_every_issue_type_maps_to_a_lesson_that_exists(content):
    assert content.issue_types()
    for issue_type in content.issue_types():
        lesson = content.lesson_for_issue(issue_type)
        assert lesson is not None
        assert content.lesson(lesson.lesson_id) is lesson


def test_every_lesson_has_a_quiz_of_the_agreed_length(content):
    for lesson in content.lessons:
        quiz = content.quiz_for_lesson(lesson.lesson_id)
        assert quiz is not None, lesson.lesson_id
        assert MIN_QUIZ_QUESTIONS <= quiz.total_questions <= MAX_QUIZ_QUESTIONS


def test_every_quiz_answer_is_one_of_its_own_options(content):
    for quiz in content.quizzes:
        for question in quiz.questions:
            assert question.answer_id in {o.option_id for o in question.options}


def test_every_scenario_points_at_a_real_lesson(content):
    for scenario in content.scenarios:
        assert content.lesson(scenario.lesson_id) is not None


def test_the_architecture_topics_are_all_covered(content):
    covered = set(content.issue_types())
    assert {"seatbelt_compliance", "idle_reduction", "swing_zone_awareness", "loading_efficiency"} <= covered


def test_content_is_labelled_synthetic_and_claims_no_official_source(content):
    for lesson in content.lessons:
        assert lesson.synthetic_flag is True
        assert lesson.disclaimer
        assert "caterpillar" not in lesson.summary.lower()
    for scenario in content.scenarios:
        assert scenario.synthetic_flag is True


def test_unknown_issue_type_maps_to_no_lesson(content):
    assert content.lesson_id_for_issue("not_an_issue") is None


def test_a_malformed_pack_is_rejected_rather_than_half_loaded(tmp_path):
    (tmp_path / "lessons.yaml").write_text(
        "lessons:\n"
        "  - lesson_id: L001\n"
        "    title: One\n"
        "    issue_types: [dup_issue]\n"
        "    quiz_id: Q001\n"
        "  - lesson_id: L002\n"
        "    title: Two\n"
        "    issue_types: [dup_issue]\n"
        "    quiz_id: Q001\n",
        encoding="utf-8",
    )
    (tmp_path / "quizzes.yaml").write_text("quizzes: []\n", encoding="utf-8")
    (tmp_path / "scenarios.yaml").write_text("scenarios: []\n", encoding="utf-8")
    clear_content_cache()
    with pytest.raises(ContentError):
        load_content(tmp_path)
    clear_content_cache()


def test_missing_content_directory_raises_a_clear_error(tmp_path):
    clear_content_cache()
    with pytest.raises(ContentError, match="not found"):
        load_content(tmp_path / "nowhere")
    clear_content_cache()


# --- quiz scoring ------------------------------------------------------------

def test_quiz_scoring_counts_only_correct_answers(content):
    quiz = content.quiz_for_lesson("L001")
    answers = {q.question_id: q.answer_id for q in quiz.questions}
    result = score_quiz(quiz, answers)
    assert result.correct == quiz.total_questions
    assert result.score == 1.0
    assert result.passed is True


def test_a_wrong_answer_is_not_credited(content):
    quiz = content.quiz_for_lesson("L001")
    answers = {q.question_id: q.answer_id for q in quiz.questions}
    first = quiz.questions[0]
    wrong = next(o.option_id for o in first.options if o.option_id != first.answer_id)
    answers[first.question_id] = wrong

    result = score_quiz(quiz, answers)
    assert result.correct == quiz.total_questions - 1
    assert result.score == pytest.approx((quiz.total_questions - 1) / quiz.total_questions)
    assert result.answers[0].correct is False


def test_unanswered_questions_score_zero_and_are_reported(content):
    quiz = content.quiz_for_lesson("L002")
    result = score_quiz(quiz, {})
    assert result.correct == 0
    assert result.answered == 0
    assert result.score == 0.0
    assert result.passed is False


def test_answers_to_questions_outside_the_quiz_are_ignored(content):
    quiz = content.quiz_for_lesson("L003")
    answers = {q.question_id: q.answer_id for q in quiz.questions}
    answers["QZZZ-9"] = "a"
    result = score_quiz(quiz, answers)
    assert result.total == quiz.total_questions
    assert result.correct == quiz.total_questions


def test_pass_threshold_is_configurable_and_inclusive(content):
    quiz = content.quiz_for_lesson("L003")
    answers = {q.question_id: q.answer_id for q in quiz.questions}
    answers[quiz.questions[0].question_id] = None
    score = (quiz.total_questions - 1) / quiz.total_questions
    assert score_quiz(quiz, answers, pass_score=score).passed is True
    assert score_quiz(quiz, answers, pass_score=score + 0.01).passed is False


# --- improvement measurement -------------------------------------------------

def test_improvement_for_a_lower_is_better_metric():
    result = compare_metrics("idle_ratio", before=0.30, after=0.24)
    assert result.improvement_pct == pytest.approx(20.0)
    assert result.improved is True
    assert result.comparable is True


def test_a_metric_that_got_worse_reports_a_negative_improvement():
    result = compare_metrics("idle_ratio", before=0.20, after=0.25)
    assert result.improvement_pct == pytest.approx(-25.0)
    assert result.improved is False
    assert result.target_met is False


def test_higher_is_better_metrics_are_supported():
    result = compare_metrics("tonnes_per_hour", before=100.0, after=115.0, lower_is_better=False)
    assert result.improvement_pct == pytest.approx(15.0)
    assert result.improved is True


def test_a_before_metric_of_zero_is_division_safe():
    result = compare_metrics("safety_events", before=0.0, after=0.0)
    assert isinstance(result, ImprovementResult)
    assert result.improvement_pct == 0.0
    assert result.comparable is False
    assert result.improved is False
    assert result.reasons == (REASON_BASELINE_ZERO,)


def test_a_before_metric_of_zero_does_not_claim_a_regression_either():
    result = compare_metrics("safety_events", before=0.0, after=3.0)
    assert result.comparable is False
    assert result.improvement_pct == 0.0


def test_no_change_is_neither_improvement_nor_regression():
    result = compare_metrics("idle_ratio", before=0.25, after=0.25)
    assert result.improvement_pct == 0.0
    assert result.improved is False


def test_target_is_met_exactly_at_the_configured_percentage():
    result = compare_metrics("idle_ratio", before=100.0, after=90.0, target_pct=10.0)
    assert result.target_met is True


# --- the closed loop ---------------------------------------------------------

def test_a_measured_improvement_closes_the_loop(content):
    trigger = check_training_trigger(repeated(3), operator_id=OPERATOR, issue_type=ISSUE)
    quiz = content.quiz_for_lesson(trigger.lesson_id)
    quiz_result = score_quiz(quiz, {q.question_id: q.answer_id for q in quiz.questions})
    improvement = compare_metrics("idle_ratio", before=0.30, after=0.20)

    outcome = complete_cycle(trigger, quiz_result, improvement)
    assert outcome.next_escalation_state == ESCALATION_RESOLVED
    assert outcome.to_dict()["quiz_result"]["passed"] is True
    assert outcome.synthetic_flag is True


def test_passing_the_quiz_alone_does_not_resolve_the_issue(content):
    """Only the follow-up measurement closes the loop."""
    trigger = check_training_trigger(repeated(3), operator_id=OPERATOR, issue_type=ISSUE)
    quiz = content.quiz_for_lesson(trigger.lesson_id)
    quiz_result = score_quiz(quiz, {q.question_id: q.answer_id for q in quiz.questions})
    improvement = compare_metrics("idle_ratio", before=0.30, after=0.30)

    outcome = complete_cycle(trigger, quiz_result, improvement)
    assert quiz_result.passed is True
    assert outcome.next_escalation_state == ESCALATION_REPEAT


def test_an_unmeasurable_follow_up_does_not_count_as_success(content):
    trigger = check_training_trigger(repeated(3), operator_id=OPERATOR, issue_type=ISSUE)
    outcome = complete_cycle(trigger, None, compare_metrics("safety_events", before=0.0, after=0.0))
    assert outcome.next_escalation_state == ESCALATION_REPEAT
