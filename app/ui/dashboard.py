"""Task Planning Dashboard -- the single shift home screen (Feature 05).

    streamlit run app/ui/dashboard.py

Everything on screen is either a real output of an integrated feature or an
explicit "not integrated yet" notice. Nothing is filled in with a placeholder
number, because a plausible-looking ETA is indistinguishable from a real one.

Leakage rule for the UI: recorded outcomes (`actual_*`) appear only in the
end-of-shift comparison, labelled as what happened, and never beside a
prediction as though they were one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.buddy.buddy import ask  # noqa: E402
from features.buddy.retrieval import retrieve  # noqa: E402
from features.buddy.safe_state import MachineStateSnapshot, evaluate_safe_state  # noqa: E402
from features.dashboard import adapters  # noqa: E402
from features.dashboard.data import DashboardData, MissingDatasetError  # noqa: E402
from features.passport.authorization import check_authorization  # noqa: E402
from features.training.lessons import load_content  # noqa: E402
from features.training.progress import compare_metrics, next_escalation_state, score_quiz  # noqa: E402
from features.training.trigger import (  # noqa: E402
    ESCALATION_ESCALATED,
    ESCALATION_FIRST,
    ESCALATION_REPEAT,
    evaluate_training_gate,
)
from shared.config import load_settings  # noqa: E402
from shared.schemas import BehaviorResult  # noqa: E402

st.set_page_config(page_title="CAT Operator Copilot", layout="wide")


@st.cache_data(show_spinner=False)
def load_table(synthetic_dir: str, name: str) -> pd.DataFrame:
    return DashboardData(Path(synthetic_dir))._read(name)


def unavailable(result: adapters.AdapterResult) -> None:
    st.info(f"Not integrated yet — {result.feature} (owner: {result.owner})", icon=":material/link_off:")


def render_integration_strip() -> None:
    status = adapters.integration_status()
    ready = [f for f, r in status.items() if r.available]
    st.caption(
        f"Integrated: {', '.join(ready) if ready else 'none yet'} · "
        f"Awaiting: {', '.join(f for f, r in status.items() if not r.available) or 'none'}"
    )


def render_header(context, operator, machine, events: pd.DataFrame) -> None:
    st.subheader("Shift status")
    authorization = check_authorization(operator, machine) if operator and machine else None

    columns = st.columns(5)
    columns[0].metric("Operator", context.operator_id)
    columns[1].metric("Machine", f"{context.machine_id}", context.machine.machine_type or None)
    if authorization is not None:
        columns[2].metric("Authorization", "Authorized" if authorization.authorized else "Refused")
    columns[3].metric("Task", context.task_id or "—", context.task.task_type or None)

    severities = events.severity.value_counts().to_dict() if not events.empty else {}
    worst = next((s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO") if s in severities), "None")
    columns[4].metric("Worst recorded safety event", worst, f"{len(events)} in session")

    if authorization is not None and not authorization.authorized:
        st.error(
            "Authorization refused: " + ", ".join(authorization.reasons)
            + ". No session may be opened for this operator on this machine."
        )


def render_plan(context) -> None:
    st.subheader("Plan")
    result = adapters.get_plan(tasks=None, session_context=context)
    if not result.available:
        unavailable(result)
        st.caption(
            "When Optimal Task Sequencing lands this shows the optimized order, the next "
            "task with its priority and deadline, and the reason for that ordering."
        )
        return
    st.write(result.value)


def render_prediction(context) -> None:
    st.subheader("Prediction")
    result = adapters.get_prediction(context)
    if not result.available:
        unavailable(result)
        st.caption("ETA P10/P50/P90, fuel range and confidence appear here once the model lands.")
        return

    prediction = result.value
    columns = st.columns(4)
    columns[0].metric("ETA P50 (min)", f"{prediction.eta_p50:.1f}")
    columns[1].metric("ETA range", f"{prediction.eta_p10:.0f} – {prediction.eta_p90:.0f}")
    columns[2].metric("Fuel P50 (L)", f"{prediction.fuel_p50:.1f}")
    columns[3].metric("Confidence", f"{prediction.confidence:.2f}")
    if prediction.factors:
        st.caption("Contributing factors: " + ", ".join(prediction.factors)
                   + " — contribution to the prediction, not causes.")


def render_conditions(context) -> None:
    st.subheader("Conditions")
    columns = st.columns(4)
    with columns[0]:
        st.markdown("**Weather**")
        st.write({
            "rain_mm": context.weather.rain_mm,
            "temperature_c": context.weather.temperature_c,
            "visibility_m": context.weather.visibility_m,
            "dust_level": context.weather.dust_level,
            "day_night": context.weather.day_night,
        })
    with columns[1]:
        st.markdown("**Site**")
        st.write({
            "soil_material": context.site.soil_material,
            "soil_hardness_index": context.site.hardness,
            "slope_deg": context.site.slope,
            "surface": context.site.surface,
        })
    with columns[2]:
        st.markdown("**Traffic**")
        st.write({"congestion": context.traffic.congestion})
    with columns[3]:
        st.markdown("**Machine**")
        st.write({
            "condition": context.machine.machine_condition,
            "engine_hours": context.machine.engine_hours,
            "attachment": context.machine.attachment,
        })


def render_live_operation(data: DashboardData, session_id: str, context, events: pd.DataFrame) -> None:
    st.subheader("Live operation")

    behavior = adapters.get_behavior(context)
    if not behavior.available:
        unavailable(behavior)
    else:
        st.write(behavior.value)

    if events.empty:
        st.caption("No safety events recorded for this session.")
    else:
        st.markdown("**Safety events recorded in this session**")
        st.caption(
            "Recorded in the dataset. Live evaluation is the Safety Guardian's decision, "
            "which is deterministic and owned by Member 2."
        )
        st.dataframe(
            events[["timestamp", "severity", "event_type", "trigger_reason",
                    "worker_distance_m", "closing_speed_mps", "machine_state"]],
            width="stretch", hide_index=True,
        )


def render_end_of_shift(data: DashboardData, session_id: str) -> None:
    st.subheader("End of shift — predicted vs actual")
    outcome = data.outcome(session_id)
    prediction = adapters.get_prediction(data.build_session_context(session_id))

    columns = st.columns(4)
    columns[0].metric("Actual duration (min)", f"{outcome['actual_task_duration_min']:.1f}")
    columns[1].metric("Actual fuel (L)", f"{outcome['actual_fuel_used_l']:.1f}")
    columns[2].metric("Actual idle (min)", f"{outcome['actual_idle_time_min']:.1f}")
    columns[3].metric("Actual cycles", int(outcome["actual_cycle_count"]))

    if prediction.available:
        st.caption("Compared against the prediction above.")
    else:
        st.caption(
            "These are recorded outcomes, shown after the fact. They are the prediction "
            "targets and are never used as model inputs."
        )


def render_buddy(context, telemetry: pd.DataFrame, events: pd.DataFrame) -> None:
    st.subheader("Operating Buddy")
    st.caption(
        "Available only when the machine is parked or in verified safe idle with no "
        "attachment movement. The Buddy explains; it never makes a safety decision."
    )

    if telemetry.empty:
        st.warning("No telemetry for this session, so machine state is unknown — the Buddy stays blocked.")
        return

    states = telemetry.machine_state.unique().tolist()
    chosen_state = st.selectbox("Machine state (from this session's telemetry)", states)
    row = telemetry[telemetry.machine_state == chosen_state].iloc[0]

    snapshot = MachineStateSnapshot(
        machine_state=row.machine_state,
        attachment_movement=row.attachment_movement,
        arm_speed=row.arm_speed,
        bucket_state=row.bucket_state,
        machine_speed_kmh=row.machine_speed_kmh,
    )
    gate = evaluate_safe_state(snapshot)
    if gate.allowed:
        st.success("Buddy enabled — machine is in a safe state.")
    else:
        st.warning("Buddy blocked: " + ", ".join(gate.reasons))

    st.caption(
        "Try: *what is my next task* · *how much fuel is left* · *is it safe to swing* · "
        "*what is the stopping procedure* · *is it dangerous near the river* (not covered — defers)"
    )
    question = st.text_input("Ask the Buddy", "Is it safe to swing right now?")
    if st.button("Ask"):
        telemetry_row = row.to_dict()
        evidence = retrieve(
            question,
            session_context=context,
            telemetry_row=telemetry_row,
            safety_events=events.to_dict(orient="records") if not events.empty else (),
        )
        response = ask(question, snapshot, evidence)
        if response.answered:
            st.success(response.answer)
        else:
            st.warning(f"{response.status}: {response.reason}")
        with st.expander("Evidence and provenance"):
            st.write(response.to_dict())


def render_training(operator_id: str) -> None:
    st.subheader("Training Hub")
    content = load_content()
    gate_tab, lesson_tab, measure_tab = st.tabs(
        ["Gate", "Lesson, scenario and quiz", "Measured improvement"]
    )
    with gate_tab:
        render_training_gate(operator_id)
    with lesson_tab:
        render_lesson(content)
    with measure_tab:
        render_measurement()


def render_lesson(content) -> None:
    titles = {lesson.title: lesson for lesson in content.lessons}
    lesson = titles[st.selectbox("Lesson", list(titles))]

    st.markdown(f"**{lesson.title}** · {lesson.duration_min} min · covers: "
                f"{', '.join(lesson.issue_types)}")
    st.write(lesson.summary)
    if lesson.objectives:
        st.markdown("**Objectives**")
        for item in lesson.objectives:
            st.markdown(f"- {item}")
    if lesson.key_points:
        st.markdown("**Key points**")
        for item in lesson.key_points:
            st.markdown(f"- {item}")
    if lesson.practice_prompt:
        st.info(lesson.practice_prompt)
    if lesson.disclaimer:
        st.caption(lesson.disclaimer)

    for scenario in content.scenarios_for_lesson(lesson.lesson_id):
        with st.expander(f"Scenario — {scenario.title}"):
            st.write(scenario.situation)
            if scenario.context_factors:
                st.markdown("**Context at the time:** " + ", ".join(scenario.context_factors))
            if scenario.what_the_model_saw:
                st.markdown(f"**What the model saw:** {scenario.what_the_model_saw}")
            for question in scenario.discussion_questions:
                st.markdown(f"- {question}")
            if scenario.takeaway:
                st.success(scenario.takeaway)

    quiz = content.quiz_for_lesson(lesson.lesson_id)
    if quiz is None:
        st.warning("No quiz is mapped to this lesson.")
        return

    st.markdown(f"### {quiz.title}")
    with st.form(key=f"quiz_{quiz.quiz_id}"):
        answers = {}
        for index, question in enumerate(quiz.questions, start=1):
            labels = {option.text: option.option_id for option in question.options}
            chosen = st.radio(
                f"{index}. {question.prompt}", list(labels),
                key=f"{quiz.quiz_id}_{question.question_id}", index=None,
            )
            answers[question.question_id] = labels.get(chosen)
        submitted = st.form_submit_button("Submit answers")

    if submitted:
        result = score_quiz(quiz, answers)
        if result.passed:
            st.success(f"Passed — {result.correct}/{result.total} "
                       f"({result.score:.0%}), pass mark {result.pass_score:.0%}.")
        else:
            st.error(f"Not passed — {result.correct}/{result.total} "
                     f"({result.score:.0%}), pass mark {result.pass_score:.0%}.")
        st.caption(
            "Passing the quiz does not close the issue. Only a follow-up measurement "
            "that meets the improvement target does."
        )
        for answer in result.answers:
            marker = "correct" if answer.correct else "incorrect"
            with st.expander(f"{answer.question_id} — {marker}"):
                if answer.explanation:
                    st.write(answer.explanation)
                if not answer.correct:
                    st.caption(f"Expected: {answer.correct_answer_id}")


def render_measurement() -> None:
    st.caption(
        "The loop closes on measurement, not on completing a lesson. Enter the metric "
        "before coaching and after it."
    )
    columns = st.columns(4)
    metric = columns[0].selectbox(
        "Metric", ["idle_ratio", "cycle_time_sec", "fuel_l_per_cycle", "safety_event_rate"]
    )
    before = columns[1].number_input("Before", value=0.24, step=0.01, format="%.3f")
    after = columns[2].number_input("After", value=0.19, step=0.01, format="%.3f")
    state = columns[3].selectbox(
        "Current escalation state", [ESCALATION_FIRST, ESCALATION_REPEAT, ESCALATION_ESCALATED]
    )

    result = compare_metrics(metric, before, after, lower_is_better=True)
    if not result.comparable:
        st.warning(
            f"No baseline to compare against ({', '.join(result.reasons)}). "
            "No improvement percentage is reported rather than inventing one."
        )
    elif result.target_met:
        st.success(f"Improved {result.improvement_pct:.1f}% — target "
                   f"{result.target_pct:.0f}% met.")
    else:
        st.info(f"Changed {result.improvement_pct:.1f}% — target "
                f"{result.target_pct:.0f}% not met.")

    st.markdown(
        f"Escalation: `{state}` → `{next_escalation_state(state, result.comparable and result.target_met)}`"
    )


def render_training_gate(operator_id: str) -> None:
    st.caption(
        "Behavioral Fingerprint is not integrated yet, so these inputs are set by hand to "
        "demonstrate the gate. They are not model output."
    )

    columns = st.columns(4)
    attribution = columns[0].selectbox(
        "Attribution", ["operator-driven", "mixed", "context-driven", "insufficient_evidence"]
    )
    confidence = columns[1].slider("Confidence", 0.0, 1.0, 0.85, 0.01)
    operator_residual = columns[2].slider("Operator residual", 0.0, 0.2, 0.06, 0.01)
    context_component = columns[3].slider("Context-explained component", 0.0, 0.2, 0.02, 0.01)
    occurrences = st.slider("Times this issue occurred", 1, 6, 2)

    history = [
        BehaviorResult(
            session_id=f"S{index:06d}", attribution=attribution, confidence=confidence,
            observed_value=0.26, expected_value=0.16, operator_residual=operator_residual,
            context_explained_component=context_component, coaching_eligible=False,
            synthetic_flag=True,
        )
        for index in range(occurrences)
    ]
    decision = evaluate_training_gate(history, operator_id=operator_id, issue_type="idle_reduction")

    if decision.triggered:
        st.success(
            f"Training triggered — lesson {decision.trigger.lesson_id}, "
            f"escalation {decision.trigger.escalation_state}."
        )
    else:
        st.info("No training triggered: " + ", ".join(decision.reasons))
    st.caption(
        "The gate reports that observed behaviour differs from what was expected under the "
        "current operating context. It does not conclude anything about the operator."
    )


def main() -> None:
    st.title("CAT Operator Copilot")
    st.caption("All data is synthetic. Nothing here is real CAT field telemetry.")

    settings = load_settings()
    data = DashboardData.from_settings(settings)

    if not data.available():
        st.error("The synthetic dataset is not present.")
        st.code(
            "git checkout origin/feature/sruthi-data-models -- data/synthetic/\n"
            "git reset -q data/synthetic/",
            language="bash",
        )
        return

    synthetic_dir = str(data.synthetic_dir)
    sessions = load_table(synthetic_dir, "task_sessions.csv")

    with st.sidebar:
        st.header("Session")
        operator_id = st.selectbox("Operator", sorted(sessions.operator_id.unique()))
        operator_sessions = sessions[sessions.operator_id == operator_id]
        session_id = st.selectbox("Session", operator_sessions.session_id.tolist())
        st.caption(f"{len(operator_sessions)} sessions for {operator_id}")

    context = data.build_session_context(session_id)
    repository = data.repository()
    operator = repository.get_operator(operator_id)
    machine = repository.get_machine(context.machine_id)

    events = load_table(synthetic_dir, "safety_events.csv")
    events = events[events.session_id == session_id]
    telemetry = load_table(synthetic_dir, "telemetry.csv")
    telemetry = telemetry[telemetry.session_id == session_id]

    render_integration_strip()
    render_header(context, operator, machine, events)
    st.divider()
    render_plan(context)
    st.divider()
    render_prediction(context)
    st.divider()
    render_conditions(context)
    st.divider()
    render_live_operation(data, session_id, context, events)
    st.divider()
    render_buddy(context, telemetry, events)
    st.divider()
    render_training(operator_id)
    st.divider()
    render_end_of_shift(data, session_id)


if __name__ == "__main__":
    main()
