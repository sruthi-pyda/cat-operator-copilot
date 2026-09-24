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
from datetime import datetime
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
from features.dashboard.attention_candidates import (  # noqa: E402
    replan_candidate,
    safety_candidate,
)
from features.dashboard.comparison import compare_prediction_to_outcome  # noqa: E402
from features.dashboard.data import DashboardData, MissingDatasetError  # noqa: E402
from features.dashboard.replan import describe_context_change, replan_reason  # noqa: E402
from features.dashboard.replay import TelemetryReplay  # noqa: E402
from features.passport.authorization import check_authorization  # noqa: E402
from features.training.lessons import load_content  # noqa: E402
from features.training.peer import find_similar, load_peer_examples  # noqa: E402
from features.training.progress import compare_metrics, next_escalation_state, score_quiz  # noqa: E402
from features.training.trigger import (  # noqa: E402
    ESCALATION_ESCALATED,
    ESCALATION_FIRST,
    ESCALATION_REPEAT,
    TrainingThresholds,
    evaluate_training_gate,
)
from shared.config import load_settings  # noqa: E402
from shared.schemas import BehaviorResult  # noqa: E402

st.set_page_config(page_title="CAT Operator Copilot", layout="wide")


@st.cache_data(show_spinner=False)
def load_table(synthetic_dir: str, name: str) -> pd.DataFrame:
    return DashboardData(Path(synthetic_dir))._read(name)


def safe_section(render, *args, name: str) -> None:
    """Render one section; a failure inside it must not blank the shift screen.

    The error is shown with the section that raised it rather than swallowed --
    a teammate's module failing should be visible and attributable, just not
    fatal. Streamlit otherwise stops the whole script on an uncaught exception,
    so every section below the broken one would silently disappear.
    """
    try:
        render(*args)
    except Exception as exc:  # noqa: BLE001 - deliberate boundary
        st.error(
            f"**{name} failed to render.** {type(exc).__name__}: {exc}\n\n"
            "The rest of the dashboard is unaffected. This is a fault in the "
            "feature behind this section, not in the data."
        )


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


def _clock(value: Any) -> str:
    text = str(value or "")
    return text[11:16] if len(text) >= 16 else text


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

    plan = result.value
    steps = plan.get("steps") or []

    if not steps:
        st.info("No feasible task for this operator and machine in this shift.")
    else:
        first = steps[0]
        eta, fuel = first.get("eta_min", {}), first.get("fuel_l", {})
        st.markdown(f"**Next task — {first.get('task_id')}**")
        columns = st.columns(4)
        columns[0].metric("Starts", _clock(first.get("start")))
        columns[1].metric(
            "ETA P50", f"{eta.get('p50', 0):.0f} min",
            f"P10-P90 {eta.get('p10', 0):.0f}-{eta.get('p90', 0):.0f}", delta_color="off",
        )
        columns[2].metric("Fuel P50", f"{fuel.get('p50', 0):.1f} L")
        columns[3].metric("Confidence", f"{first.get('prediction_confidence', 0):.2f}")

        st.dataframe(
            pd.DataFrame([
                {
                    "#": index,
                    "task": step.get("task_id"),
                    "start": _clock(step.get("start")),
                    "ends (P50)": _clock(step.get("end_p50")),
                    "ETA P50 (min)": round(step.get("eta_min", {}).get("p50", 0), 1),
                    "fuel P50 (L)": round(step.get("fuel_l", {}).get("p50", 0), 1),
                    "cost": round(sum(step.get("cost_breakdown", {}).values()), 1),
                }
                for index, step in enumerate(steps, start=1)
            ]),
            width="stretch", hide_index=True,
        )

    st.markdown(f"**Why this order:** {plan.get('reason', 'not stated')}")

    columns = st.columns(3)
    columns[0].metric("Total cost", f"{plan.get('total_cost', 0):.0f}")
    columns[1].metric("Deadlines met", str(plan.get("deadlines_met", "—")))
    columns[2].metric("Shift planned", f"{plan.get('shift_minutes_planned', 0):.0f} min")

    breakdown = plan.get("cost_breakdown") or {}
    if breakdown:
        with st.expander("Cost breakdown — what the ordering traded off"):
            st.dataframe(
                pd.DataFrame([{"component": k, "cost": round(v, 1)} for k, v in breakdown.items()]),
                width="stretch", hide_index=True,
            )
            st.caption("Cost units are abstract planning units, not money or minutes alone.")

    excluded = plan.get("excluded") or {}
    if excluded:
        # The optimizer returns {task_id: reason}; tolerate a list of rows too,
        # since this is a boundary between two people's code.
        if isinstance(excluded, dict):
            rows = [{"task": task, "why it was refused": reason}
                    for task, reason in excluded.items()]
        else:
            rows = [r if isinstance(r, dict) else {"task": str(r)} for r in excluded]

        blocked = sum(1 for r in rows if "CRITICAL" in str(r.get("why it was refused", "")))
        with st.expander(f"Excluded from the plan ({len(rows)})"):
            st.caption(
                "Tasks the optimizer refused, and why. A CRITICAL safety finding is a hard "
                "block, not a cost to be traded away against time or fuel."
            )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if blocked:
                st.warning(
                    f"{blocked} task(s) blocked outright on CRITICAL safety findings.",
                    icon=":material/block:",
                )


def render_replan(data: DashboardData, sessions: pd.DataFrame, session_id: str, context):
    st.subheader("Replanning")
    operator_sessions = sessions[sessions.operator_id == context.operator_id]
    ordered = operator_sessions.sort_values("start_timestamp").session_id.tolist()
    position = ordered.index(session_id)

    if position == 0:
        st.caption("Earliest session for this operator — nothing to compare against.")
        return ()

    previous = data.build_session_context(ordered[position - 1])
    changes = describe_context_change(previous, context)
    plan = adapters.get_plan(tasks=None, session_context=context)

    if not changes:
        st.caption("No material change in operating context since the previous session.")
    elif plan.available:
        st.warning(f"**Plan changed**\n\nReason: {replan_reason(changes)}")
    else:
        st.warning(
            f"**Conditions changed since the previous session**\n\n"
            f"Reason: {replan_reason(changes)}"
        )
        st.caption(
            "No plan has been recalculated — Optimal Task Sequencing is not integrated yet "
            f"(owner: {plan.owner}). Deciding whether this warrants a replan is that "
            "feature's call, not the dashboard's."
        )

    for change in changes:
        st.markdown(f"- {change.detail()}")

    if changes:
        st.caption(
            "Compared against this operator's previous session, which may be at a different "
            "location — so a site change can appear here alongside a genuine weather change. "
            "A true within-shift comparison needs the telemetry replay."
        )
    return changes


def render_attention(events: pd.DataFrame, changes, telemetry: pd.DataFrame) -> None:
    st.subheader("Attention queue")
    st.caption(
        "Candidate events this slice proposes. Only the Attention Manager decides what "
        "reaches the operator — nothing here has been ranked or suppressed, and every "
        "candidate is still marked `pending`."
    )

    machine_state = telemetry.machine_state.iloc[0] if not telemetry.empty else None
    candidates = [safety_candidate(event) for event in events.to_dict(orient="records")]
    if changes:
        candidates.append(replan_candidate(changes, machine_state))

    if not candidates:
        st.caption("Nothing proposed for this session.")
        return

    routed = [adapters.route_event(candidate) for candidate in candidates]
    integrated = routed[0].available

    rows = []
    for candidate, outcome in zip(candidates, routed):
        decision = candidate.decision
        reason = ""
        if outcome.available and isinstance(outcome.value, dict):
            decision = outcome.value.get("decision", decision)
            reason = outcome.value.get("reason", "")
        rows.append({
            "event_type": candidate.event_type, "severity": candidate.severity,
            "urgency": candidate.urgency, "actionability": candidate.actionability,
            "operator_state": candidate.operator_state,
            "decision": decision, "reason": reason,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if not integrated:
        st.info(
            f"Not integrated yet — attention (owner: {routed[0].owner}). Until it lands these "
            "stay `pending`; deciding between them is that feature's job, not the dashboard's.",
            icon=":material/link_off:",
        )


def render_prediction(context) -> None:
    st.subheader("Prediction")
    result = adapters.get_prediction(context)
    if not result.available:
        unavailable(result)
        st.caption("ETA P10/P50/P90, fuel range and confidence appear here once the model lands.")
        return

    prediction = result.value
    columns = st.columns(4)
    columns[0].metric("ETA P50 (min)", f"{prediction.eta_p50:.1f}" if prediction.eta_p50 else "—")
    columns[1].metric(
        "ETA range",
        f"{prediction.eta_p10:.0f} – {prediction.eta_p90:.0f}" if prediction.eta_p50 else "—",
    )
    # A zeroed fuel field means this call predicted ETA only; a 0.0 on screen
    # would read as a prediction of no fuel use.
    columns[2].metric(
        "Fuel P50 (L)", f"{prediction.fuel_p50:.1f}" if prediction.fuel_p50 else "not predicted"
    )
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

    steps = list(TelemetryReplay(data, session_id).steps())
    if steps:
        position = st.slider(
            "Replay position", 0, len(steps) - 1, 0,
            help="Drag to move through the recorded shift.",
        )
        step = steps[position]
        st.progress((position + 1) / len(steps), text=f"Step {position + 1} of {len(steps)}")

        columns = st.columns(4)
        columns[0].metric("Time", step.timestamp[11:19])
        columns[1].metric("Machine state", step.machine_state)
        columns[2].metric("Buddy", "available" if step.buddy_available else "blocked")
        columns[3].metric(
            "Worker distance",
            f"{step.worker_distance_m:.1f} m" if step.worker_distance_m is not None else "—",
        )

        if step.safety_events:
            for event in step.safety_events:
                st.error(
                    f"{event.get('severity')} — {event.get('trigger_reason')} "
                    f"(action: {event.get('required_action')})"
                )
        if not step.buddy_available:
            st.caption("Buddy blocked here: " + ", ".join(step.safe_state.reasons))

        seen = sum(len(s.safety_events) for s in steps[: position + 1])
        st.caption(
            f"{seen} safety event(s) recorded up to this point in the shift. "
            "The Buddy verdict is evaluated live from this row, not replayed."
        )
    else:
        st.caption("No telemetry for this session, so there is nothing to replay.")

    behavior = adapters.get_behavior(context)
    if not behavior.available:
        unavailable(behavior)
    else:
        result = behavior.value
        st.markdown("**Behavioural fingerprint**")
        columns = st.columns(4)
        columns[0].metric("Observed", f"{result.observed_value:.3f}")
        columns[1].metric("Expected under context", f"{result.expected_value:.3f}")
        columns[2].metric("Attribution", result.attribution,
                          f"confidence {result.confidence:.2f}", delta_color="off")
        columns[3].metric("Coaching eligible", "yes" if result.coaching_eligible else "no")
        st.caption(
            f"Of the gap, {result.operator_residual:+.4f} is operator-linked and "
            f"{result.context_explained_component:+.4f} is explained by site, machine and "
            "weather conditions."
        )
        st.caption(
            "This says observed behaviour differs from what was expected under the current "
            "operating context. It is not a judgement that the operator performed badly, and "
            "a gap the context explains never becomes coaching."
        )

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

    if not prediction.available:
        unavailable(prediction)
        st.caption(
            "These are recorded outcomes, shown after the fact. They are the prediction "
            "targets and are never used as model inputs."
        )
        return

    rows = compare_prediction_to_outcome(prediction.value, outcome)
    st.dataframe(
        pd.DataFrame([row.to_dict() for row in rows]), width="stretch", hide_index=True
    )
    st.caption(
        "The actual falls inside the P10–P90 band or it does not — that is the honest "
        "test of the uncertainty estimate, not the P50 error alone. One session proves "
        "nothing either way; held-out metrics across many sessions are the real check."
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
        # Freshness is judged against the moment being replayed, not wall-clock.
        # The recorded shifts are months old, so using "now" would mark every
        # piece of session evidence stale and the Buddy would refuse to answer.
        as_of = None
        for stamp in (telemetry_row.get("timestamp"), context.timestamp):
            try:
                as_of = datetime.fromisoformat(str(stamp))
                break
            except (TypeError, ValueError):
                continue
        response = ask(question, snapshot, evidence, as_of=as_of)
        if response.answered:
            st.success(response.answer)
        else:
            st.warning(f"{response.status}: {response.reason}")
        with st.expander("Evidence and provenance"):
            st.write(response.to_dict())


def render_training(operator_id: str, context) -> None:
    st.subheader("Training Hub")
    content = load_content()
    gate_tab, lesson_tab, peer_tab, measure_tab = st.tabs(
        ["Gate", "Lesson, scenario and quiz", "Peer techniques", "Measured improvement"]
    )
    with gate_tab:
        render_training_gate(operator_id)
    with lesson_tab:
        render_lesson(content)
    with peer_tab:
        render_peer(context)
    with measure_tab:
        render_measurement()


def render_peer(context) -> None:
    st.caption(
        "How other operators handled a comparable task in comparable conditions. "
        "Sources are anonymised, only approved examples are shown, and the similarity "
        "score is always visible — an easy site is not advice for hard ground."
    )
    try:
        examples = load_peer_examples()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    columns = st.columns(2)
    floor = columns[0].slider("Minimum context similarity", 0.0, 1.0, 0.5, 0.05)
    same_site = columns[1].checkbox("Same site condition only", value=False)

    matches = find_similar(
        examples,
        task_type=context.task.task_type,
        machine_type=context.machine.machine_type,
        site_condition=context.site.surface if same_site else None,
        min_similarity=floor,
    )

    st.caption(
        f"Matching on task `{context.task.task_type}` and machine `{context.machine.machine_type}`"
        + (f", surface `{context.site.surface}`" if same_site else "")
    )

    if not matches:
        st.info(
            "No approved peer example is similar enough to this context. Nothing is shown "
            "rather than offering advice from conditions that do not match."
        )
        return

    for example in matches:
        with st.expander(
            f"{example.source_operator_anonymized_id} — similarity "
            f"{example.context_similarity_score:.2f}"
        ):
            st.write(example.technique_description)
            st.caption(
                f"Site: {example.site_condition} · attachment: {example.attachment_type} · "
                f"observed: {example.observed_metric}"
            )


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


@st.cache_data(show_spinner="Analysing this operator's recent sessions...")
def recent_behavior_history(synthetic_dir: str, operator_id: str, window: int) -> list:
    """The operator's most recent sessions, scored by the Behavioral Fingerprint.

    Only the last `window` are fetched because that is all the gate weighs --
    analysing the operator's whole history would cost minutes for no difference.
    """
    data = DashboardData(Path(synthetic_dir))
    sessions = data.sessions()
    recent = (
        sessions[sessions.operator_id == operator_id]
        .sort_values("start_timestamp")
        .session_id.tolist()[-window:]
    )
    history = []
    for session_id in recent:
        result = adapters.get_behavior(data.build_session_context(session_id))
        if result.available:
            history.append(result.value)
    return history


def render_training_gate(operator_id: str) -> None:
    thresholds = TrainingThresholds.from_settings()
    history = recent_behavior_history(
        str(DashboardData.from_settings().synthetic_dir), operator_id, thresholds.history_window
    )

    if not history:
        st.info("Behavioral Fingerprint is not integrated, so there is no history to gate on.")
    else:
        decision = evaluate_training_gate(
            history, operator_id=operator_id, issue_type="idle_reduction"
        )
        st.markdown(f"**Real gate decision for {operator_id}** — issue `idle_reduction`")
        st.caption(
            f"From the Behavioral Fingerprint's own output on this operator's last "
            f"{decision.considered} sessions. Not a simulation."
        )

        if decision.triggered:
            trigger = decision.trigger
            st.success(
                f"Training triggered — lesson **{trigger.lesson_id}**, "
                f"attribution **{trigger.attribution_type}**, confidence "
                f"**{trigger.confidence:.3f}**, escalation **{trigger.escalation_state}**."
            )
        else:
            st.info("No training triggered: " + ", ".join(decision.reasons))

        columns = st.columns(3)
        columns[0].metric("Qualifying occurrences",
                          f"{decision.qualifying_occurrences}/{decision.considered}")
        columns[1].metric("Needs at least", str(thresholds.min_occurrences))
        columns[2].metric("Confidence floor", f"{thresholds.min_confidence:.2f}")

        with st.expander("Per-session detail — why each one did or did not count"):
            st.dataframe(
                pd.DataFrame([
                    {
                        "session": o.session_id,
                        "attribution": o.attribution,
                        "confidence": round(o.confidence, 3),
                        "context share": o.context_share,
                        "counts": "yes" if o.qualifies else "no",
                        "why not": ", ".join(o.reasons),
                    }
                    for o in decision.occurrences
                ]),
                width="stretch", hide_index=True,
            )
            st.caption(
                "An occurrence counts only if it is operator-linked, confident, not "
                "primarily explained by context, and not in the operator's favour — "
                "all four at once."
            )

    st.divider()
    st.markdown("**Explore the gate with manual inputs**")
    st.caption(
        "A what-if explorer, not model output. Change these to see what would and would "
        "not trigger coaching."
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
    safe_section(render_header, context, operator, machine, events, name="Shift status")
    st.divider()
    safe_section(render_plan, context, name="Plan")
    st.divider()
    changes = ()
    try:
        changes = render_replan(data, sessions, session_id, context) or ()
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Replanning failed to render.** {type(exc).__name__}: {exc}")
    st.divider()
    safe_section(render_prediction, context, name="Prediction")
    st.divider()
    safe_section(render_conditions, context, name="Conditions")
    st.divider()
    safe_section(render_live_operation, data, session_id, context, events, name="Live operation")
    st.divider()
    safe_section(render_attention, events, changes, telemetry, name="Attention queue")
    st.divider()
    safe_section(render_buddy, context, telemetry, events, name="Operating Buddy")
    st.divider()
    safe_section(render_training, operator_id, context, name="Training Hub")
    st.divider()
    safe_section(render_end_of_shift, data, session_id, name="End of shift")


if __name__ == "__main__":
    main()
