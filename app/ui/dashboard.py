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
from features.buddy.evidence import (  # noqa: E402
    SOURCE_SAFETY_INCIDENT,
    SOURCE_TASK_PLAN,
    SOURCE_TELEMETRY,
    Evidence,
)
from features.buddy.safe_state import MachineStateSnapshot, evaluate_safe_state  # noqa: E402
from features.dashboard import adapters  # noqa: E402
from features.dashboard.data import DashboardData, MissingDatasetError  # noqa: E402
from features.passport.authorization import check_authorization  # noqa: E402
from features.training.trigger import evaluate_training_gate  # noqa: E402
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


def render_buddy(context, telemetry: pd.DataFrame) -> None:
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

    question = st.text_input("Ask the Buddy", "Why did my task order change?")
    if st.button("Ask"):
        evidence = [
            Evidence(source=SOURCE_TASK_PLAN, value=context.task_id,
                     content=f"The current task is {context.task_id} ({context.task.task_type}).",
                     timestamp=context.timestamp, confidence=0.8),
            Evidence(source=SOURCE_TELEMETRY, value=row.machine_state,
                     content=f"The machine is currently {row.machine_state}.",
                     timestamp=context.timestamp, confidence=0.9),
        ]
        response = ask(question, snapshot, evidence)
        if response.answered:
            st.success(response.answer)
        else:
            st.warning(f"{response.status}: {response.reason}")
        with st.expander("Evidence and provenance"):
            st.write(response.to_dict())


def render_training_gate(operator_id: str) -> None:
    st.subheader("Training gate")
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
    render_buddy(context, telemetry)
    st.divider()
    render_training_gate(operator_id)
    st.divider()
    render_end_of_shift(data, session_id)


if __name__ == "__main__":
    main()
