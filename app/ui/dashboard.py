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

import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from features.dashboard.demo_selector import scenario_for_label, scenario_labels  # noqa: E402
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


@st.cache_data(show_spinner=False)
def task_meta(synthetic_dir: str) -> dict:
    """task_id -> {priority, deadline}. The optimizer's steps carry neither, so
    the Next Task card reads them from tasks.csv rather than going without."""
    tasks = load_table(synthetic_dir, "tasks.csv")
    return {
        row.task_id: {
            "priority": getattr(row, "priority", None),
            "deadline": getattr(row, "deadline_timestamp", None),
        }
        for row in tasks.itertuples(index=False)
    }


# --- presentation layer ------------------------------------------------------
# Streamlit cannot style these shapes natively, so a few small helpers return
# HTML strings. Tokens live in one :root block injected once by _inject_css().
#
# Two colour rules, kept deliberately strict:
#   --critical  only an active CRITICAL safety event or a hard-blocked task
#   --accent    system/active state (the NOW marker, the brand rule) -- not borders

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --bg:#171A1C; --panel:#202428; --panel-raised:#262B2F; --border:#33393D;
  --text:#EDEDE9; --text-muted:#9AA0A6;
  --accent:#F2B544; --ok:#7FA88F; --warn:#E0942C; --critical:#D34C3F;
}
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.mono { font-family:'IBM Plex Mono',ui-monospace,monospace; }

.deck-bar {
  display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;
  background:var(--panel); border:1px solid var(--border);
  border-left:3px solid var(--accent);
  padding:.7rem 1rem; margin-bottom:.4rem;
}
.deck-bar .brand { font-weight:600; letter-spacing:.02em; color:var(--text); }
.deck-bar .sep { color:var(--border); }
.deck-bar .lbl { color:var(--text-muted); font-size:.72rem; display:block; }
.deck-bar .val { font-family:'IBM Plex Mono',monospace; color:var(--text); font-size:.95rem; }
.deck-bar .spacer { flex:1; }

.badge {
  font-size:.74rem; padding:.2rem .55rem; border:1px solid var(--border);
  border-radius:2px; font-family:'IBM Plex Mono',monospace; white-space:nowrap;
}
.badge.ok       { color:var(--ok);       border-color:var(--ok); }
.badge.warn     { color:var(--warn);     border-color:var(--warn); }
.badge.critical { color:var(--critical); border-color:var(--critical); }
.badge.muted    { color:var(--text-muted); }

.panel {
  background:var(--panel); border:1px solid var(--border);
  padding:.85rem 1rem; margin-bottom:.5rem;
}
.panel.raised { background:var(--panel-raised); }
.panel h4 { margin:0 0 .5rem 0; font-size:.8rem; font-weight:600; color:var(--text-muted); }

.safety-clear {
  border-left:3px solid var(--ok); color:var(--ok);
  font-size:.92rem; padding:.6rem 1rem; background:var(--panel);
  border-top:1px solid var(--border); border-right:1px solid var(--border);
  border-bottom:1px solid var(--border);
}
.safety-alert { border-left:3px solid var(--critical); background:var(--panel); padding:.8rem 1rem;
  border-top:1px solid var(--border); border-right:1px solid var(--border);
  border-bottom:1px solid var(--border); }
.safety-alert.high { border-left-color:var(--warn); }
.safety-alert .head { font-weight:600; letter-spacing:.04em; margin-bottom:.35rem; }
.safety-alert.critical .head { color:var(--critical); }
.safety-alert.high .head { color:var(--warn); }
.safety-alert .facts { display:flex; gap:1.6rem; flex-wrap:wrap; margin-top:.5rem; }
.safety-alert .facts .lbl { color:var(--text-muted); font-size:.7rem; display:block; }
.safety-alert .facts .val { font-family:'IBM Plex Mono',monospace; color:var(--text); }

.steps { list-style:none; margin:0; padding:0; }
.steps li {
  display:flex; align-items:baseline; gap:.75rem;
  padding:.45rem 0; border-bottom:1px solid var(--border);
}
.steps li:last-child { border-bottom:none; }
.steps .idx { font-family:'IBM Plex Mono',monospace; color:var(--text-muted); width:1.4rem; }
.steps .tid { font-family:'IBM Plex Mono',monospace; color:var(--text); }
.steps .when { font-family:'IBM Plex Mono',monospace; color:var(--text-muted); font-size:.85rem; }
.steps .now { color:var(--accent); font-size:.68rem; letter-spacing:.09em;
  border:1px solid var(--accent); padding:.05rem .35rem; }
.steps .late { color:var(--warn); font-size:.7rem; }
.steps .grow { flex:1; }

.rb { margin-bottom:.9rem; }
.rb .top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.3rem; }
.rb .name { color:var(--text-muted); font-size:.78rem; }
.rb .p50 { font-family:'IBM Plex Mono',monospace; font-size:1.25rem; color:var(--text); }
.rb .track { position:relative; height:6px; background:var(--panel-raised);
  border:1px solid var(--border); }
.rb .fill { position:absolute; top:0; bottom:0; background:rgba(242,181,68,.22); }
.rb .mark { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--accent); }
.rb .ends { display:flex; justify-content:space-between; margin-top:.25rem;
  font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--text-muted); }

.cond { display:flex; gap:1.8rem; flex-wrap:wrap; }
.cond .item .lbl { color:var(--text-muted); font-size:.7rem; display:block; }
.cond .item .val { font-family:'IBM Plex Mono',monospace; color:var(--text); font-size:.9rem; }

.log { list-style:none; margin:0; padding:0; }
.log li { display:flex; gap:.7rem; align-items:baseline; padding:.3rem 0;
  border-bottom:1px solid var(--border); font-size:.86rem; }
.log li:last-child { border-bottom:none; }
.log .t { font-family:'IBM Plex Mono',monospace; color:var(--text-muted); }
.log .dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.log .reason { color:var(--text); }
</style>
"""

_SEVERITY_CLASS = {"CRITICAL": "critical", "HIGH": "warn", "MEDIUM": "warn"}
_SEVERITY_COLOR = {
    "CRITICAL": "var(--critical)", "HIGH": "var(--warn)", "MEDIUM": "var(--warn)",
    "LOW": "var(--text-muted)", "INFO": "var(--text-muted)",
}


SCENARIO_CARD_TEMPLATE = Path(__file__).with_name("demo_selector.html")


def render_scenario_card(scenario) -> str:
    """Fill the sidebar scenario card. The markup lives in demo_selector.html so
    the styling stays out of Python; it inherits the :root tokens.

    Placeholders are replaced rather than str.format()ed -- the template carries
    a <style> block, and every CSS brace would otherwise be read as a field.
    """
    markup = SCENARIO_CARD_TEMPLATE.read_text(encoding="utf-8")
    for field_name, value in (
        ("category", scenario.category), ("title", scenario.title),
        ("session_id", scenario.session_id), ("operator_id", scenario.operator_id),
        ("show", scenario.show), ("say", scenario.say),
    ):
        markup = markup.replace("{" + field_name + "}", _esc(value))
    return markup


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


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


def _worst_severity(events: pd.DataFrame) -> str:
    if events.empty:
        return "NONE"
    present = set(events.severity)
    return next((s for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO") if s in present), "NONE")


def render_header(context, operator, machine, events: pd.DataFrame) -> None:
    authorization = check_authorization(operator, machine) if operator and machine else None
    worst = _worst_severity(events)

    if authorization is None:
        auth_badge = '<span class="badge muted">AUTHORIZATION UNKNOWN</span>'
    elif authorization.authorized:
        auth_badge = '<span class="badge ok">AUTHORIZED &#10003;</span>'
    else:
        auth_badge = '<span class="badge critical">REFUSED</span>'

    severity_class = {"CRITICAL": "critical", "HIGH": "warn", "MEDIUM": "warn"}.get(worst, "muted")
    field = '<div><span class="lbl">{}</span><span class="val">{}</span></div>'

    _html(
        '<div class="deck-bar">'
        '<span class="brand">CAT OPERATOR COPILOT</span>'
        '<span class="sep">|</span>'
        + field.format("operator", _esc(context.operator_id))
        + field.format(
            "machine",
            f"{_esc(context.machine_id)} &middot; {_esc(context.machine.machine_type or '')}",
        )
        + field.format(
            "task", f"{_esc(context.task_id or '—')} &middot; {_esc(context.task.task_type or '')}"
        )
        + field.format("session", _esc(context.session_id))
        + '<span class="spacer"></span>'
        + auth_badge
        + f'<span class="badge {severity_class}">SAFETY {_esc(worst)}</span>'
        "</div>"
    )

    if authorization is not None and not authorization.authorized:
        st.error(
            "Authorization refused: " + ", ".join(authorization.reasons)
            + ". No session may be opened for this operator on this machine."
        )


def render_safety(events: pd.DataFrame) -> None:
    """Quiet by default; loud only when something is actually wrong.

    Reads the same safety_events frame the live-operation section uses -- this
    surfaces it earlier and larger, it does not re-decide anything. Severity and
    required_action come from the Safety Guardian's own columns.
    """
    urgent = events[events.severity.isin(["CRITICAL", "HIGH"])] if not events.empty else events

    if urgent.empty:
        count = len(events)
        tail = f" &middot; {count} lower-severity event(s) logged" if count else ""
        _html(f'<div class="safety-clear">CLEAR &mdash; no critical or high safety event'
              f' in this session{tail}</div>')
        return

    for event in urgent.to_dict(orient="records"):
        severity = str(event.get("severity", "")).upper()
        tone = "critical" if severity == "CRITICAL" else "high"
        facts = [
            ("worker distance", f"{event.get('worker_distance_m', '—')} m"),
            ("closing speed", f"{event.get('closing_speed_mps', '—')} m/s"),
            ("machine state", event.get("machine_state", "—")),
            ("required action", event.get("required_action", "—")),
            ("recorded", str(event.get("timestamp", ""))[11:19]),
        ]
        body = "".join(
            f'<div><span class="lbl">{_esc(label)}</span>'
            f'<span class="val">{_esc(value)}</span></div>'
            for label, value in facts
        )
        _html(
            f'<div class="safety-alert {tone}">'
            f'<div class="head">{_esc(severity)} &mdash; {_esc(event.get("event_type", "safety event"))}</div>'
            f'<div>{_esc(event.get("trigger_reason", ""))}</div>'
            f'<div class="facts">{body}</div>'
            "</div>"
        )

    st.caption(
        "Recorded by the Safety Guardian, which is deterministic and rule-based. "
        "Distance alone never decides severity — motion, swing path and closing speed do."
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
    meta = task_meta(str(DashboardData.from_settings().synthetic_dir))

    if not steps:
        st.info(
            "No feasible task for this operator and machine in this shift. "
            "Check the exclusions below — the fatigue budget and CRITICAL safety "
            "findings are hard blocks, not costs."
        )
    else:
        items = []
        for index, step in enumerate(steps, start=1):
            now = '<span class="now">NOW</span>' if index == 1 else ""
            late = "" if step.get("deadline_met", True) else '<span class="late">past deadline</span>'
            at_risk = (
                '<span class="late">deadline at risk</span>'
                if step.get("deadline_at_risk") and step.get("deadline_met", True) else ""
            )
            items.append(
                f'<li><span class="idx">{index}</span>'
                f'<span class="tid">{_esc(step.get("task_id"))}</span>'
                f'{now}<span class="grow"></span>{late}{at_risk}'
                f'<span class="when">{_clock(step.get("start"))}&ndash;{_clock(step.get("end_p50"))}</span>'
                f'<span class="when">{step.get("eta_min", {}).get("p50", 0):.0f} min</span></li>'
            )
        _html(f'<div class="panel"><h4>Ordered sequence</h4><ul class="steps">{"".join(items)}</ul></div>')

        first = steps[0]
        eta, fuel = first.get("eta_min", {}), first.get("fuel_l", {})
        info = meta.get(first.get("task_id"), {})
        priority = info.get("priority")
        deadline = str(info.get("deadline") or "")[:16].replace("T", " ")
        rows = [
            ("priority", f"P{priority}" if priority is not None else "—"),
            ("deadline", deadline or "—"),
            ("starts", _clock(first.get("start"))),
            ("ETA P50", f"{eta.get('p50', 0):.0f} min"),
            ("fuel P50", f"{fuel.get('p50', 0):.1f} L"),
            ("confidence", f"{first.get('prediction_confidence', 0):.2f}"),
        ]
        body = "".join(
            f'<div class="item"><span class="lbl">{_esc(k)}</span>'
            f'<span class="val">{_esc(v)}</span></div>' for k, v in rows
        )
        _html(
            f'<div class="panel raised"><h4>Next task &mdash; '
            f'<span class="mono">{_esc(first.get("task_id"))}</span></h4>'
            f'<div class="cond">{body}</div></div>'
        )
        if first.get("reason"):
            st.caption(f"Chosen because: {first['reason']}")

    st.markdown(f"**Why this order:** {plan.get('reason', 'not stated')}")

    totals = [
        ("total cost", f"{plan.get('total_cost', 0):.0f}"),
        ("deadlines met", str(plan.get("deadlines_met", "—"))),
        ("shift planned", f"{plan.get('shift_minutes_planned', 0):.0f} min"),
    ]
    _html(
        '<div class="panel"><div class="cond">'
        + "".join(
            f'<div class="item"><span class="lbl">{_esc(k)}</span>'
            f'<span class="val">{_esc(v)}</span></div>' for k, v in totals
        )
        + "</div></div>"
    )

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
    bars = _range_bar("Estimated duration", prediction.eta_p10, prediction.eta_p50,
                      prediction.eta_p90, "min")
    # A zeroed percentile means this call predicted nothing for that quantity;
    # rendering 0.0 would read as a prediction of zero.
    bars += _range_bar("Fuel", prediction.fuel_p10, prediction.fuel_p50,
                       prediction.fuel_p90, "L")
    _html(
        f'<div class="panel">{bars}'
        f'<div class="cond"><div class="item"><span class="lbl">confidence</span>'
        f'<span class="val">{prediction.confidence:.2f}</span></div></div></div>'
    )
    if prediction.factors:
        st.caption("Contributing factors: " + ", ".join(prediction.factors)
                   + " — contribution to the prediction, not causes.")


def _range_bar(label: str, p10, p50, p90, unit: str) -> str:
    """The page's one instrument: a P10-P90 track with the P50 marked.

    A zeroed P50 means this quantity was not predicted, which is shown as such
    rather than as a bar sitting at zero.
    """
    if not p50:
        return (f'<div class="rb"><div class="top"><span class="name">{_esc(label)}</span>'
                f'<span class="p50">not predicted</span></div></div>')

    low, high = float(p10 or 0), float(p90 or 0)
    span = max(high - low, 1e-9)
    marker = min(max((float(p50) - low) / span, 0.0), 1.0) * 100
    return (
        f'<div class="rb">'
        f'<div class="top"><span class="name">{_esc(label)}</span>'
        f'<span class="p50">{float(p50):.1f} <span class="name">{_esc(unit)}</span></span></div>'
        f'<div class="track"><div class="fill" style="left:0;right:0"></div>'
        f'<div class="mark" style="left:{marker:.1f}%"></div></div>'
        f'<div class="ends"><span>P10 {low:.1f}</span>'
        f'<span>P50 {float(p50):.1f}</span><span>P90 {high:.1f}</span></div></div>'
    )


def render_conditions(context) -> None:
    items = [
        ("rain", f"{context.weather.rain_mm} mm"),
        ("temp", f"{context.weather.temperature_c} C"),
        ("visibility", f"{context.weather.visibility_m} m"),
        ("dust", context.weather.dust_level),
        ("light", context.weather.day_night),
        ("soil", context.site.soil_material),
        ("hardness", context.site.hardness),
        ("slope", f"{context.site.slope} deg"),
        ("surface", context.site.surface),
        ("congestion", context.traffic.congestion),
        ("machine", context.machine.machine_condition),
        ("attachment", context.machine.attachment),
    ]
    body = "".join(
        f'<div class="item"><span class="lbl">{_esc(k)}</span>'
        f'<span class="val">{_esc(v)}</span></div>' for k, v in items
    )
    _html(f'<div class="panel"><h4>Conditions</h4><div class="cond">{body}</div></div>')


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

        # Match the loudness to the severity. Showing an INFO event in critical
        # red -- "worker outside envelope, machine stationary, none_required" --
        # says the opposite of what the Safety Guardian decided, and undercuts
        # the whole point that proximity alone is not danger.
        for event in step.safety_events:
            severity = str(event.get("severity", "")).upper()
            line = (f"{severity} — {event.get('trigger_reason')} "
                    f"(action: {event.get('required_action')})")
            if severity == "CRITICAL":
                st.error(line)
            elif severity in ("HIGH", "MEDIUM"):
                st.warning(line)
            else:
                st.caption(line)
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
        st.markdown("**Event log**")
        entries = "".join(
            f'<li><span class="t">{_esc(str(e.get("timestamp",""))[11:19])}</span>'
            f'<span class="dot" style="background:'
            f'{_SEVERITY_COLOR.get(str(e.get("severity","")).upper(), "var(--text-muted)")}"></span>'
            f'<span class="t">{_esc(e.get("severity",""))}</span>'
            f'<span class="reason">{_esc(e.get("trigger_reason",""))}</span></li>'
            for e in events.sort_values("timestamp").to_dict(orient="records")
        )
        _html(f'<ul class="log">{entries}</ul>')
        with st.expander("Full event records"):
            st.dataframe(
                events[["timestamp", "severity", "event_type", "trigger_reason",
                        "worker_distance_m", "closing_speed_mps", "machine_state"]],
                width="stretch", hide_index=True,
            )
            st.caption(
                "Recorded in the dataset. Live evaluation is the Safety Guardian's decision, "
                "which is deterministic and owned by Member 2."
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
            # Warning, not error: a missed quiz is a result to act on, not a
            # critical condition. Red is reserved for safety.
            st.warning(f"Not passed — {result.correct}/{result.total} "
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
    _inject_css()
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
        st.header("Demo scenarios")
        scenario = None
        labels = scenario_labels()
        choice = st.selectbox(
            "Jump to a scenario", ["— browse sessions manually —"] + labels, index=1,
            help="Twenty curated sessions, each showing one thing. Verified by "
                 "tests/demo_scenarios.py.",
        )
        if choice in labels:
            scenario = scenario_for_label(choice)
            _html(render_scenario_card(scenario))

        st.divider()
        st.header("Session")
        operators = sorted(sessions.operator_id.unique())
        operator_default = operators.index(scenario.operator_id) if scenario else 0
        operator_id = st.selectbox("Operator", operators, index=operator_default)
        operator_sessions = sessions[sessions.operator_id == operator_id]
        session_ids = operator_sessions.session_id.tolist()
        session_default = (
            session_ids.index(scenario.session_id)
            if scenario and scenario.session_id in session_ids else 0
        )
        session_id = st.selectbox("Session", session_ids, index=session_default)
        st.caption(f"{len(operator_sessions)} sessions for {operator_id}")

    context = data.build_session_context(session_id)
    repository = data.repository()
    operator = repository.get_operator(operator_id)
    machine = repository.get_machine(context.machine_id)

    events = load_table(synthetic_dir, "safety_events.csv")
    events = events[events.session_id == session_id]
    telemetry = load_table(synthetic_dir, "telemetry.csv")
    telemetry = telemetry[telemetry.session_id == session_id]

    # Priority order: safety, then what to do, then how it should go, then
    # context, then what has happened. Secondary features sit below in tabs so
    # they stay fully available without competing for attention.
    safe_section(render_header, context, operator, machine, events, name="Shift status")
    safe_section(render_safety, events, name="Safety")

    plan_column, next_column = st.columns([3, 2], gap="medium")
    with plan_column:
        safe_section(render_plan, context, name="Plan")
    with next_column:
        safe_section(render_prediction, context, name="Prediction")

    safe_section(render_conditions, context, name="Conditions")

    changes = ()
    try:
        changes = render_replan(data, sessions, session_id, context) or ()
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Replanning failed to render.** {type(exc).__name__}: {exc}")

    safe_section(render_live_operation, data, session_id, context, events, name="Live operation")

    attention_tab, buddy_tab, training_tab, shift_tab = st.tabs(
        ["Attention queue", "Operating Buddy", "Training Hub", "End of shift"]
    )
    with attention_tab:
        safe_section(render_attention, events, changes, telemetry, name="Attention queue")
    with buddy_tab:
        safe_section(render_buddy, context, telemetry, events, name="Operating Buddy")
    with training_tab:
        safe_section(render_training, operator_id, context, name="Training Hub")
    with shift_tab:
        safe_section(render_end_of_shift, data, session_id, name="End of shift")

    render_integration_strip()


if __name__ == "__main__":
    main()
