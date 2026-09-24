"""
Feature 02 — Safety Guardian
Owner: Aneesha (Team Member 2)

Deterministic, rule-based safety evaluation. Every decision (severity,
required_action) is a threshold comparison defined in config/safety_rules.yaml,
so the same input always yields the same output and every alert can be traced
to a named rule. No LLM participates in any decision.

The only LLM use is context_aware_safety_reasoning(): an optional, advisory
explanation for HIGH/MEDIUM events. It cannot change the decision, is never
called for CRITICAL, and falls back to the rule's own text when the local LLM
is unavailable or its answer contradicts the rule.

Public functions:
  evaluate_safety(session_context)  → SafetyEvent          (most severe; API contract)
  evaluate_all(session_context)     → List[SafetyEvent]    (every rule that fired)
  evaluate_record(record)           → List[SafetyEvent]    (flat dict, e.g. a telemetry row)
  evaluate_records(records)         → List[List[SafetyEvent]] (batch of flat dicts, one pass)
  scan_telemetry(...)               → List[SafetyEvent]    (batch over telemetry.csv)
  load_logged_events(...)           → List[SafetyEvent]    (safety_events.csv as SafetyEvents)
  validate_against_logged(...)      → dict                 (rule engine vs. logged labels)
  context_aware_safety_reasoning(rule_decision, context_dict) → dict (decision + advisory reasoning)
"""

import os
import sys
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.schemas import SafetyEvent, SessionContext
from shared.constants import DATA_QUALITY_CONFIDENCE_DEFAULT, SEVERITY_LEVELS
from features.llm.ollama_client import OllamaClient

RULES_PATH = os.path.join(ROOT, "config", "safety_rules.yaml")
DATA_DIR   = os.path.join(ROOT, "data", "synthetic")

_rules_cache: Dict[str, dict] = {}


# ── Rule loading ──────────────────────────────────────────────────────────────

def load_rules(path: str = RULES_PATH) -> dict:
    """Load and validate the YAML rule file (cached per path)."""
    if path not in _rules_cache:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for rule in cfg["rules"]:
            if rule["severity"] not in SEVERITY_LEVELS:
                raise ValueError(f"Rule {rule['id']}: unknown severity {rule['severity']}")
            for cond in rule["conditions"]:
                if cond["op"] not in _OPS:
                    raise ValueError(f"Rule {rule['id']}: unknown op {cond['op']}")
                if "ref" in cond:
                    _resolve_ref(cfg, cond["ref"])  # fail fast on typos
        _rules_cache[path] = cfg
    return _rules_cache[path]


def _resolve_ref(cfg: dict, ref: str) -> Any:
    node = cfg
    for part in ref.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"safety_rules.yaml: unresolved ref '{ref}'")
        node = node[part]
    return node


def _severity_rank(cfg: dict, severity: str) -> int:
    return cfg["severity_order"].index(severity)


# ── Condition evaluation (vectorised; one code path for single + batch) ──────

def _is_true(col: pd.Series) -> pd.Series:
    return col.map(lambda v: v is True or v == 1 or (isinstance(v, str) and v.lower() == "true"))


_OPS = {
    "gt":       lambda c, v: c > v,
    "gte":      lambda c, v: c >= v,
    "lt":       lambda c, v: c < v,
    "lte":      lambda c, v: c <= v,
    "eq":       lambda c, v: c == v,
    "ne":       lambda c, v: c != v,
    "in":       lambda c, v: c.isin(v),
    "not_in":   lambda c, v: ~c.isin(v),
    "is_true":  lambda c, v: _is_true(c),
    "is_false": lambda c, v: ~_is_true(c),
}


def _condition_mask(df: pd.DataFrame, cfg: dict, cond: dict) -> pd.Series:
    field = cond["field"]
    if field not in df.columns:
        return pd.Series(False, index=df.index)
    col = df[field]
    present = col.notna()
    value = _resolve_ref(cfg, cond["ref"]) if "ref" in cond else cond.get("value")
    if cond["op"] in ("gt", "gte", "lt", "lte"):
        col = pd.to_numeric(col, errors="coerce")
        present &= col.notna()
    return present & _OPS[cond["op"]](col, value).fillna(False).astype(bool)


def _rule_mask(df: pd.DataFrame, cfg: dict, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in rule["conditions"]:
        mask &= _condition_mask(df, cfg, cond)
    return mask


# ── Fact derivation ───────────────────────────────────────────────────────────

def derive_facts(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Add the derived fields the rules reference. Thresholds come from the YAML;
    this function only reshapes raw fields (units, max/min, set membership).
    """
    cfg = cfg or load_rules()
    df = df.copy()

    def num(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    if "shift_hours" not in df.columns and "shift_elapsed_min" in df.columns:
        df["shift_hours"] = num("shift_elapsed_min") / 60.0

    if "machine_active" not in df.columns and "machine_state" in df.columns:
        active = set(cfg["machine_state"]["active_states"])
        df["machine_active"] = df["machine_state"].where(df["machine_state"].notna()).map(
            lambda s: None if s is None or (isinstance(s, float) and np.isnan(s)) else s in active)

    # Telemetry has no explicit envelope flag: a worker is "in envelope" when a
    # swinging machine has them inside the envelope radius.
    if "machine_state" in df.columns and "worker_distance_m" in df.columns:
        derived_env = (df["machine_state"] == "swinging") & \
                      (num("worker_distance_m") <= cfg["proximity"]["envelope_radius_m"])
        if "worker_in_envelope" in df.columns:
            df["worker_in_envelope"] = df["worker_in_envelope"].where(df["worker_in_envelope"].notna(), derived_env)
        else:
            df["worker_in_envelope"] = derived_env

    df["min_health_score"] = pd.concat([num("engine_health_score"), num("hydraulic_health_score")], axis=1).min(axis=1)
    df["max_load_pct"]     = pd.concat([num("engine_load_pct"), num("hydraulic_load_pct")], axis=1).max(axis=1)
    return df


# ── Core evaluation ───────────────────────────────────────────────────────────

_EVIDENCE_FIELDS = [
    "worker_distance_m", "closing_speed_mps", "machine_state", "machine_speed_kmh",
    "seatbelt_status", "shift_hours", "site_slope_deg", "engine_health_score",
    "hydraulic_health_score", "maintenance_status", "fuel_level_pct", "rain_mm",
    "visibility_m", "wind_speed_kmh", "heat_index_c", "dust_level", "congestion_level",
    "fatigue_proxy", "max_load_pct",
]


def _event_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10].upper()
    return f"SG-{digest}"


def _clean(v: Any) -> Any:
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.bool_,)):    return bool(v)
    if isinstance(v, float) and np.isnan(v): return None
    return v


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _confidence(row: Dict[str, Any], cfg: dict) -> float:
    base = _num(row.get("data_quality_confidence"))
    if base is None:
        base = _num(row.get("confidence"))
    if base is None:
        base = DATA_QUALITY_CONFIDENCE_DEFAULT
    vis = _num(row.get("visibility_m"))
    degraded = (vis is not None and vis < cfg["visibility"]["low_visibility_m"]) or                row.get("dust_level") == cfg["visibility"]["dust_high"]
    if degraded:
        base *= cfg["visibility"]["confidence_penalty"]
    return round(base, 3)


def _build_event(row: Dict[str, Any], rule: dict, cfg: dict) -> SafetyEvent:
    ts = row.get("timestamp")
    ts = ts if isinstance(ts, str) and ts else datetime.now().isoformat(timespec="seconds")
    evidence = {f: _clean(row[f]) for f in _EVIDENCE_FIELDS
                if f in row and _clean(row[f]) is not None}
    return SafetyEvent(
        event_id=_event_id(row.get("session_id"), row.get("machine_id"), ts, rule["id"]),
        severity=rule["severity"],
        trigger_reason=rule.get("trigger_reason", rule["id"]),
        required_action=rule["required_action"],
        confidence=_confidence(row, cfg),
        timestamp=ts,
        synthetic_flag=bool(_clean(row.get("synthetic_flag", True))),
        recommendation=rule["recommendation"],
        session_id=_clean(row.get("session_id")),
        operator_id=_clean(row.get("operator_id")),
        machine_id=_clean(row.get("machine_id")),
        rule_id=rule["id"],
        evidence=evidence,
    )


def _evaluate_frame(df: pd.DataFrame, cfg: dict,
                    rules: Optional[Iterable[dict]] = None) -> pd.DataFrame:
    """Return a boolean frame (rows × rule ids) of fired rules, after group de-dup."""
    rules = list(rules if rules is not None else cfg["rules"])
    fired = pd.DataFrame({r["id"]: _rule_mask(df, cfg, r) for r in rules}, index=df.index)

    # Within a group keep only the most severe fired rule (ties: YAML order).
    groups: Dict[str, List[dict]] = {}
    for r in rules:
        if "group" in r:
            groups.setdefault(r["group"], []).append(r)
    for members in groups.values():
        members = sorted(members, key=lambda r: -_severity_rank(cfg, r["severity"]))
        already = pd.Series(False, index=df.index)
        for r in members:
            fired[r["id"]] &= ~already
            already |= fired[r["id"]]
    return fired


def _events_from_frame(df: pd.DataFrame, fired: pd.DataFrame, cfg: dict) -> List[SafetyEvent]:
    by_id = {r["id"]: r for r in cfg["rules"]}
    any_hit = fired.any(axis=1)
    rows = dict(zip(df.index[any_hit], df[any_hit].to_dict("records")))  # one conversion, not per event
    events = []
    for rule_id in fired.columns:
        for idx in fired.index[fired[rule_id]]:
            events.append(_build_event(rows[idx], by_id[rule_id], cfg))
    events.sort(key=lambda e: (-_severity_rank(cfg, e.severity), e.timestamp))
    return events


def evaluate_records(records: List[Dict[str, Any]], cfg: Optional[dict] = None) -> List[List[SafetyEvent]]:
    """Evaluate many flat records in one vectorised pass. One list per record, most severe first."""
    cfg = cfg or load_rules()
    if not records:
        return []
    df = derive_facts(pd.DataFrame(records), cfg)
    fired = _evaluate_frame(df, cfg)
    by_id = {r["id"]: r for r in cfg["rules"]}
    rows = df.to_dict("records")
    out = []
    for i, idx in enumerate(df.index):
        events = [_build_event(rows[i], by_id[rid], cfg) for rid in fired.columns[fired.loc[idx].values]]
        events.sort(key=lambda e: -_severity_rank(cfg, e.severity))
        out.append(events)
    return out


def evaluate_record(record: Dict[str, Any], cfg: Optional[dict] = None) -> List[SafetyEvent]:
    """Evaluate one flat record (telemetry-style field names). Most severe first."""
    return evaluate_records([record], cfg)[0]


# ── SessionContext adapter ────────────────────────────────────────────────────

def context_to_record(ctx: SessionContext) -> Dict[str, Any]:
    """Flatten a SessionContext into the telemetry field names the rules use."""
    def g(obj, attr):
        if obj is None: return None
        return obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)

    m, op, site, tr = ctx.machine, ctx.operator, ctx.site, ctx.traffic
    wx, tmp, dq = ctx.weather, ctx.temporal, ctx.data_quality
    maint = g(m, "maintenance_indicators") or {}

    return {
        "session_id": ctx.session_id, "operator_id": ctx.operator_id,
        "machine_id": ctx.machine_id, "timestamp": ctx.timestamp or g(tmp, "timestamp"),
        # machine
        "machine_state":      g(m, "engine_state"),
        "machine_speed_kmh":  g(m, "speed"),
        "fuel_level_pct":     g(m, "fuel_level"),
        "engine_load_pct":    g(m, "load"),
        "seatbelt_status":    maint.get("seatbelt_status"),
        "hydraulic_load_pct": maint.get("hydraulic_load_pct"),
        "engine_health_score":    maint.get("engine_health_score"),
        "hydraulic_health_score": maint.get("hydraulic_health_score"),
        "maintenance_status":     maint.get("maintenance_status"),
        # operator
        "fatigue_proxy":      g(op, "fatigue_proxy"),
        # site / traffic
        "site_slope_deg":     g(site, "slope"),
        "worker_distance_m":  g(tr, "worker_proximity"),
        "closing_speed_mps":  g(tr, "closing_speed"),
        "congestion_level":   g(tr, "congestion") or g(site, "congestion"),
        "worker_in_envelope": maint.get("worker_in_envelope"),
        # weather
        "rain_mm":        g(wx, "rain_mm"),
        "visibility_m":   g(wx, "visibility_m"),
        "wind_speed_kmh": g(wx, "wind_speed_kmh"),
        "heat_index_c":   g(wx, "heat_index_c"),
        "dust_level":     g(wx, "dust_level"),
        # temporal
        "shift_elapsed_min": g(tmp, "shift_elapsed_min"),
        # data quality
        "synthetic_flag":          g(dq, "synthetic_flag") if dq else True,
        "data_quality_confidence": g(dq, "confidence") if dq else None,
    }


def evaluate_all(session_context: SessionContext) -> List[SafetyEvent]:
    """Every safety rule that fires for this context, most severe first."""
    return evaluate_record(context_to_record(session_context))


def evaluate_safety(session_context: SessionContext) -> SafetyEvent:
    """
    API contract entry point: the single most severe SafetyEvent.
    Returns an INFO 'no_hazard_detected' event when no rule fires.
    """
    events = evaluate_all(session_context)
    if events:
        return events[0]
    rec = context_to_record(session_context)
    ts = rec["timestamp"] or datetime.now().isoformat(timespec="seconds")
    return SafetyEvent(
        event_id=_event_id(rec["session_id"], rec["machine_id"], ts, "no_hazard_detected"),
        severity="INFO", trigger_reason="no_hazard_detected", required_action="none_required",
        confidence=_confidence(rec, load_rules()), timestamp=ts,
        synthetic_flag=bool(rec["synthetic_flag"]),
        recommendation="No safety rule triggered.",
        session_id=rec["session_id"], operator_id=rec["operator_id"], machine_id=rec["machine_id"],
        rule_id=None,
    )


# ── Advisory LLM reasoning (never decides) ────────────────────────────────────

_REASONING_SYSTEM = (
    "You are a safety assistant for a heavy-equipment operator. A deterministic safety rule "
    "has already decided the severity and required action. Do NOT change, question, soften or "
    "second-guess that decision. In at most two short sentences, explain why the action matters "
    "given the site context provided. Use only the facts given; do not invent any."
)

_llm_client: Optional[OllamaClient] = None


def _get_llm_client() -> OllamaClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaClient()
    return _llm_client


def _rule_reasoning(ev: SafetyEvent) -> str:
    facts = ", ".join(f"{k}={v}" for k, v in (ev.evidence or {}).items())
    base = ev.recommendation or ev.trigger_reason
    return f"{base} (rule {ev.rule_id or ev.trigger_reason}{'; ' + facts if facts else ''})"


def context_aware_safety_reasoning(rule_decision: SafetyEvent,
                                   context_dict: Optional[Dict[str, Any]] = None,
                                   client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    """
    Add a contextual, human-readable reasoning string to a rule decision.

    The decision itself (severity, required_action, trigger_reason) is always
    copied from `rule_decision` unchanged; `decision_changed` is always False.
      * CRITICAL, LOW, INFO → rule-based reasoning only; the LLM is not called.
      * HIGH, MEDIUM        → local LLM explains the decision in context.
    Falls back to rule-based reasoning when the LLM is unavailable, errors,
    returns nothing, or returns text that contradicts the rule.
    """
    cfg = load_rules()["llm_reasoning"]
    rule_text = _rule_reasoning(rule_decision)
    result = {
        "event_id": rule_decision.event_id,
        "severity": rule_decision.severity,
        "required_action": rule_decision.required_action,
        "trigger_reason": rule_decision.trigger_reason,
        "rule_id": rule_decision.rule_id,
        "decision_changed": False,
        "reasoning": rule_text,
        "reasoning_source": "rules",
        "llm_note": None,
        "event": rule_decision,
    }

    if rule_decision.severity not in cfg["enabled_severities"]:
        result["llm_note"] = f"LLM not used for {rule_decision.severity} events"
        return result

    context = {**(rule_decision.evidence or {}), **(context_dict or {})}
    prompt = (
        f"Rule decision: severity={rule_decision.severity}, "
        f"required_action={rule_decision.required_action}, reason={rule_decision.trigger_reason}.\n"
        f"Rule recommendation: {rule_decision.recommendation}\n"
        "Site context:\n" + "\n".join(f"- {k}: {v}" for k, v in context.items() if v is not None)
    )
    gen = (client or _get_llm_client()).generate(prompt, system=_REASONING_SYSTEM,
                                                 fallback=lambda _: rule_text)
    if gen.source != "ollama":
        result["llm_note"] = f"LLM unavailable: {gen.error}"
        return result

    text = " ".join(gen.text.split())[: cfg["max_chars"]]
    contradiction = next((p for p in cfg["rejected_phrases"] if p in text.lower()), None)
    if not text or contradiction:
        result["llm_note"] = (f"LLM text discarded: contradicts rule ('{contradiction}')"
                              if contradiction else "LLM returned empty text")
        return result

    result.update(reasoning=f"{rule_decision.recommendation} {text}".strip(),
                  reasoning_source="llm", llm_note=f"advisory text from {gen.model}")
    return result


# ── Batch: telemetry.csv ──────────────────────────────────────────────────────

def _enrich_telemetry(tel: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Join machine health/maintenance and operator fatigue proxy onto telemetry."""
    machines  = pd.read_csv(os.path.join(data_dir, "machines.csv"))
    operators = pd.read_csv(os.path.join(data_dir, "operators.csv"))
    tel = tel.merge(machines[["machine_id", "engine_health_score", "hydraulic_health_score",
                              "maintenance_status"]], on="machine_id", how="left")
    tel = tel.merge(operators[["operator_id", "fatigue_proxy"]], on="operator_id", how="left")
    return tel


def scan_telemetry(telemetry: Optional[pd.DataFrame] = None,
                   data_dir: str = DATA_DIR,
                   min_severity: str = "LOW",
                   first_per_session: bool = True) -> List[SafetyEvent]:
    """
    Run all rules across telemetry. With first_per_session=True, each rule
    fires at most once per session (the first time it is breached) so a
    10-hour shift doesn't generate thousands of identical fatigue alerts.
    """
    cfg = load_rules()
    if telemetry is None:
        telemetry = pd.read_csv(os.path.join(data_dir, "telemetry.csv"))
    df = derive_facts(_enrich_telemetry(telemetry, data_dir), cfg)
    df = df.sort_values("timestamp").reset_index(drop=True)

    floor = _severity_rank(cfg, min_severity)
    rules = [r for r in cfg["rules"] if _severity_rank(cfg, r["severity"]) >= floor]
    fired = _evaluate_frame(df, cfg, rules)

    if first_per_session and "session_id" in df.columns:
        for rule_id in fired.columns:
            hits = fired[rule_id]
            first = ~df.loc[hits, "session_id"].duplicated()
            fired[rule_id] = False
            fired.loc[first.index[first], rule_id] = True
    return _events_from_frame(df, fired, cfg)


# ── safety_events.csv ─────────────────────────────────────────────────────────

def load_logged_events(path: Optional[str] = None) -> List[SafetyEvent]:
    """Load historical safety_events.csv rows as SafetyEvent objects."""
    df = pd.read_csv(path or os.path.join(DATA_DIR, "safety_events.csv"))
    cfg = load_rules()
    by_reason = {r.get("trigger_reason", r["id"]): r for r in cfg["rules"]}
    return [
        SafetyEvent(
            event_id=r.event_id, severity=r.severity, trigger_reason=r.trigger_reason,
            required_action=r.required_action, confidence=float(r.confidence),
            timestamp=r.timestamp, synthetic_flag=bool(r.synthetic_flag),
            recommendation=by_reason.get(r.trigger_reason, {}).get("recommendation", ""),
            session_id=r.session_id, operator_id=r.operator_id, machine_id=r.machine_id,
            rule_id=None,
            evidence={"worker_distance_m": r.worker_distance_m,
                      "closing_speed_mps": r.closing_speed_mps,
                      "machine_state": r.machine_state, "event_type": r.event_type},
        )
        for r in df.itertuples(index=False)
    ]


def validate_against_logged(path: Optional[str] = None, group: str = "worker_proximity") -> dict:
    """
    Re-evaluate each logged event's context with the rule engine and compare
    severities. Only rules relevant to what the log records (the given group
    plus seatbelt) are applied, so unrelated rules (e.g. slope) don't skew it.
    """
    cfg = load_rules()
    df = derive_facts(pd.read_csv(path or os.path.join(DATA_DIR, "safety_events.csv")), cfg)
    rules = [r for r in cfg["rules"] if r.get("group") == group or r["id"] == "seatbelt_off_while_moving"]
    fired = _evaluate_frame(df, cfg, rules)

    by_id = {r["id"]: r for r in rules}
    def top(row):
        hits = [by_id[c] for c in fired.columns if row[c]]
        return max(hits, key=lambda r: _severity_rank(cfg, r["severity"]))["severity"] if hits else "NONE"
    predicted = fired.apply(top, axis=1)

    agree = predicted == df["severity"]
    confusion = pd.crosstab(df["severity"], predicted, rownames=["logged"], colnames=["rules"])
    return {
        "n_events": int(len(df)),
        "agreement": round(float(agree.mean()), 4),
        "confusion": confusion,
        "by_event_type": df.assign(ok=agree).groupby("event_type")["ok"].mean().round(3).to_dict(),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    from shared.schemas import (MachineContext, OperatorContext, SiteContext,
                                TrafficContext, WeatherContext, TemporalContext)
    ctx = SessionContext(
        session_id="S_DEMO", operator_id="OP1002", machine_id="EXC003",
        timestamp="2026-09-23T16:30:00",
        operator=OperatorContext(operator_id="OP1002", fatigue_proxy=0.66),
        machine=MachineContext(machine_id="EXC003", engine_state="swinging", speed=0.0,
                               fuel_level=22.3, load=70.0,
                               maintenance_indicators={"seatbelt_status": "on",
                                                       "engine_health_score": 0.81,
                                                       "hydraulic_health_score": 0.79}),
        site=SiteContext(slope=6.0),
        traffic=TrafficContext(worker_proximity=9.0, closing_speed=1.8, congestion="medium"),
        weather=WeatherContext(rain_mm=0.0, visibility_m=3000, wind_speed_kmh=12, heat_index_c=30),
        temporal=TemporalContext(shift_elapsed_min=630),
    )
    print("--- evaluate_all (demo context) ---")
    for e in evaluate_all(ctx):
        print(f"  {e.severity:8s} {e.rule_id:55s} -> {e.required_action}")
    print(f"  evaluate_safety -> {evaluate_safety(ctx).severity} / {evaluate_safety(ctx).trigger_reason}")

    print("--- scan_telemetry (first breach per session, MEDIUM+) ---")
    events = scan_telemetry(min_severity="MEDIUM")
    print(f"  {len(events)} events;",
          pd.Series([e.severity for e in events]).value_counts().to_dict())
    print("  top rules:", pd.Series([e.rule_id for e in events]).value_counts().head(8).to_dict())

    print("--- validate_against_logged ---")
    v = validate_against_logged()
    print(f"  agreement={v['agreement']} over {v['n_events']} logged events")
    print(v["confusion"].to_string())


if __name__ == "__main__":
    os.chdir(ROOT)
    _smoke_test()
