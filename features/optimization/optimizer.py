"""
Feature 05 — Optimal Task Sequencing
Owner: Aneesha (Team Member 2)

Greedy, multi-objective task optimizer (see docs/DECISIONS.md D005).

Objective per (operator, machine, task) candidate — weighted sum, lower is better:
    time_cost             ETA P50 (Sruthi's prediction API)
    fuel_cost             Fuel P50 (Sruthi's prediction API)
    deadline_risk         P50/P90 finish vs. deadline (maximises deadlines met)
    transition_cost       travel + attachment swap + blocked-route detour
    safety_condition_risk Safety Guardian findings for the planned context
Ranking score = total_cost / priority_factor, so priority-1 work wins ties.

Hard constraints (candidate is excluded, with the reason recorded):
    task open (pending/active), dependency complete, machine type matches,
    operator authorised + skilled + certified, no CRITICAL safety finding.
Safety is never traded off against cost: CRITICAL findings block outright.

Public functions:
  rank_assignments(...)                 → List[dict]  (who does what next, ranked)
  generate_plan(tasks, session_context) → dict        (API contract: sequence for one operator/machine)
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.schemas import (SessionContext, OperatorContext, MachineContext, TaskContext,
                            SiteContext, WeatherContext, TemporalContext, DataQuality)
from features.prediction.api import predict_combined
from features.safety.safety_guardian import evaluate_records, load_rules

DATA_DIR      = os.path.join(ROOT, "data", "synthetic")
SETTINGS_PATH = os.path.join(ROOT, "config", "settings.yaml")

SKILL_RANK   = {"novice": 0, "intermediate": 1, "advanced": 2, "expert": 3}
OPEN_STATUS  = {"pending", "active"}


# ── Config + data ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_settings() -> dict:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["optimization"]


@dataclass
class SiteData:
    operators: pd.DataFrame
    machines:  pd.DataFrame
    tasks:     pd.DataFrame
    sites:     pd.DataFrame
    weather:   pd.DataFrame

    @classmethod
    def load(cls, data_dir: str = DATA_DIR) -> "SiteData":
        rd = lambda n: pd.read_csv(os.path.join(data_dir, n))
        tasks = rd("tasks.csv")
        weather = rd("weather.csv")
        weather["ts"] = pd.to_datetime(weather["timestamp"])
        return cls(rd("operators.csv"), rd("machines.csv"), tasks,
                   rd("site_conditions.csv").set_index("location_id"), weather.sort_values("ts"))

    def __post_init__(self):
        self._weather_by_loc = {loc: g for loc, g in self.weather.groupby("location_id")}

    def weather_at(self, location_id: str, when: datetime) -> Dict[str, Any]:
        """Latest weather observation at the location at or before `when`."""
        w = self._weather_by_loc.get(location_id, self.weather.iloc[0:0])
        prior = w[w["ts"] <= when]
        row = (prior.iloc[-1] if not prior.empty else w.iloc[0]) if not w.empty else None
        return {} if row is None else row.to_dict()


@dataclass
class Candidate:
    operator_id: str
    machine_id: str
    task_id: str
    start: datetime
    end_p50: datetime
    eta: Tuple[float, float, float]
    fuel: Tuple[float, float, float]
    prediction_confidence: float
    breakdown: Dict[str, float]
    total_cost: float
    score: float
    deadline_met: bool
    deadline_at_risk: bool
    safety_flags: List[str]
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "operator_id": self.operator_id, "machine_id": self.machine_id, "task_id": self.task_id,
            "start": self.start.isoformat(timespec="minutes"),
            "end_p50": self.end_p50.isoformat(timespec="minutes"),
            "eta_min": {"p10": self.eta[0], "p50": self.eta[1], "p90": self.eta[2]},
            "fuel_l":  {"p10": self.fuel[0], "p50": self.fuel[1], "p90": self.fuel[2]},
            "prediction_confidence": self.prediction_confidence,
            "cost_breakdown": self.breakdown,
            "total_cost": self.total_cost, "score": self.score,
            "deadline_met": self.deadline_met, "deadline_at_risk": self.deadline_at_risk,
            "safety_flags": self.safety_flags,
            "reason": "; ".join(self.reasoning),
            "synthetic_flag": True,
        }


# ── Feasibility (hard constraints) ────────────────────────────────────────────

def _infeasible_reason(op: pd.Series, mc: pd.Series, task: pd.Series,
                       completed: set, when: datetime) -> Optional[Tuple[str, str]]:
    """(code, detail) for the first violated hard constraint, else None."""
    if task["status"] not in OPEN_STATUS:
        return "task_not_open", f"task status is {task['status']}"
    dep = task["dependency_task_id"]
    if isinstance(dep, str) and dep not in ("", "null") and dep not in completed:
        return "dependency_pending", f"waiting on dependency {dep}"
    if mc["machine_type"] != task["required_machine_type"]:
        return "machine_type_mismatch", f"needs {task['required_machine_type']}, machine is {mc['machine_type']}"
    if mc["machine_type"] not in str(op["authorized_machine_types"]).split(","):
        return "operator_not_authorised", f"operator not authorised for {mc['machine_type']}"
    if SKILL_RANK.get(op["machine_skill_level"], -1) < SKILL_RANK.get(task["required_skill_level"], 99):
        return "skill_below_required", f"requires {task['required_skill_level']} skill, operator is {op['machine_skill_level']}"
    if op["certification_status"] != "active" or pd.to_datetime(op["certification_expiry"]) < when:
        return "certification_inactive", "operator certification not active"
    return None


# ── Safety (delegated to the deterministic Safety Guardian) ───────────────────

def _safety_record(op: pd.Series, mc: pd.Series, site: pd.Series,
                   wx: Dict[str, Any], shift_elapsed_min: float) -> Dict[str, Any]:
    """
    The planned working context, in Safety Guardian field names.
    The machine is assumed active (it will be working the task); worker
    proximity is unknown at planning time and therefore not evaluated.
    """
    return {
        "machine_state": "digging",  # any active state; enables slope/congestion rules
        "site_slope_deg": site.get("slope_deg"),
        "congestion_level": site.get("congestion_level"),
        "engine_health_score": mc.get("engine_health_score"),
        "hydraulic_health_score": mc.get("hydraulic_health_score"),
        "maintenance_status": mc.get("maintenance_status"),
        "fuel_level_pct": mc.get("current_fuel_level_pct"),
        "fatigue_proxy": op.get("fatigue_proxy"),
        "shift_elapsed_min": shift_elapsed_min,
        **{k: wx.get(k) for k in ("rain_mm", "visibility_m", "wind_speed_kmh", "heat_index_c", "dust_level")},
    }


# ── Prediction (Sruthi's API) ─────────────────────────────────────────────────

def _build_context(op: pd.Series, mc: pd.Series, task: pd.Series, site: pd.Series,
                   wx: Dict[str, Any], when: datetime, shift_elapsed_min: float) -> SessionContext:
    return SessionContext(
        session_id=f"PLAN-{op['operator_id']}-{mc['machine_id']}-{task['task_id']}",
        operator_id=op["operator_id"], machine_id=mc["machine_id"], task_id=task["task_id"],
        timestamp=when.isoformat(timespec="seconds"),
        operator=OperatorContext(operator_id=op["operator_id"], experience=float(op["years_experience"]),
                                 machine_skill=op["machine_skill_level"], task_skill=op["task_skill_level"],
                                 fatigue_proxy=float(op["fatigue_proxy"])),
        machine=MachineContext(machine_id=mc["machine_id"], machine_type=mc["machine_type"],
                               machine_condition=mc["machine_condition"], engine_hours=float(mc["engine_hours"]),
                               attachment=mc["attachment_type"], fuel_level=float(mc["current_fuel_level_pct"])),
        task=TaskContext(task_id=task["task_id"], task_type=task["task_type"], task_phase=task["task_phase"],
                         workload=task["workload"], target_output=float(task["target_output"]),
                         priority=int(task["priority"]), deadline=task["deadline_timestamp"],
                         location=task["location_id"]),
        site=SiteContext(soil_material=site.get("soil_material"), moisture=site.get("soil_moisture_pct"),
                         hardness=site.get("soil_hardness_index"), slope=site.get("slope_deg"),
                         surface=site.get("surface_type"), route=site.get("route_condition"),
                         work_zone_constraints=site.get("work_zone_constraint"),
                         haul_distance=float(task["estimated_travel_distance_km"]),
                         congestion=site.get("congestion_level")),
        weather=WeatherContext(**{k: wx.get(k) for k in ("rain_mm", "temperature_c", "heat_index_c",
                                                         "wind_speed_kmh", "visibility_m", "dust_level", "day_night")}),
        temporal=TemporalContext(timestamp=when.isoformat(timespec="seconds"),
                                 shift_elapsed_min=shift_elapsed_min,
                                 recent_idle_ratio=float(op["baseline_idle_ratio"])),
        data_quality=DataQuality(timestamp=when.isoformat(timespec="seconds")),
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

class _Scorer:
    """Scores candidates; caches predictions and safety checks within one run."""

    def __init__(self, data: SiteData):
        self.data = data
        self.cfg = load_settings()
        self._pred_cache: Dict[tuple, Any] = {}
        self._safety_cache: Dict[tuple, List[Tuple[str, str]]] = {}

    def predict(self, op, mc, task, site, wx, when, shift_min):
        # Prediction features depend on the task/site/weather, machine condition and
        # operator experience/idle baseline — not on the specific IDs.
        key = (task["task_id"], mc["machine_condition"], op["years_experience"],
               op["baseline_idle_ratio"], wx.get("timestamp"))
        if key not in self._pred_cache:
            self._pred_cache[key] = predict_combined(
                _build_context(op, mc, task, site, wx, when, shift_min))
        return self._pred_cache[key]

    def _safety_key(self, op, mc, task, when, shift_min):
        wx = self.data.weather_at(task["location_id"], when)
        return (op["operator_id"], mc["machine_id"], task["location_id"], wx.get("timestamp"), shift_min), wx

    def prefetch_safety(self, items: List[tuple]) -> None:
        """Batch-evaluate safety for (op, mc, task, when, shift_min) items in one Safety Guardian pass."""
        keys, records = [], []
        for op, mc, task, when, shift_min in items:
            key, wx = self._safety_key(op, mc, task, when, shift_min)
            if key not in self._safety_cache and key not in keys:
                keys.append(key)
                records.append(_safety_record(op, mc, self.data.sites.loc[task["location_id"]], wx, shift_min))
        for key, events in zip(keys, evaluate_records(records)):
            self._safety_cache[key] = [(e.severity, e.rule_id) for e in events]

    def safety(self, op, mc, task, when, shift_min):
        key, _ = self._safety_key(op, mc, task, when, shift_min)
        if key not in self._safety_cache:
            self.prefetch_safety([(op, mc, task, when, shift_min)])
        return self._safety_cache[key]

    def score(self, op: pd.Series, mc: pd.Series, task: pd.Series, when: datetime,
              shift_elapsed_min: float = 0.0, prev_location: Optional[str] = None,
              current_attachment: Optional[str] = None) -> Tuple[Optional[Candidate], Optional[Tuple[str, str]]]:
        w = self.cfg["weights"]
        site = self.data.sites.loc[task["location_id"]]
        wx = self.data.weather_at(task["location_id"], when)

        findings = self.safety(op, mc, task, when, shift_elapsed_min)
        critical = [r for s, r in findings if s == "CRITICAL"]
        if critical:
            return None, ("safety_block", f"CRITICAL safety: {', '.join(critical)}")

        pred = self.predict(op, mc, task, site, wx, when, shift_elapsed_min)
        eta = (pred.eta_p10, pred.eta_p50, pred.eta_p90)
        fuel = (pred.fuel_p10, pred.fuel_p50, pred.fuel_p90)
        reasons = [f"ETA {eta[1]:.0f} min (P90 {eta[2]:.0f}), fuel {fuel[1]:.1f} L"]

        # Transition: travel (skipped if already at this location), attachment swap, route detour
        travel_km = 0.0 if prev_location == task["location_id"] else float(task["estimated_travel_distance_km"])
        attachment = current_attachment or mc["attachment_type"]
        swap = attachment != task["required_attachment"]
        blocked = str(site.get("route_condition")) in ("blocked", "partially_blocked")
        transition = travel_km * w["transition_per_km"] + (w["attachment_swap"] if swap else 0.0) \
                     + (w["blocked_route"] if blocked else 0.0)
        if travel_km == 0 and prev_location:
            reasons.append("same location as previous task")
        if swap:
            reasons.append(f"attachment swap {attachment}->{task['required_attachment']}")
        if blocked:
            reasons.append(f"route {site['route_condition']}")

        # Deadline: P50 finish drives miss, P90 finish drives at-risk
        transit_min = travel_km * w["transition_per_km"] + (w["attachment_swap"] if swap else 0.0)
        start = when + timedelta(minutes=transit_min)
        end_p50 = start + timedelta(minutes=eta[1])
        end_p90 = start + timedelta(minutes=eta[2])
        deadline = pd.to_datetime(task["deadline_timestamp"]).to_pydatetime()
        met, at_risk = end_p50 <= deadline, end_p50 <= deadline < end_p90
        if not met:
            late_h = (end_p50 - deadline).total_seconds() / 3600
            deadline_risk = w["deadline_miss"] + late_h * w["deadline_late_per_hour"]
            reasons.append(f"misses deadline by {late_h:.1f} h")
        elif at_risk:
            deadline_risk = w["deadline_at_risk"]
            reasons.append("deadline at risk at P90")
        else:
            deadline_risk = 0.0
            reasons.append(f"meets deadline with {(deadline - end_p90).total_seconds() / 3600:.1f} h margin")

        sev_cost = self.cfg["safety_severity_cost"]
        safety_risk = float(sum(sev_cost.get(s, 0) for s, _ in findings))
        flags = [f"{s}:{r}" for s, r in findings if s in ("HIGH", "MEDIUM")]
        if flags:
            reasons.append("safety: " + ", ".join(r for s, r in findings if s in ("HIGH", "MEDIUM")))

        breakdown = {
            "time_cost":             round(eta[1] * w["time_per_min"], 2),
            "fuel_cost":             round(fuel[1] * w["fuel_per_l"], 2),
            "deadline_risk":         round(deadline_risk, 2),
            "transition_cost":       round(transition, 2),
            "safety_condition_risk": round(safety_risk, 2),
        }
        total = round(sum(breakdown.values()), 2)
        factor = float(self.cfg["priority_factor"][int(task["priority"])])
        reasons.insert(0, f"priority {int(task['priority'])}")
        return Candidate(op["operator_id"], mc["machine_id"], task["task_id"], start, end_p50,
                         eta, fuel, pred.confidence, breakdown, total, round(total / factor, 2),
                         met, at_risk, flags, reasons), None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_pool(tasks: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    open_tasks = tasks[tasks["status"].isin(OPEN_STATUS)]
    return open_tasks.sort_values(["priority", "deadline_timestamp"]).head(pool_size)


def _completed(tasks: pd.DataFrame) -> set:
    return set(tasks.loc[tasks["status"] == "completed", "task_id"])


def _plan_start(now) -> datetime:
    return pd.to_datetime(now or load_settings()["plan_start"]).to_pydatetime()


# ── Public API ────────────────────────────────────────────────────────────────

def rank_assignments(operators: Optional[pd.DataFrame] = None,
                     machines: Optional[pd.DataFrame] = None,
                     tasks: Optional[pd.DataFrame] = None,
                     now: Optional[str] = None,
                     pool_size: Optional[int] = None,
                     data: Optional[SiteData] = None) -> Dict[str, Any]:
    """
    Rank next-task assignments across the available fleet.

    Scores every feasible (operator, machine, task) triple, then assigns greedily
    by score so each operator, machine and task is used at most once.
    Returns {"assignments": [...ranked...], "excluded": {reason: count}, ...}.
    """
    data = data or SiteData.load()
    operators = data.operators if operators is None else operators
    machines  = data.machines  if machines  is None else machines
    all_tasks = data.tasks     if tasks     is None else tasks
    when = _plan_start(now)
    pool = _open_pool(all_tasks, pool_size or load_settings()["candidate_pool_size"])
    completed = _completed(data.tasks)

    scorer = _Scorer(data)
    excluded: Dict[str, int] = {}
    feasible = []
    for _, task in pool.iterrows():
        for _, mc in machines.iterrows():
            for _, op in operators.iterrows():
                why = _infeasible_reason(op, mc, task, completed, when)
                if why is None:
                    feasible.append((op, mc, task))
                else:
                    excluded[why[0]] = excluded.get(why[0], 0) + 1

    scorer.prefetch_safety([(op, mc, task, when, 0.0) for op, mc, task in feasible])
    candidates: List[Candidate] = []
    for op, mc, task in feasible:
        cand, why = scorer.score(op, mc, task, when)
        if cand:
            candidates.append(cand)
        else:
            excluded[why[0]] = excluded.get(why[0], 0) + 1

    candidates.sort(key=lambda c: (c.score, c.total_cost))
    used_op, used_mc, used_task, ranked = set(), set(), set(), []
    for c in candidates:
        if c.operator_id in used_op or c.machine_id in used_mc or c.task_id in used_task:
            continue
        used_op.add(c.operator_id); used_mc.add(c.machine_id); used_task.add(c.task_id)
        d = c.to_dict()
        d["rank"] = len(ranked) + 1
        ranked.append(d)

    unassigned = [t for t in pool["task_id"] if t not in used_task]
    return {
        "plan_start": when.isoformat(timespec="minutes"),
        "assignments": ranked,
        "candidates_scored": len(candidates),
        "excluded": dict(sorted(excluded.items(), key=lambda kv: -kv[1])),
        "unassigned_tasks": unassigned,
        "synthetic_flag": True,
    }


def generate_plan(tasks: Optional[List[str]], session_context: SessionContext,
                  now: Optional[str] = None, data: Optional[SiteData] = None) -> Dict[str, Any]:
    """
    API contract: sequence tasks for the operator + machine in session_context.

    Greedy: at each step pick the feasible task with the lowest score given the
    current location, attachment and clock, until the shift budget (fatigue
    high_hours from safety_rules.yaml) would be exceeded. Dependencies may be
    satisfied by tasks scheduled earlier in the same plan.
    """
    data = data or SiteData.load()
    op = data.operators.set_index("operator_id").loc[session_context.operator_id]
    op["operator_id"] = session_context.operator_id
    mc = data.machines.set_index("machine_id").loc[session_context.machine_id]
    mc["machine_id"] = session_context.machine_id

    if tasks:
        pool = data.tasks[data.tasks["task_id"].isin(tasks)]
    else:
        pool = _open_pool(data.tasks, load_settings()["candidate_pool_size"] * 4)
        pool = pool[pool["required_machine_type"] == mc["machine_type"]]

    shift_start_min = 0.0
    if session_context.temporal and session_context.temporal.shift_elapsed_min:
        shift_start_min = float(session_context.temporal.shift_elapsed_min)
    budget_min = load_rules()["fatigue"]["high_hours"] * 60

    clock = _plan_start(now or session_context.timestamp)
    shift_min = shift_start_min
    location = session_context.task.location if session_context.task else None
    attachment = mc["attachment_type"]
    completed = _completed(data.tasks)
    scorer = _Scorer(data)

    steps, excluded, remaining = [], {}, pool.copy()
    while not remaining.empty:
        best, best_task = None, None
        feasible = []
        for _, task in remaining.iterrows():
            why = _infeasible_reason(op, mc, task, completed, clock)
            if why is None:
                feasible.append(task)
            else:
                excluded[task["task_id"]] = why[1]
        scorer.prefetch_safety([(op, mc, task, clock, shift_min) for task in feasible])
        for task in feasible:
            cand, why = scorer.score(op, mc, task, clock, shift_min, location, attachment)
            if cand is None:
                excluded[task["task_id"]] = why[1]
            elif best is None or (cand.score, cand.total_cost) < (best.score, best.total_cost):
                best, best_task = cand, task
        if best is None:
            break
        duration = (best.end_p50 - clock).total_seconds() / 60
        if shift_min + duration > budget_min:
            excluded[best.task_id] = f"exceeds {budget_min / 60:.0f} h shift budget (fatigue rule)"
            remaining = remaining[remaining["task_id"] != best.task_id]
            continue
        step = best.to_dict()
        step["sequence"] = len(steps) + 1
        steps.append(step)
        excluded.pop(best.task_id, None)
        completed.add(best.task_id)
        clock, shift_min = best.end_p50, shift_min + duration
        location, attachment = best_task["location_id"], best_task["required_attachment"]
        remaining = remaining[remaining["task_id"] != best.task_id]

    totals = {k: round(sum(s["cost_breakdown"][k] for s in steps), 2)
              for k in ("time_cost", "fuel_cost", "deadline_risk", "transition_cost", "safety_condition_risk")}
    met = sum(s["deadline_met"] for s in steps)
    return {
        "session_id": session_context.session_id,
        "operator_id": session_context.operator_id,
        "machine_id": session_context.machine_id,
        "optimized_sequence": [s["task_id"] for s in steps],
        "total_cost": round(sum(totals.values()), 2),
        "cost_breakdown": totals,
        "deadlines_met": f"{met}/{len(steps)}",
        "shift_minutes_planned": round(shift_min - shift_start_min, 1),
        "steps": steps,
        "excluded": excluded,
        "reason": ("Greedy lowest priority-weighted cost per step (time + fuel + deadline + transition "
                   "+ safety); CRITICAL safety findings and hard constraints excluded; stops at "
                   f"{budget_min / 60:.0f} h fatigue budget"),
        "synthetic_flag": True,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    import time, warnings
    warnings.filterwarnings("ignore")
    t0 = time.time()
    data = SiteData.load()

    result = rank_assignments(data=data)
    print(f"--- rank_assignments ({time.time() - t0:.1f}s, {result['candidates_scored']} candidates) ---")
    for a in result["assignments"][:8]:
        print(f"  #{a['rank']} {a['operator_id']} + {a['machine_id']} -> {a['task_id']}  "
              f"score={a['score']:.1f}  {a['reason']}")
    print("  excluded:", result["excluded"])

    t1 = time.time()
    first = result["assignments"][0]
    ctx = SessionContext(session_id="S_PLAN", operator_id=first["operator_id"], machine_id=first["machine_id"])
    plan = generate_plan(None, ctx, data=data)
    print(f"--- generate_plan {first['operator_id']}/{first['machine_id']} ({time.time() - t1:.1f}s) ---")
    print(f"  sequence={plan['optimized_sequence']}  deadlines_met={plan['deadlines_met']}  "
          f"total_cost={plan['total_cost']}  minutes={plan['shift_minutes_planned']}")
    print("  breakdown:", plan["cost_breakdown"])
    for s in plan["steps"][:4]:
        print(f"   {s['sequence']}. {s['task_id']}  {s['reason']}")


if __name__ == "__main__":
    os.chdir(ROOT)  # prediction API uses repo-relative paths
    _smoke_test()
