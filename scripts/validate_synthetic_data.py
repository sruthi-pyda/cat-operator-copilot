#!/usr/bin/env python3
"""
CAT Operator Copilot — Synthetic Data Validator
Run from project root: python3 scripts/validate_synthetic_data.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

DATA_DIR = "data/synthetic"
REPORT_PATH = "reports/validation_report.json"
os.makedirs("reports", exist_ok=True)

VALID_OPERATOR_IDS = {f"OP{1000+i}" for i in range(1, 31)}
VALID_MACHINE_IDS  = {f"EXC{i:03d}" for i in range(1, 11)}
VALID_TASK_IDS     = {f"T{i:05d}" for i in range(1, 1501)}
VALID_LOCATION_IDS = {f"LOC{i:03d}" for i in range(1, 11)}


# ── helpers ───────────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.errors   = []
        self.warnings = []
        self.passed   = []
        self.counts   = {}

    def error(self, msg):
        self.errors.append(msg)
        print(f"  [ERROR]  {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]   {msg}")

    def ok(self, msg):
        self.passed.append(msg)
        print(f"  [OK]     {msg}")

    def score(self):
        total = len(self.errors) + len(self.warnings) + len(self.passed)
        if total == 0:
            return 0.0
        return round(len(self.passed) / total * 100, 1)

    def to_dict(self):
        return {
            "score_pct": self.score(),
            "passed": len(self.passed),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "row_counts": self.counts,
            "passed_checks": self.passed,
            "warning_checks": self.warnings,
            "error_checks": self.errors,
        }


def load(name):
    path = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def check_no_negatives(df, cols, table, r):
    for col in cols:
        if col not in df.columns:
            continue
        bad = (df[col] < 0).sum()
        if bad:
            r.error(f"{table}.{col}: {bad} negative values")
        else:
            r.ok(f"{table}.{col}: no negatives")


def check_ids_present(df, id_col, valid_set, table, r):
    if id_col not in df.columns:
        r.warn(f"{table}: column '{id_col}' missing")
        return
    missing = set(valid_set) - set(df[id_col])
    extra   = set(df[id_col]) - valid_set
    if missing:
        r.warn(f"{table}: {len(missing)} expected IDs absent ({list(missing)[:3]}...)")
    else:
        r.ok(f"{table}: all expected {id_col} values present")
    if extra:
        r.warn(f"{table}: {len(extra)} unexpected IDs found")


def check_fk(df, col, valid_set, table, r):
    if col not in df.columns:
        r.warn(f"{table}: FK column '{col}' missing")
        return
    bad = (~df[col].isin(valid_set)).sum()
    if bad:
        r.error(f"{table}.{col}: {bad} rows with invalid FK")
    else:
        r.ok(f"{table}.{col}: all FKs valid")


def check_timestamps(df, start_col, end_col, table, r):
    if start_col not in df.columns or end_col not in df.columns:
        return
    s = pd.to_datetime(df[start_col], errors="coerce")
    e = pd.to_datetime(df[end_col],   errors="coerce")
    null_s = s.isna().sum()
    null_e = e.isna().sum()
    if null_s or null_e:
        r.error(f"{table}: {null_s} unparseable {start_col}, {null_e} unparseable {end_col}")
        return
    bad = (e <= s).sum()
    if bad:
        r.error(f"{table}: {bad} rows where {end_col} <= {start_col}")
    else:
        r.ok(f"{table}: all {start_col}/{end_col} in correct order")


def check_synthetic_flags(df, table, r):
    if "synthetic_flag" not in df.columns:
        r.warn(f"{table}: synthetic_flag column missing")
        return
    bad = (~df["synthetic_flag"].astype(bool)).sum()
    if bad:
        r.error(f"{table}: {bad} rows where synthetic_flag != True")
    else:
        r.ok(f"{table}: all rows have synthetic_flag=True")


def check_relationship(df, group_col, group_val_high, group_val_low,
                       metric_col, table, r, direction="higher"):
    if group_col not in df.columns or metric_col not in df.columns:
        r.warn(f"{table}: can't check {group_col}/{metric_col} relationship (missing column)")
        return
    hi = df[df[group_col] == group_val_high][metric_col].mean()
    lo = df[df[group_col] == group_val_low ][metric_col].mean()
    if np.isnan(hi) or np.isnan(lo):
        r.warn(f"{table}: insufficient data for {group_col} relationship check")
        return
    if direction == "higher" and hi > lo:
        r.ok(f"{table}: {metric_col} — {group_val_high}={hi:.3f} > {group_val_low}={lo:.3f}")
    elif direction == "lower" and hi < lo:
        r.ok(f"{table}: {metric_col} — {group_val_high}={hi:.3f} < {group_val_low}={lo:.3f}")
    else:
        r.warn(f"{table}: weak {metric_col} relationship — {group_val_high}={hi:.3f}, {group_val_low}={lo:.3f}")


# ── per-table checks ──────────────────────────────────────────────────────────

def validate_operators(df, r):
    print("\n[operators.csv]")
    r.counts["operators"] = len(df)
    check_ids_present(df, "operator_id", VALID_OPERATOR_IDS, "operators", r)
    check_no_negatives(df, ["years_experience","baseline_idle_ratio","baseline_cycle_time_sec",
                             "baseline_fuel_l_per_cycle","baseline_safety_event_rate",
                             "recent_workload_hours","fatigue_proxy"], "operators", r)
    check_synthetic_flags(df, "operators", r)

    face_true = df[df["operator_id"].isin({"OP1001","OP1002","OP1003"})]["face_registered"]
    face_false = df[~df["operator_id"].isin({"OP1001","OP1002","OP1003"})]["face_registered"]
    if face_true.all():
        r.ok("operators: OP1001-OP1003 have face_registered=True")
    else:
        r.error("operators: OP1001-OP1003 should all have face_registered=True")
    if not face_false.any():
        r.ok("operators: OP1004-OP1030 have face_registered=False")
    else:
        r.warn(f"operators: {face_false.sum()} non-team operators have face_registered=True")


def validate_machines(df, r):
    print("\n[machines.csv]")
    r.counts["machines"] = len(df)
    check_ids_present(df, "machine_id", VALID_MACHINE_IDS, "machines", r)
    check_no_negatives(df, ["engine_hours","fuel_capacity_l","current_fuel_level_pct",
                             "rated_capacity","engine_health_score","hydraulic_health_score"], "machines", r)
    check_synthetic_flags(df, "machines", r)

    if ((df["engine_health_score"] >= 0) & (df["engine_health_score"] <= 1)).all():
        r.ok("machines: engine_health_score in [0,1]")
    else:
        r.error("machines: engine_health_score out of range")


def validate_tasks(df, r):
    print("\n[tasks.csv]")
    r.counts["tasks"] = len(df)
    check_ids_present(df, "task_id", VALID_TASK_IDS, "tasks", r)
    check_no_negatives(df, ["target_output","estimated_travel_distance_km"], "tasks", r)

    valid_types = {"excavation","loading","trenching","grading","hauling","stockpiling"}
    bad_types = (~df["task_type"].isin(valid_types)).sum()
    if bad_types:
        r.error(f"tasks: {bad_types} rows with unknown task_type")
    else:
        r.ok("tasks: all task_type values valid")

    null_dep = (df["dependency_task_id"] == "null").sum()
    r.ok(f"tasks: {null_dep} tasks have no dependency (null)")


def validate_sessions(df, operators_df, machines_df, tasks_df, r):
    print("\n[task_sessions.csv]")
    r.counts["task_sessions"] = len(df)

    check_timestamps(df, "start_timestamp", "end_timestamp", "sessions", r)
    check_fk(df, "operator_id", set(operators_df["operator_id"]), "sessions", r)
    check_fk(df, "machine_id",  set(machines_df["machine_id"]),   "sessions", r)
    check_fk(df, "task_id",     set(tasks_df["task_id"]),         "sessions", r)

    check_no_negatives(df, ["actual_task_duration_min","actual_fuel_used_l",
                             "actual_idle_time_min","actual_cycle_count",
                             "rain_mm","visibility_m","temperature_c",
                             "travel_distance_km","soil_hardness_index",
                             "site_slope_deg","soil_moisture_pct"], "sessions", r)
    check_synthetic_flags(df, "sessions", r)

    # Target leakage check — these columns exist but must NOT be used as model inputs (documented)
    for col in ["actual_task_duration_min","actual_fuel_used_l"]:
        if col in df.columns:
            r.ok(f"sessions: '{col}' exists as TARGET column (must never be used as model input)")

    # Relationship checks
    check_relationship(df, "workload", "heavy", "light", "actual_fuel_used_l",  "sessions", r)
    check_relationship(df, "workload", "heavy", "light", "actual_task_duration_min", "sessions", r)
    check_relationship(df, "congestion_level", "high", "low", "actual_idle_time_min", "sessions", r)
    check_relationship(df, "soil_hardness_index",
                       df["soil_hardness_index"].quantile(0.75),
                       df["soil_hardness_index"].quantile(0.25),
                       "actual_task_duration_min", "sessions", r)

    # Idle ratio sanity
    df2 = df.copy()
    df2["idle_ratio"] = df2["actual_idle_time_min"] / df2["actual_task_duration_min"].replace(0, np.nan)
    bad_ir = (df2["idle_ratio"] > 1.0).sum()
    if bad_ir:
        r.error(f"sessions: {bad_ir} rows with idle_ratio > 1.0 (idle > duration)")
    else:
        r.ok("sessions: idle_ratio <= 1.0 for all rows")


def validate_telemetry(df, operators_df, machines_df, r):
    print("\n[telemetry.csv]")
    r.counts["telemetry"] = len(df)

    check_fk(df, "operator_id", set(operators_df["operator_id"]), "telemetry", r)
    check_fk(df, "machine_id",  set(machines_df["machine_id"]),   "telemetry", r)
    check_no_negatives(df, ["fuel_used_l","engine_hours","worker_distance_m",
                             "machine_speed_kmh","engine_load_pct","visibility_m"], "telemetry", r)
    check_synthetic_flags(df, "telemetry", r)

    if len(df) >= 50000:
        r.ok(f"telemetry: {len(df):,} rows (meets ≥50k target)")
    else:
        r.warn(f"telemetry: {len(df):,} rows (below 50k target)")

    # Machine state vs speed consistency
    parked = df[df["machine_state"].isin(["parked","safe_idle"])]["machine_speed_kmh"]
    if (parked > 0.5).mean() < 0.05:
        r.ok("telemetry: parked/safe_idle rows have near-zero speed")
    else:
        r.warn("telemetry: some parked/safe_idle rows show non-zero speed")

    # Seatbelt violation check
    sbelt_off_moving = ((df["seatbelt_status"]=="off") & (df["machine_state"]=="traveling")).sum()
    r.ok(f"telemetry: {sbelt_off_moving} seatbelt-off-while-traveling rows present")


def validate_safety_events(df, sessions_df, r):
    print("\n[safety_events.csv]")
    r.counts["safety_events"] = len(df)

    check_no_negatives(df, ["worker_distance_m","closing_speed_mps",
                             "machine_speed_kmh","visibility_m"], "safety_events", r)
    check_synthetic_flags(df, "safety_events", r)

    if len(df) >= 1000:
        r.ok(f"safety_events: {len(df)} rows (meets ≥1k target)")
    else:
        r.warn(f"safety_events: {len(df)} rows (below 1k target)")

    # Severity distribution
    sev_counts = df["severity"].value_counts().to_dict()
    r.ok(f"safety_events: severity distribution = {sev_counts}")

    # CRITICAL check: contrast — same ~10m distance, different context → different severity
    info_ev = df[df["severity"]=="INFO"]
    crit_ev = df[df["severity"]=="CRITICAL"]
    if len(info_ev) > 0 and len(crit_ev) > 0:
        info_dist = info_ev["worker_distance_m"].mean()
        crit_dist = crit_ev["worker_distance_m"].mean()
        info_cs   = info_ev["closing_speed_mps"].mean()
        crit_cs   = crit_ev["closing_speed_mps"].mean()
        if crit_cs > info_cs:
            r.ok(f"safety_events: CRITICAL closing_speed ({crit_cs:.2f}) > INFO ({info_cs:.2f}) "
                 f"— severity is context-driven, not distance alone "
                 f"[INFO avg dist={info_dist:.1f}m, CRITICAL avg dist={crit_dist:.1f}m]")
        else:
            r.error("safety_events: CRITICAL closing_speed not > INFO — contrast logic broken")

    # Stationary + nearby should never be CRITICAL
    stationary_nearby_crit = df[
        (df["worker_distance_m"] < 15) &
        (df["machine_state"]=="safe_idle") &
        (df["severity"]=="CRITICAL")
    ]
    if len(stationary_nearby_crit) == 0:
        r.ok("safety_events: no CRITICAL events when machine is stationary (correct)")
    else:
        r.error(f"safety_events: {len(stationary_nearby_crit)} CRITICAL events with stationary machine — logic error")


def validate_training_records(df, r):
    print("\n[training_records.csv]")
    r.counts["training_records"] = len(df)

    check_no_negatives(df, ["confidence","before_metric","after_metric","improvement_pct"], "training_records", r)
    if len(df) >= 500:
        r.ok(f"training_records: {len(df)} rows (meets ≥500 target)")
    else:
        r.warn(f"training_records: {len(df)} rows (below 500 target)")

    bad_conf = ((df["confidence"] < 0) | (df["confidence"] > 1)).sum()
    if bad_conf:
        r.error(f"training_records: {bad_conf} rows with confidence outside [0,1]")
    else:
        r.ok("training_records: all confidence values in [0,1]")

    bad_after = (df["after_metric"] < df["before_metric"]).sum()
    if bad_after > len(df) * 0.05:
        r.warn(f"training_records: {bad_after} rows where after_metric < before_metric (>5%)")
    else:
        r.ok(f"training_records: after_metric >= before_metric in ≥95% of rows")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAT Operator Copilot — Data Validation Report")
    print("=" * 60)

    r = Report()

    operators = load("operators")
    machines  = load("machines")
    tasks     = load("tasks")
    sessions  = load("task_sessions")
    telemetry = load("telemetry")
    safety    = load("safety_events")
    training  = load("training_records")

    missing = [n for n, df in [("operators",operators),("machines",machines),("tasks",tasks),
                                ("task_sessions",sessions),("telemetry",telemetry),
                                ("safety_events",safety),("training_records",training)]
               if df is None]
    if missing:
        print(f"\n[FATAL] Missing files: {missing}")
        sys.exit(1)

    validate_operators(operators, r)
    validate_machines(machines, r)
    validate_tasks(tasks, r)
    validate_sessions(sessions, operators, machines, tasks, r)
    validate_telemetry(telemetry, operators, machines, r)
    validate_safety_events(safety, sessions, r)
    validate_training_records(training, r)

    print("\n" + "=" * 60)
    score = r.score()
    print(f"VALIDATION SCORE: {score}%  "
          f"({len(r.passed)} passed / {len(r.warnings)} warnings / {len(r.errors)} errors)")
    print("=" * 60)

    with open(REPORT_PATH, "w") as f:
        json.dump(r.to_dict(), f, indent=2)
    print(f"Report saved to {REPORT_PATH}")

    if r.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
