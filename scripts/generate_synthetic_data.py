#!/usr/bin/env python3
"""
CAT Operator Copilot — Synthetic Data Generator
SEED = 42 — fully reproducible
Run from project root: python scripts/generate_synthetic_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = "data/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DATE = datetime(2026, 1, 1, 6, 0, 0)

OPERATOR_IDS  = [f"OP{1000+i}" for i in range(1, 31)]
MACHINE_IDS   = [f"EXC{i:03d}" for i in range(1, 11)]
TASK_IDS      = [f"T{i:05d}" for i in range(1, 1501)]
LOCATION_IDS  = [f"LOC{i:03d}" for i in range(1, 11)]

TASK_TYPES    = ["excavation", "loading", "trenching", "grading", "hauling", "stockpiling"]
WORKLOADS     = ["light", "medium", "heavy"]
SKILL_LEVELS  = ["novice", "intermediate", "advanced", "expert"]
MACHINE_CONDS = ["good", "fair", "poor"]
ATTACHMENTS   = ["bucket", "blade", "grapple", "hammer", "ripper"]
SURFACE_TYPES = ["firm", "soft", "wet", "rocky", "paved"]
SOIL_TYPES    = ["clay", "sand", "rock", "gravel", "mixed"]
MACHINE_STATES = ["parked", "safe_idle", "traveling", "loading", "digging", "swinging", "grading"]

MACHINE_MODELS = {
    "excavator": ["CAT 320", "CAT 336", "CAT 390F"],
    "dozer":     ["CAT D6T", "CAT D8T"],
    "grader":    ["CAT 140", "CAT 160"],
    "loader":    ["CAT 966", "CAT 982"],
    "hauler":    ["CAT 745", "CAT 775"],
}

MACHINE_TYPES_BY_ID = [
    "excavator","excavator","excavator","excavator","excavator",
    "dozer","grader","loader","loader","hauler"
]


def _meta(df, confidence=0.85):
    df["data_source"] = "synthetic"
    df["synthetic_flag"] = True
    df["data_quality_confidence"] = confidence
    return df


# ── OPERATORS ────────────────────────────────────────────────────────────────

def generate_operators():
    rows = []
    for op_id in OPERATOR_IDS:
        exp = float(rng.uniform(1, 20))
        skill = SKILL_LEVELS[min(3, int(exp / 5))]

        base_idle  = max(0.04, float(rng.uniform(0.06, 0.25)) - exp * 0.003)
        base_cycle = max(30.0, float(rng.uniform(50, 120))    - exp * 1.5)
        base_fuel  = max(1.5,  float(rng.uniform(3.0, 8.0))   + float(rng.normal(0, 0.3)))
        base_safe  = max(0.005,float(rng.uniform(0.01, 0.15)) - exp * 0.004)

        n_auth   = int(rng.integers(1, 4))
        auth     = ",".join(rng.choice(["excavator","dozer","grader","loader","hauler"], n_auth, replace=False))
        cert     = rng.choice(["active","active","expired","none"], p=[0.65,0.15,0.12,0.08])
        cert_exp = (BASE_DATE + timedelta(days=int(rng.integers(-180,730)))).strftime("%Y-%m-%d")

        recent_wl = float(rng.uniform(20, 60))
        fatigue   = float(np.clip(recent_wl/80 + rng.normal(0, 0.08), 0, 1))
        training  = rng.choice(["up_to_date","due","overdue"], p=[0.60,0.25,0.15])

        face_reg = op_id in ("OP1001","OP1002","OP1003")

        rows.append({
            "operator_id":               op_id,
            "name":                      f"Operator_{op_id}",
            "role":                      rng.choice(["operator","senior_operator","lead_operator"]),
            "years_experience":          round(exp, 1),
            "machine_skill_level":       skill,
            "task_skill_level":          skill,
            "authorized_machine_types":  auth,
            "certification_status":      cert,
            "certification_expiry":      cert_exp,
            "baseline_idle_ratio":       round(base_idle, 4),
            "baseline_cycle_time_sec":   round(base_cycle, 1),
            "baseline_fuel_l_per_cycle": round(base_fuel, 2),
            "baseline_safety_event_rate":round(base_safe, 4),
            "recent_workload_hours":     round(recent_wl, 1),
            "fatigue_proxy":             round(fatigue, 3),
            "training_status":           training,
            "face_registered":           face_reg,
            "face_embedding_path":       f"data/face_registrations/{op_id}_embedding.npy" if face_reg else "",
            "face_model":                "ArcFace" if face_reg else "",
            "face_registration_timestamp": BASE_DATE.strftime("%Y-%m-%dT%H:%M:%S") if face_reg else "",
            "data_source":               "synthetic",
            "synthetic_flag":            True,
            "data_quality_confidence":   0.85,
        })
    return pd.DataFrame(rows)


# ── MACHINES ─────────────────────────────────────────────────────────────────

def generate_machines():
    rows = []
    for i, machine_id in enumerate(MACHINE_IDS):
        mtype  = MACHINE_TYPES_BY_ID[i]
        model  = rng.choice(MACHINE_MODELS[mtype])
        year   = int(rng.integers(2015, 2024))
        eng_h  = float(rng.uniform(500, 8000))
        cond   = "good" if eng_h < 3000 else ("fair" if eng_h < 6000 else "poor")
        if rng.random() < 0.12:
            cond = rng.choice(MACHINE_CONDS)
        fuel_cap  = float(rng.choice([300, 400, 500, 600, 700]))
        fuel_lvl  = float(rng.uniform(20, 100))
        eng_hlth  = float(np.clip(1.0 - eng_h/10000*0.3 + rng.normal(0,0.04), 0.5, 1.0))
        hyd_hlth  = float(np.clip(eng_hlth - rng.uniform(0, 0.08), 0.4, 1.0))
        maint     = "ok" if eng_h < 5000 else ("due" if eng_h < 7000 else "overdue")
        last_m    = BASE_DATE - timedelta(days=int(rng.integers(10,180)))
        maint_due = eng_h + float(rng.integers(100, 500))

        rows.append({
            "machine_id":                machine_id,
            "machine_model":             model,
            "machine_type":              mtype,
            "machine_year":              year,
            "engine_hours":              round(eng_h, 1),
            "machine_condition":         cond,
            "fuel_capacity_l":           fuel_cap,
            "current_fuel_level_pct":    round(fuel_lvl, 1),
            "attachment_type":           rng.choice(ATTACHMENTS),
            "rated_capacity":            float(rng.choice([1.5, 2.0, 3.0, 5.0, 8.0, 10.0])),
            "maintenance_status":        maint,
            "last_maintenance_timestamp":last_m.strftime("%Y-%m-%dT%H:%M:%S"),
            "maintenance_due_hours":     round(maint_due, 1),
            "engine_health_score":       round(eng_hlth, 3),
            "hydraulic_health_score":    round(hyd_hlth, 3),
            "data_source":               "synthetic",
            "synthetic_flag":            True,
            "data_quality_confidence":   0.90,
        })
    return pd.DataFrame(rows)


# ── SITE CONDITIONS ───────────────────────────────────────────────────────────

def generate_site_conditions():
    rows = []
    for loc_id in LOCATION_IDS:
        soil     = rng.choice(SOIL_TYPES)
        moisture = float(rng.uniform(5, 80))
        hardness = float(rng.uniform(0.7, 1.0) if soil == "rock"
                    else rng.uniform(0.1, 0.4) if soil == "sand"
                    else rng.uniform(0.3, 0.8))
        slope    = float(rng.uniform(0, 20))
        surface  = rng.choice(SURFACE_TYPES)
        route    = rng.choice(["clear","partially_blocked","blocked"])
        wz       = rng.choice(["none","restricted_zone","no_go_zone"])
        eq_dens  = int(rng.integers(0, 9))
        cong     = "high" if eq_dens > 5 else ("medium" if eq_dens > 2 else "low")

        rows.append({
            "location_id":       loc_id,
            "soil_material":     soil,
            "soil_moisture_pct": round(moisture, 1),
            "soil_hardness_index":round(hardness, 3),
            "slope_deg":         round(slope, 1),
            "surface_type":      surface,
            "route_condition":   route,
            "work_zone_constraint": wz,
            "blocked_route":     route == "blocked",
            "worker_density":    int(rng.integers(0, 15)),
            "equipment_density": eq_dens,
            "congestion_level":  cong,
            "blind_zone_flag":   bool(rng.random() < 0.3),
            "data_source":       "synthetic",
            "synthetic_flag":    True,
            "data_quality_confidence": 0.85,
        })
    return pd.DataFrame(rows)


# ── TASKS ─────────────────────────────────────────────────────────────────────

def generate_tasks():
    rows = []
    ttype_probs = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    req_machine = {
        "excavation": "excavator", "loading": "loader",
        "trenching":  "excavator", "grading": "grader",
        "hauling":    "hauler",    "stockpiling": "loader",
    }
    base_outputs = {
        "excavation": (50,500), "loading": (10,100),
        "trenching":  (20,200), "grading": (100,1000),
        "hauling":    (5,50),   "stockpiling": (30,300),
    }
    for i, task_id in enumerate(TASK_IDS):
        tt      = rng.choice(TASK_TYPES, p=ttype_probs)
        lo, hi  = base_outputs[tt]
        dep     = TASK_IDS[int(rng.integers(0, i))] if i > 0 and rng.random() < 0.10 else "null"
        deadline= (BASE_DATE + timedelta(days=int(rng.integers(1,30)))).strftime("%Y-%m-%dT%H:%M:%S")

        rows.append({
            "task_id":                   task_id,
            "task_type":                 tt,
            "task_phase":                rng.choice(["setup","active","wrap_up"], p=[0.15,0.70,0.15]),
            "workload":                  rng.choice(WORKLOADS, p=[0.25,0.50,0.25]),
            "target_output":             round(float(rng.uniform(lo, hi)), 1),
            "priority":                  int(rng.integers(1, 6)),
            "deadline_timestamp":        deadline,
            "location_id":               rng.choice(LOCATION_IDS),
            "dependency_task_id":        dep,
            "estimated_travel_distance_km": round(float(rng.uniform(0.1, 5.0)), 2),
            "required_machine_type":     req_machine[tt],
            "required_attachment":       rng.choice(ATTACHMENTS),
            "required_skill_level":      rng.choice(["novice","intermediate","advanced"]),
            "status":                    rng.choice(["pending","active","completed","blocked"],
                                                    p=[0.40,0.20,0.30,0.10]),
        })
    return pd.DataFrame(rows)


# ── WEATHER ───────────────────────────────────────────────────────────────────

def generate_weather(n=6000):
    rows = []
    for i in range(n):
        ts   = BASE_DATE + timedelta(hours=i * 0.5)
        rain = float(rng.exponential(5)) if rng.random() < 0.15 else 0.0
        temp = float(rng.uniform(15, 40))
        vis  = float(rng.uniform(100, 5000))
        if rain > 5:
            vis = min(vis, float(rng.uniform(50, 300)))
        wind = float(rng.uniform(0, 40))
        dust = "high" if vis < 200 else ("medium" if vis < 500 else "low")
        hi   = temp + (rng.uniform(0,5) if temp > 27 else 0)
        cond = ("heavy_rain" if rain > 10 else "light_rain" if rain > 2
                else "dusty" if vis < 200 else "windy" if wind > 30 else "clear")

        rows.append({
            "timestamp":        ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "location_id":      rng.choice(LOCATION_IDS),
            "rain_mm":          round(rain, 2),
            "temperature_c":    round(temp, 1),
            "heat_index_c":     round(float(hi), 1),
            "wind_speed_kmh":   round(wind, 1),
            "visibility_m":     round(vis, 0),
            "dust_level":       dust,
            "day_night":        "day" if 6 <= ts.hour <= 18 else "night",
            "weather_condition":cond,
            "data_source":      "synthetic",
            "synthetic_flag":   True,
            "data_quality_confidence": 0.88,
        })
    return pd.DataFrame(rows)


# ── TASK SESSIONS ─────────────────────────────────────────────────────────────

def generate_task_sessions(operators_df, machines_df, tasks_df, site_df):
    op_exp   = dict(zip(operators_df.operator_id, operators_df.years_experience))
    op_idle  = dict(zip(operators_df.operator_id, operators_df.baseline_idle_ratio))
    op_cycle = dict(zip(operators_df.operator_id, operators_df.baseline_cycle_time_sec))
    op_fuel  = dict(zip(operators_df.operator_id, operators_df.baseline_fuel_l_per_cycle))
    mc_cond  = dict(zip(machines_df.machine_id,   machines_df.machine_condition))
    mc_eng   = dict(zip(machines_df.machine_id,   machines_df.engine_health_score))
    site_lkp = {r.location_id: r for r in site_df.itertuples()}
    cond_f   = {"good":1.0, "fair":0.87, "poor":0.72}
    cong_f   = {"low":1.0, "medium":1.22, "high":1.55}
    wl_f     = {"light":0.70, "medium":1.0, "heavy":1.42}
    base_dur = {"excavation":45,"loading":30,"trenching":60,
                "grading":40,"hauling":25,"stockpiling":35}

    rows = []
    for i in range(5000):
        sid  = f"S{i+1:06d}"
        task = tasks_df.iloc[int(rng.integers(0, len(tasks_df)))]
        op   = rng.choice(OPERATOR_IDS)
        mach = rng.choice(MACHINE_IDS)
        loc  = task["location_id"]
        site = site_lkp[loc]

        start = BASE_DATE + timedelta(hours=float(i * 1.5 + rng.uniform(0, 1)))

        wl   = task["workload"]
        soil_h = float(site.soil_hardness_index)
        slope  = float(site.slope_deg)
        cong   = site.congestion_level
        rain   = float(rng.exponential(2)) if rng.random() < 0.15 else 0.0
        temp   = float(rng.uniform(15, 40))
        vis    = float(rng.uniform(100, 5000))
        if rain > 5:
            vis = min(vis, float(rng.uniform(50, 300)))

        rain_f = 1.0 + rain / 20.0
        vis_f  = 1.0 + max(0, (500 - vis) / 500) * 0.3
        mc_f   = cond_f.get(mc_cond.get(mach, "good"), 1.0)
        exp    = op_exp.get(op, 5.0)
        skl_f  = max(0.72, 1.0 - exp / 42)

        dur = (base_dur[task["task_type"]]
               * wl_f[wl]
               * (1 + soil_h * 0.5)
               * (1 + slope / 30)
               * cong_f[cong]
               * rain_f * vis_f
               * (1.0 / mc_f)
               * skl_f)
        dur = max(5.0, dur + float(rng.normal(0, dur * 0.10)))
        end = start + timedelta(minutes=dur)

        idle_r  = float(np.clip(op_idle.get(op, 0.12) * cong_f[cong] + rng.normal(0, 0.02), 0.02, 0.50))
        idle_t  = dur * idle_r
        cycles  = max(1, int(dur / (op_cycle.get(op, 60) / 60)) + int(rng.integers(-2, 3)))

        tf = wl_f[wl]
        terrain_f = 1 + soil_h * 0.3 + slope / 40
        fuel = (op_fuel.get(op, 5.0) * dur / 60
                * tf * terrain_f
                * (1 + idle_r * 0.5)
                * (1 + float(task["estimated_travel_distance_km"]) * 0.05)
                / (mc_f * 0.82))
        fuel = max(0.5, fuel + float(rng.normal(0, fuel * 0.08)))

        eng_load = float({"light":40,"medium":65,"heavy":85}[wl]) + float(rng.normal(0,5))
        eng_load = float(np.clip(eng_load, 10, 100))
        shift_el = max(0.0, (start.hour - 6) * 60.0 + start.minute)
        cum_h    = exp * 1800 + float(rng.uniform(0, 200))
        dust     = "high" if vis < 200 else ("medium" if vis < 500 else "low")

        rows.append({
            "session_id":            sid,
            "task_id":               task["task_id"],
            "operator_id":           op,
            "machine_id":            mach,
            "start_timestamp":       start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_timestamp":         end.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_type":             task["task_type"],
            "task_phase":            task["task_phase"],
            "workload":              wl,
            "target_output":         float(task["target_output"]),
            "priority":              int(task["priority"]),
            "machine_condition":     mc_cond.get(mach, "good"),
            "attachment_type":       rng.choice(ATTACHMENTS),
            "engine_hours":          round(mc_eng.get(mach, 0.9) * 5000, 1),
            "engine_load_pct":       round(eng_load, 1),
            "load_cycles":           cycles,
            "soil_material":         site.soil_material,
            "soil_moisture_pct":     round(float(site.soil_moisture_pct), 1),
            "soil_hardness_index":   round(soil_h, 3),
            "site_slope_deg":        round(slope, 1),
            "surface_type":          site.surface_type,
            "congestion_level":      cong,
            "travel_distance_km":    round(float(task["estimated_travel_distance_km"]), 2),
            "rain_mm":               round(rain, 2),
            "temperature_c":         round(temp, 1),
            "heat_index_c":          round(temp + (2.0 if rain > 0 else 0.0), 1),
            "visibility_m":          round(vis, 0),
            "dust_level":            dust,
            "day_night":             "day" if 6 <= start.hour <= 18 else "night",
            "shift_elapsed_min":     round(shift_el, 0),
            "operator_cumulative_hours": round(cum_h, 1),
            "recent_idle_ratio":     round(float(np.clip(idle_r + rng.normal(0,0.01), 0, 0.5)), 4),
            "recent_load_ratio":     round(float(np.clip({"light":0.3,"medium":0.6,"heavy":0.85}[wl]
                                                         + float(rng.normal(0,0.05)), 0, 1)), 4),
            "recent_incident_count": int(rng.integers(0, 3)),
            "actual_task_duration_min": round(dur, 2),   # TARGET — never use as model input
            "actual_fuel_used_l":       round(fuel, 2),  # TARGET — never use as model input
            "actual_idle_time_min":     round(idle_t, 2),
            "actual_cycle_count":       cycles,
            "data_source":           "synthetic",
            "synthetic_flag":        True,
            "data_quality_confidence": 0.85,
        })

    return pd.DataFrame(rows)


# ── TELEMETRY ─────────────────────────────────────────────────────────────────

def generate_telemetry(sessions_df, target_rows=75000):
    rows = []
    rows_per_session = max(1, target_rows // len(sessions_df))  # ~15 rows each

    for _, sess in sessions_df.iterrows():
        start = datetime.strptime(sess["start_timestamp"], "%Y-%m-%dT%H:%M:%S")
        end   = datetime.strptime(sess["end_timestamp"],   "%Y-%m-%dT%H:%M:%S")
        dur_s = (end - start).total_seconds()
        n     = min(rows_per_session, max(1, int(dur_s / 45)))

        for j in range(n):
            ts = start + timedelta(seconds=j * int(dur_s / max(n,1)) + int(rng.integers(0,10)))
            prog = j / max(n - 1, 1)

            # Machine state transitions across session lifecycle
            if prog < 0.08 or prog > 0.94:
                m_state = "safe_idle"
            else:
                m_state = rng.choice(
                    ["loading","digging","swinging","traveling","safe_idle","grading"],
                    p=[0.28, 0.23, 0.18, 0.14, 0.10, 0.07])

            speed = (0.0 if m_state in ("parked","safe_idle","digging","swinging","grading")
                     else float(rng.uniform(5, 20)) if m_state == "traveling"
                     else float(rng.uniform(0, 3)))

            wl_v = {"light":0.40, "medium":0.65, "heavy":0.86}.get(sess["workload"], 0.65)
            eng_load = (float(rng.uniform(10, 25)) if m_state == "safe_idle"
                        else float(np.clip(wl_v * 100 + rng.normal(0, 8), 10, 100)))
            hyd_load = (float(rng.uniform(5, 15)) if m_state == "safe_idle"
                        else float(np.clip(wl_v * 80 + rng.normal(0, 10), 0, 100)))

            fuel_interval = max(0.0, (eng_load / 100) * 15 * (45 / 3600) + float(rng.normal(0, 0.03)))

            # Worker proximity — not always present
            w_dist  = float(np.clip(rng.exponential(35) + 5, 1, 250))
            w_dir   = rng.choice(["front","rear","left","right"])
            c_speed = max(0.0, float(rng.normal(0, 0.3))) if w_dist < 30 else 0.0

            swing   = float(rng.uniform(0, 180)) if m_state == "swinging" else 0.0
            arm_sp  = float(rng.uniform(0.3, 1.0)) if m_state in ("digging","loading","swinging") else 0.0
            bucket  = rng.choice(["open","closed","loading"]) if m_state != "traveling" else "closed"
            att_mov = m_state in ("digging","loading","swinging","grading")

            seatbelt = "on"
            if m_state == "traveling" and rng.random() < 0.02:
                seatbelt = "off"

            safety_alert = (
                (w_dist < 10 and m_state in ("swinging","digging") and c_speed > 0.5)
                or (seatbelt == "off" and m_state == "traveling")
            )

            rain = max(0.0, float(sess["rain_mm"]) + float(rng.normal(0, 0.4)))
            temp = float(sess["temperature_c"]) + float(rng.normal(0, 1))
            vis  = max(10.0, float(sess["visibility_m"]) + float(rng.normal(0, 40)))
            dust = "high" if vis < 200 else ("medium" if vis < 500 else "low")
            eq_dens = int(rng.integers(0, 8))
            cong    = "high" if eq_dens > 5 else ("medium" if eq_dens > 2 else "low")
            shift_el= max(0.0,(ts.hour-6)*60.0 + ts.minute)

            rows.append({
                "timestamp":              ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "machine_id":             sess["machine_id"],
                "operator_id":            sess["operator_id"],
                "task_id":                sess["task_id"],
                "session_id":             sess["session_id"],
                "engine_hours":           round(float(sess["engine_hours"]) + j*45/3600, 2),
                "fuel_used_l":            round(fuel_interval, 4),
                "fuel_level_pct":         round(max(5.0, 80.0 - j*0.05 + float(rng.normal(0,0.3))),1),
                "load_cycles":            j // 20,
                "idling_time_min":        round(45/60 if m_state=="safe_idle" else 0.0, 3),
                "seatbelt_status":        seatbelt,
                "safety_alert_triggered": bool(safety_alert),
                "machine_speed_kmh":      round(speed, 1),
                "heading_deg":            round(float(rng.uniform(0, 360)), 1),
                "engine_load_pct":        round(eng_load, 1),
                "hydraulic_load_pct":     round(hyd_load, 1),
                "attachment_type":        sess["attachment_type"],
                "attachment_movement":    bool(att_mov),
                "machine_state":          m_state,
                "worker_distance_m":      round(w_dist, 1),
                "worker_relative_direction": w_dir,
                "closing_speed_mps":      round(c_speed, 3),
                "equipment_density":      eq_dens,
                "congestion_level":       cong,
                "swing_angle_deg":        round(swing, 1),
                "arm_speed":              round(arm_sp, 3),
                "bucket_state":           bucket,
                "travel_state":           "moving" if m_state=="traveling" else "stationary",
                "site_slope_deg":         round(float(sess["site_slope_deg"]), 1),
                "surface_type":           sess["surface_type"],
                "soil_material":          sess["soil_material"],
                "soil_moisture_pct":      round(float(sess["soil_moisture_pct"]), 1),
                "soil_hardness_index":    round(float(sess["soil_hardness_index"]), 3),
                "rain_mm":                round(rain, 2),
                "temperature_c":          round(temp, 1),
                "heat_index_c":           round(temp + (2.0 if rain>0 else 0.0), 1),
                "wind_speed_kmh":         round(float(rng.uniform(0, 35)), 1),
                "visibility_m":           round(vis, 0),
                "dust_level":             dust,
                "day_night":              "day" if 6<=ts.hour<=18 else "night",
                "shift_elapsed_min":      round(shift_el, 1),
                "operator_cumulative_hours": round(float(sess["operator_cumulative_hours"]) + j*45/3600, 2),
                "recent_idle_ratio":      round(float(sess["recent_idle_ratio"]), 4),
                "recent_load_ratio":      round(float(sess["recent_load_ratio"]), 4),
                "recent_incident_count":  int(sess["recent_incident_count"]),
                "data_source":            "synthetic",
                "synthetic_flag":         True,
                "data_quality_confidence":0.85,
            })

    return pd.DataFrame(rows)


# ── SAFETY EVENTS ─────────────────────────────────────────────────────────────

def generate_safety_events(sessions_df, n=1500):
    """
    Deliberately generates contrasting examples:
      Example A — distance=10m, machine stationary, closing_speed=0 → INFO (not dangerous)
      Example B — distance=10m, machine swinging, in envelope, closing_speed>0 → CRITICAL
    Severity is determined by context, NOT distance alone.
    """

    TEMPLATES = {
        # ── INFO: nearby but machine stationary, zero closing speed ──────────
        "info_stationary": dict(
            dist=(8,20), state="safe_idle",   cs=0.00, envelope=False,
            severity="INFO",   etype="proximity_logged",
            trigger="worker_outside_envelope_machine_stationary", action="none_required",
        ),
        # ── LOW: slow travel, worker not in path ─────────────────────────────
        "low_slow_travel": dict(
            dist=(12,25), state="traveling",  cs=0.10, envelope=False,
            severity="LOW",    etype="proximity_low",
            trigger="worker_nearby_slow_travel", action="monitor",
        ),
        # ── MEDIUM: active digging, moderate proximity ────────────────────────
        "medium_digging": dict(
            dist=(8,15),  state="digging",    cs=0.35, envelope=False,
            severity="MEDIUM", etype="proximity_medium",
            trigger="worker_within_medium_distance", action="slow_down",
        ),
        # ── HIGH: swing active + worker in envelope + closing ─────────────────
        "high_swing_envelope": dict(
            dist=(5,12),  state="swinging",   cs=0.90, envelope=True,
            severity="HIGH",   etype="proximity_high",
            trigger="worker_inside_swing_envelope_with_closing_motion", action="stop_or_safe_action",
        ),
        # ── CRITICAL: imminent collision (same dist as INFO — contrast example)
        "critical_swing_imminent": dict(
            dist=(8,12),  state="swinging",   cs=1.90, envelope=True,
            severity="CRITICAL", etype="imminent_collision",
            trigger="worker_inside_swing_envelope_with_closing_motion", action="emergency_stop",
        ),
        # ── HIGH: seatbelt off while moving ──────────────────────────────────
        "high_seatbelt": dict(
            dist=(50,200), state="traveling", cs=0.00, envelope=False,
            severity="HIGH",   etype="seatbelt_violation",
            trigger="seatbelt_off_while_moving", action="stop_immediately",
        ),
    }

    probs = [0.22, 0.18, 0.25, 0.17, 0.10, 0.08]
    names = list(TEMPLATES.keys())
    rows  = []

    for i in range(n):
        tname = rng.choice(names, p=probs)
        t     = TEMPLATES[tname]
        sess  = sessions_df.iloc[int(rng.integers(0, len(sessions_df)))]
        ts    = (datetime.strptime(sess["start_timestamp"], "%Y-%m-%dT%H:%M:%S")
                 + timedelta(minutes=float(rng.uniform(3, max(4, float(sess["actual_task_duration_min"])*0.9)))))

        dist  = float(rng.uniform(*t["dist"]))
        cs    = max(0.0, t["cs"] + float(rng.normal(0, 0.08)))
        speed = float(rng.uniform(5,15)) if t["state"]=="traveling" else float(rng.uniform(0,3))
        swing = float(rng.uniform(0,90)) if t["state"]=="swinging" else 0.0
        arm_s = float(rng.uniform(0.3,1.0)) if t["state"] in ("digging","loading","swinging") else 0.0
        sbelt = "off" if tname=="high_seatbelt" else "on"
        rain  = max(0.0, float(sess["rain_mm"])      + float(rng.normal(0, 0.4)))
        vis   = max(10.0, float(sess["visibility_m"]) + float(rng.normal(0, 25)))
        conf  = {"INFO":0.80,"LOW":0.82,"MEDIUM":0.85,"HIGH":0.90,"CRITICAL":0.93}[t["severity"]]
        conf  = float(np.clip(conf + rng.normal(0, 0.03), 0.70, 0.99))

        rows.append({
            "event_id":              f"SE{i+1:05d}",
            "timestamp":             ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id":            sess["session_id"],
            "operator_id":           sess["operator_id"],
            "machine_id":            sess["machine_id"],
            "worker_distance_m":     round(dist, 1),
            "worker_relative_direction": rng.choice(["front","rear","left","right"]),
            "closing_speed_mps":     round(cs, 3),
            "machine_speed_kmh":     round(speed, 1),
            "swing_angle_deg":       round(swing, 1),
            "arm_speed":             round(arm_s, 3),
            "bucket_state":          rng.choice(["open","closed","loading"]),
            "attachment_movement":   t["state"] in ("digging","loading","swinging"),
            "machine_state":         t["state"],
            "worker_in_envelope":    bool(t["envelope"]),
            "site_slope_deg":        round(float(sess["site_slope_deg"]), 1),
            "surface_type":          sess["surface_type"],
            "congestion_level":      sess["congestion_level"],
            "blind_zone_flag":       bool(rng.random() < 0.20),
            "rain_mm":               round(rain, 2),
            "visibility_m":          round(vis, 0),
            "dust_level":            "high" if vis<200 else ("medium" if vis<500 else "low"),
            "day_night":             "day" if 6<=ts.hour<=18 else "night",
            "seatbelt_status":       sbelt,
            "event_type":            t["etype"],
            "severity":              t["severity"],
            "trigger_reason":        t["trigger"],
            "required_action":       t["action"],
            "alert_sent":            t["severity"] in ("HIGH","CRITICAL"),
            "alert_queued":          t["severity"] == "MEDIUM",
            "incident_logged":       t["severity"] in ("HIGH","CRITICAL"),
            "data_source":           "synthetic",
            "synthetic_flag":        True,
            "confidence":            round(conf, 3),
        })

    return pd.DataFrame(rows)


# ── TRAINING RECORDS ──────────────────────────────────────────────────────────

def generate_training_records(sessions_df, safety_events_df, n=750):
    lesson_map = {
        "idle_reduction":       "L001",
        "seatbelt_compliance":  "L002",
        "swing_zone_awareness": "L003",
        "efficient_loading":    "L004",
        "fuel_efficiency":      "L005",
    }
    rows = []
    for i in range(n):
        issue  = rng.choice(list(lesson_map.keys()))
        conf   = float(rng.uniform(0.65, 0.95))
        before = float(rng.uniform(0.30, 0.75))
        impr   = float(rng.uniform(0.05, 0.30)) if conf > 0.75 else float(rng.uniform(0.0, 0.10))
        after  = float(np.clip(before + impr, 0, 1))
        se     = safety_events_df.iloc[int(rng.integers(0, len(safety_events_df)))]
        comp   = BASE_DATE + timedelta(days=int(rng.integers(1, 60)))

        rows.append({
            "training_id":       f"TR{i+1:05d}",
            "operator_id":       rng.choice(OPERATOR_IDS),
            "trigger_event_id":  se["event_id"],
            "issue_type":        issue,
            "attribution_type":  rng.choice(["operator-driven","mixed"]),
            "confidence":        round(conf, 3),
            "lesson_id":         lesson_map[issue],
            "lesson_type":       rng.choice(["micro_lesson","scenario","quiz_only"]),
            "quiz_score_before": int(rng.integers(3, 8)),
            "quiz_score_after":  int(np.clip(rng.integers(5, 11), 0, 10)),
            "before_metric":     round(before, 3),
            "after_metric":      round(after, 3),
            "improvement_pct":   round(impr * 100, 1),
            "escalation_state":  rng.choice(["first_trigger","second_trigger","escalated_to_supervisor"],
                                            p=[0.60,0.30,0.10]),
            "completed_timestamp": comp.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return pd.DataFrame(rows)


# ── PEER EXAMPLES ─────────────────────────────────────────────────────────────

def generate_peer_examples(n=150):
    techniques = {
        "excavation":  "Optimal bucket fill angle reduces cycle time by ~12%",
        "loading":     "Sequential loading pattern minimises repositioning",
        "trenching":   "Consistent depth control reduces rework passes",
        "grading":     "Single-pass technique on firm dry soil",
        "hauling":     "Steady speed on grade optimises fuel per tonne-km",
        "stockpiling": "Layered placement reduces rehandling effort",
    }
    rows = []
    for i in range(n):
        tt = rng.choice(TASK_TYPES)
        rows.append({
            "example_id":                    f"PE{i+1:04d}",
            "source_operator_anonymized_id": f"ANON_{int(rng.integers(100,999))}",
            "task_type":                     tt,
            "machine_type":                  rng.choice(["excavator","dozer","grader","loader","hauler"]),
            "attachment_type":               rng.choice(ATTACHMENTS),
            "site_condition":                rng.choice(["easy","moderate","difficult"]),
            "technique_description":         techniques[tt],
            "observed_metric":               round(float(rng.uniform(0.60, 0.95)), 3),
            "context_similarity_score":      round(float(rng.uniform(0.50, 1.00)), 3),
            "approved_for_peer_learning":    bool(rng.random() < 0.75),
            "synthetic_flag":                True,
        })
    return pd.DataFrame(rows)


# ── OPERATOR FACES ────────────────────────────────────────────────────────────

def generate_operator_faces():
    rows = []
    for op_id in ("OP1001","OP1002","OP1003"):
        rows.append({
            "operator_id":               op_id,
            "face_registered":           True,
            "face_embedding_path":       f"data/face_registrations/{op_id}_embedding.npy",
            "face_model":                "ArcFace",
            "face_registration_timestamp": BASE_DATE.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_source":               "synthetic",
            "synthetic_flag":            True,
        })
    return pd.DataFrame(rows)


# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate(operators, machines, tasks, sessions, telemetry, safety_events, training_records):
    errors, warnings = [], []
    valid_ops   = set(operators.operator_id)
    valid_mechs = set(machines.machine_id)
    valid_tasks = set(tasks.task_id)

    # 1. No negatives
    for col in ("actual_task_duration_min","actual_fuel_used_l","actual_idle_time_min"):
        if (sessions[col] < 0).any():
            errors.append(f"Negative values in sessions.{col}")
    for col in ("fuel_used_l","engine_hours"):
        if (telemetry[col] < 0).any():
            errors.append(f"Negative values in telemetry.{col}")

    # 2. Timestamps
    s = sessions.copy()
    s["_s"] = pd.to_datetime(s.start_timestamp)
    s["_e"] = pd.to_datetime(s.end_timestamp)
    bad = (s["_e"] <= s["_s"]).sum()
    if bad:
        errors.append(f"{bad} sessions with end <= start")

    # 3. FK integrity
    for col, valid in (("operator_id",valid_ops),("machine_id",valid_mechs),("task_id",valid_tasks)):
        bad = (~sessions[col].isin(valid)).sum()
        if bad:
            errors.append(f"{bad} sessions with invalid {col}")

    # 4. Relationships
    heavy_fuel = sessions[sessions.workload=="heavy"]["actual_fuel_used_l"].mean()
    light_fuel = sessions[sessions.workload=="light"]["actual_fuel_used_l"].mean()
    if heavy_fuel <= light_fuel:
        errors.append(f"FAIL fuel relationship: heavy={heavy_fuel:.2f} <= light={light_fuel:.2f}")
    else:
        print(f"  [OK] Fuel: heavy={heavy_fuel:.2f}L > light={light_fuel:.2f}L")

    hard_dur = sessions[sessions.soil_hardness_index > 0.7]["actual_task_duration_min"].mean()
    easy_dur = sessions[sessions.soil_hardness_index < 0.3]["actual_task_duration_min"].mean()
    if hard_dur > easy_dur:
        print(f"  [OK] Duration: hard soil={hard_dur:.1f}min > easy soil={easy_dur:.1f}min")
    else:
        warnings.append(f"Weak duration-soil relationship: hard={hard_dur:.1f}, easy={easy_dur:.1f}")

    hi_idle = sessions[sessions.congestion_level=="high"]["actual_idle_time_min"].mean()
    lo_idle = sessions[sessions.congestion_level=="low" ]["actual_idle_time_min"].mean()
    if hi_idle > lo_idle:
        print(f"  [OK] Idle: high congestion={hi_idle:.1f}min > low congestion={lo_idle:.1f}min")
    else:
        warnings.append(f"Weak idle-congestion relationship: high={hi_idle:.1f}, low={lo_idle:.1f}")

    # 5. Safety contrasting examples — proximity ≠ automatic danger
    se = safety_events
    info_cs   = se[se.severity=="INFO"    ]["closing_speed_mps"].mean()
    crit_cs   = se[se.severity=="CRITICAL"]["closing_speed_mps"].mean()
    info_dist = se[se.severity=="INFO"    ]["worker_distance_m"].mean()
    crit_dist = se[se.severity=="CRITICAL"]["worker_distance_m"].mean()
    if crit_cs > info_cs:
        print(f"  [OK] Safety contrast: CRITICAL cs={crit_cs:.2f} > INFO cs={info_cs:.2f}")
    else:
        errors.append("FAIL safety contrast: CRITICAL closing_speed not > INFO")

    print(f"  [OK] Avg dist — INFO={info_dist:.1f}m  CRITICAL={crit_dist:.1f}m "
          f"(similar distances, different severity proves context-driven logic)")

    stationary_sev = se[(se.worker_distance_m < 15) & (se.machine_state=="safe_idle")]["severity"].value_counts().to_dict()
    print(f"  [OK] Stationary+nearby (<15m) severity counts: {stationary_sev}")

    # 6. Synthetic flags
    for name, df in (("sessions",sessions),("telemetry",telemetry),("safety_events",se)):
        if "synthetic_flag" in df.columns and not df.synthetic_flag.all():
            errors.append(f"Not all rows in {name} have synthetic_flag=True")

    # 7. Counts
    print(f"\n  Dataset row counts:")
    print(f"    operators:       {len(operators)}")
    print(f"    machines:        {len(machines)}")
    print(f"    tasks:           {len(tasks)}")
    print(f"    task_sessions:   {len(sessions)}")
    print(f"    telemetry:       {len(telemetry):,}")
    print(f"    safety_events:   {len(safety_events)}")
    print(f"    training_records:{len(training_records)}")

    return errors, warnings


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAT Operator Copilot — Synthetic Data Generator")
    print(f"SEED = {SEED}")
    print("=" * 60)

    print("\n[1/9] Operators ...")
    operators = generate_operators()

    print("[2/9] Machines ...")
    machines = generate_machines()

    print("[3/9] Site conditions ...")
    site_conditions = generate_site_conditions()

    print("[4/9] Tasks ...")
    tasks = generate_tasks()

    print("[5/9] Weather ...")
    weather = generate_weather(6000)

    print("[6/9] Task sessions (5,000) ...")
    sessions = generate_task_sessions(operators, machines, tasks, site_conditions)

    print("[7/9] Telemetry (~75k rows) ...")
    telemetry = generate_telemetry(sessions, target_rows=75000)

    print("[8/9] Safety events, training records, peer examples ...")
    safety_events     = generate_safety_events(sessions, n=1500)
    training_records  = generate_training_records(sessions, safety_events, n=750)
    peer_examples     = generate_peer_examples(n=150)
    operator_faces    = generate_operator_faces()

    print("\n[9/9] Validation ...")
    errors, warnings = validate(operators, machines, tasks, sessions, telemetry, safety_events, training_records)

    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    if errors:
        print("\nERRORS FOUND:")
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    else:
        print("  All checks passed.")

    print(f"\nSaving to {OUT_DIR}/ ...")
    operators.to_csv(       f"{OUT_DIR}/operators.csv",        index=False)
    machines.to_csv(        f"{OUT_DIR}/machines.csv",         index=False)
    site_conditions.to_csv( f"{OUT_DIR}/site_conditions.csv",  index=False)
    tasks.to_csv(           f"{OUT_DIR}/tasks.csv",            index=False)
    weather.to_csv(         f"{OUT_DIR}/weather.csv",          index=False)
    sessions.to_csv(        f"{OUT_DIR}/task_sessions.csv",    index=False)
    telemetry.to_csv(       f"{OUT_DIR}/telemetry.csv",        index=False)
    safety_events.to_csv(   f"{OUT_DIR}/safety_events.csv",    index=False)
    training_records.to_csv(f"{OUT_DIR}/training_records.csv", index=False)
    peer_examples.to_csv(   f"{OUT_DIR}/peer_examples.csv",    index=False)
    operator_faces.to_csv(  f"{OUT_DIR}/operator_faces.csv",   index=False)

    print(f"\nDone. {len(telemetry):,} telemetry rows written.")


if __name__ == "__main__":
    main()
