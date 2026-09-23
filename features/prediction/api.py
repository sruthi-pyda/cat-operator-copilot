"""
Feature 04 — Predictive Task Intelligence API
Feature 03 — Behavioral Fingerprint API
Owner: Sruthi (Team Member 1)

Public functions:
  predict_task(session_context)    → PredictionResult
  predict_fuel(session_context)    → PredictionResult
  analyze_behavior(session_context) → BehaviorResult

All functions accept shared.schemas.SessionContext.
All outputs use shared.schemas types — no proprietary objects.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.schemas import SessionContext, PredictionResult, BehaviorResult
from shared.constants import DATA_SOURCE_SYNTHETIC, DATA_QUALITY_CONFIDENCE_DEFAULT
from features.behavior.behavior_model import BehaviorModel  # required for pickle deserialization

DATA_DIR       = "data/synthetic"
ETA_MODEL_DIR  = "models/eta_model"
FUEL_MODEL_DIR = "models/fuel_model"
BEHAVIOR_MODEL_PATH = "models/behavior_model/behavior_model.pkl"

_eta_models   = None
_fuel_models  = None
_eta_prep     = None
_fuel_prep    = None
_behavior_model = None
_sessions_df  = None
_operators_df = None


# ── Model loading (lazy, cached) ──────────────────────────────────────────────

def _load_prediction_models():
    global _eta_models, _fuel_models, _eta_prep, _fuel_prep
    if _eta_models is None:
        with open(f"{ETA_MODEL_DIR}/model.pkl",          "rb") as f: _eta_models  = pickle.load(f)
        with open(f"{ETA_MODEL_DIR}/preprocessing.json", "r") as f: _eta_prep    = json.load(f)
    if _fuel_models is None:
        with open(f"{FUEL_MODEL_DIR}/model.pkl",          "rb") as f: _fuel_models = pickle.load(f)
        with open(f"{FUEL_MODEL_DIR}/preprocessing.json", "r") as f: _fuel_prep   = json.load(f)


def _load_behavior_model():
    global _behavior_model
    if _behavior_model is None:
        with open(BEHAVIOR_MODEL_PATH, "rb") as f:
            _behavior_model = pickle.load(f)


def _load_data():
    global _sessions_df, _operators_df
    if _sessions_df is None:
        _sessions_df  = pd.read_csv(f"{DATA_DIR}/task_sessions.csv")
        _operators_df = pd.read_csv(f"{DATA_DIR}/operators.csv")


# ── Feature extraction from SessionContext ────────────────────────────────────

def _context_to_feature_row(ctx: SessionContext, prep: dict) -> pd.DataFrame:
    """
    Extract model features from a SessionContext.
    Falls back to sensible defaults for missing sub-contexts.
    """
    cat_maps  = prep["cat_maps"]
    features  = prep["features"]
    cat_feats = prep["categorical_features"]

    site    = ctx.site    or {}
    weather = ctx.weather or {}
    machine = ctx.machine or {}
    task    = ctx.task    or {}
    temporal= ctx.temporal or {}
    op      = ctx.operator or {}

    # Helper to get from dataclass or dict
    def g(obj, key, default=0.0):
        if obj is None: return default
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default) or default

    raw = {
        "soil_hardness_index":  g(site,    "hardness",          0.5),
        "site_slope_deg":       g(site,    "slope",             5.0),
        "rain_mm":              g(weather, "rain_mm",           0.0),
        "visibility_m":         g(weather, "visibility_m",    2000.0),
        "temperature_c":        g(weather, "temperature_c",    25.0),
        "travel_distance_km":   g(task,    "planned_sequence",  1.0),   # fallback
        "engine_load_pct":      g(machine, "load",             65.0),
        "soil_moisture_pct":    30.0,
        "recent_idle_ratio":    g(temporal,"recent_idle_ratio", 0.12),
        "recent_load_ratio":    g(temporal,"recent_load_ratio", 0.60),
        "recent_incident_count":g(temporal,"recent_incidents",   0),
        "years_experience":     g(op,      "experience",         5.0),
        "workload":             g(task,    "workload",          "medium"),
        "machine_condition":    g(machine, "machine_condition", "good"),
        "congestion_level":     g(site,    "congestion",        "low"),
        "task_type":            g(task,    "task_type",         "excavation"),
        "task_phase":           g(task,    "task_phase",        "active"),
        "day_night":            g(weather, "day_night",         "day"),
    }

    # Encode categoricals using saved maps
    for col in cat_feats:
        val = str(raw.get(col, ""))
        raw[col] = cat_maps.get(col, {}).get(val, -1)

    df = pd.DataFrame([{k: raw[k] for k in features}])
    return df


# ── Prediction helpers ────────────────────────────────────────────────────────

def _predict_with_percentiles(models, X, session_id: str,
                               prep: dict, label: str) -> PredictionResult:
    p10 = float(models["p10"].predict(X)[0])
    p50 = float(models["p50"].predict(X)[0])
    p90 = float(models["p90"].predict(X)[0])

    # Clamp and order
    p10 = max(0.1, p10)
    p50 = max(p10, p50)
    p90 = max(p50, p90)

    # Confidence: narrower interval → higher confidence
    interval_width = p90 - p10
    relative_width = interval_width / max(1.0, p50)
    confidence = float(np.clip(1.0 - relative_width * 0.5, 0.40, 0.96))

    # Contributing features (from p50 model feature importances)
    try:
        imp = models["p50"].feature_importances_
        top = sorted(zip(prep["features"], imp), key=lambda x: -x[1])[:5]
        factors = [f for f, _ in top]
    except AttributeError:
        factors = prep["features"][:5]

    if label == "eta":
        return PredictionResult(
            session_id=session_id,
            eta_p10=round(p10,2), eta_p50=round(p50,2), eta_p90=round(p90,2),
            fuel_p10=0.0, fuel_p50=0.0, fuel_p90=0.0,
            confidence=round(confidence,3),
            factors=factors,
            predicted_vs_actual=None,
            synthetic_flag=True,
            data_quality_confidence=DATA_QUALITY_CONFIDENCE_DEFAULT,
        )
    else:
        return PredictionResult(
            session_id=session_id,
            eta_p10=0.0, eta_p50=0.0, eta_p90=0.0,
            fuel_p10=round(p10,2), fuel_p50=round(p50,2), fuel_p90=round(p90,2),
            confidence=round(confidence,3),
            factors=factors,
            predicted_vs_actual=None,
            synthetic_flag=True,
            data_quality_confidence=DATA_QUALITY_CONFIDENCE_DEFAULT,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def predict_task(session_context: SessionContext) -> PredictionResult:
    """
    Predict task completion time (ETA) with uncertainty.
    Returns P10/P50/P90 in minutes.
    DO NOT pass actual_task_duration_min as input — this function never accesses it.
    """
    _load_prediction_models()
    X = _context_to_feature_row(session_context, _eta_prep)
    return _predict_with_percentiles(
        _eta_models, X, session_context.session_id, _eta_prep, "eta"
    )


def predict_fuel(session_context: SessionContext) -> PredictionResult:
    """
    Predict fuel consumption with uncertainty.
    Returns P10/P50/P90 in litres.
    DO NOT pass actual_fuel_used_l as input — this function never accesses it.
    """
    _load_prediction_models()
    X = _context_to_feature_row(session_context, _fuel_prep)
    return _predict_with_percentiles(
        _fuel_models, X, session_context.session_id, _fuel_prep, "fuel"
    )


def predict_combined(session_context: SessionContext) -> PredictionResult:
    """Convenience: returns both ETA and Fuel in one PredictionResult."""
    _load_prediction_models()
    X_eta  = _context_to_feature_row(session_context, _eta_prep)
    X_fuel = _context_to_feature_row(session_context, _fuel_prep)

    eta_r  = _predict_with_percentiles(_eta_models,  X_eta,  session_context.session_id, _eta_prep,  "eta")
    fuel_r = _predict_with_percentiles(_fuel_models, X_fuel, session_context.session_id, _fuel_prep, "fuel")

    combined_conf = round((eta_r.confidence + fuel_r.confidence) / 2, 3)
    factors = list(dict.fromkeys(eta_r.factors + fuel_r.factors))[:6]

    return PredictionResult(
        session_id=session_context.session_id,
        eta_p10=eta_r.eta_p10, eta_p50=eta_r.eta_p50, eta_p90=eta_r.eta_p90,
        fuel_p10=fuel_r.fuel_p10, fuel_p50=fuel_r.fuel_p50, fuel_p90=fuel_r.fuel_p90,
        confidence=combined_conf,
        factors=factors,
        predicted_vs_actual=None,
        synthetic_flag=True,
        data_quality_confidence=DATA_QUALITY_CONFIDENCE_DEFAULT,
    )


def analyze_behavior(session_context: SessionContext) -> BehaviorResult:
    """
    Analyze behavioral fingerprint for a completed session.

    Looks up actual session metrics from task_sessions.csv using session_id.
    Two-stage logic: context → expected → residual → attribution.
    Coaching is NEVER triggered for low-confidence or context-explained behavior.
    """
    _load_behavior_model()
    _load_data()

    session_id = session_context.session_id
    sess_row = _sessions_df[_sessions_df["session_id"] == session_id]

    if sess_row.empty:
        # No saved session — use context fields if available
        return BehaviorResult(
            session_id=session_id,
            attribution="insufficient_evidence",
            confidence=0.0,
            observed_value=0.0,
            expected_value=0.0,
            operator_residual=0.0,
            context_explained_component=0.0,
            coaching_eligible=False,
            synthetic_flag=True,
        )

    row = sess_row.iloc[0]
    op_row = _operators_df[_operators_df["operator_id"] == row["operator_id"]]
    baseline_idle = float(op_row["baseline_idle_ratio"].iloc[0]) if not op_row.empty else 0.12

    return _behavior_model.analyze_from_session_row(row, baseline_idle)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    from shared.schemas import (SessionContext, OperatorContext, MachineContext,
                                  TaskContext, SiteContext, WeatherContext, TemporalContext)

    ctx = SessionContext(
        session_id="S000001",
        operator_id="OP1001",
        machine_id="EXC001",
        task_id="T00001",
        timestamp="2026-01-01T08:00:00",
        operator=OperatorContext(operator_id="OP1001", experience=8.0, machine_skill="advanced"),
        machine=MachineContext(machine_id="EXC001", machine_condition="good", load=0.65),
        task=TaskContext(task_id="T00001", task_type="excavation", task_phase="active", workload="medium"),
        site=SiteContext(hardness=0.5, slope=6.0, congestion="low"),
        weather=WeatherContext(rain_mm=0.0, visibility_m=2000.0, temperature_c=24.0, day_night="day"),
        temporal=TemporalContext(recent_idle_ratio=0.12, recent_load_ratio=0.60, recent_incidents=0),
    )

    print("\n--- Smoke test: predict_task ---")
    eta = predict_task(ctx)
    print(f"  ETA P10={eta.eta_p10:.1f}  P50={eta.eta_p50:.1f}  P90={eta.eta_p90:.1f} min  "
          f"confidence={eta.confidence}")

    print("--- Smoke test: predict_fuel ---")
    fuel = predict_fuel(ctx)
    print(f"  Fuel P10={fuel.fuel_p10:.2f}  P50={fuel.fuel_p50:.2f}  P90={fuel.fuel_p90:.2f} L  "
          f"confidence={fuel.confidence}")

    print("--- Smoke test: analyze_behavior ---")
    beh = analyze_behavior(ctx)
    print(f"  attribution={beh.attribution}  confidence={beh.confidence}  "
          f"coaching_eligible={beh.coaching_eligible}")

    print("  [OK] All API functions returned without error.")


if __name__ == "__main__":
    _smoke_test()
