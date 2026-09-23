"""
Train ETA and Fuel prediction models.
Run from project root: python3 models/train_all.py

Features: context + operator skill (NO actual_task_duration_min, NO actual_fuel_used_l — no leakage).
Algorithm: LightGBM quantile regression → P10, P50, P90.
Split: 80/20 train/test (random, seed=42).
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    GB_AVAILABLE = True
except ImportError:
    GB_AVAILABLE = False

SEED     = 42
DATA_DIR = "data/synthetic"
REPORTS  = "reports"
os.makedirs(REPORTS, exist_ok=True)
os.makedirs("models/eta_model", exist_ok=True)
os.makedirs("models/fuel_model", exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────
# NEVER include actual_task_duration_min or actual_fuel_used_l as inputs.

NUMERIC_FEATURES = [
    "soil_hardness_index",
    "site_slope_deg",
    "rain_mm",
    "visibility_m",
    "temperature_c",
    "travel_distance_km",
    "engine_load_pct",
    "soil_moisture_pct",
    "recent_idle_ratio",
    "recent_load_ratio",
    "recent_incident_count",
    "years_experience",       # joined from operators.csv
]

CATEGORICAL_FEATURES = [
    "workload",
    "machine_condition",
    "congestion_level",
    "task_type",
    "task_phase",
    "day_night",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_ETA  = "actual_task_duration_min"
TARGET_FUEL = "actual_fuel_used_l"

CAT_MAPS = {}  # populated during prepare_data


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data():
    sessions  = pd.read_csv(f"{DATA_DIR}/task_sessions.csv")
    operators = pd.read_csv(f"{DATA_DIR}/operators.csv")[["operator_id","years_experience"]]
    df = sessions.merge(operators, on="operator_id", how="left")

    df = df.dropna(subset=[TARGET_ETA, TARGET_FUEL])
    df = df[(df[TARGET_ETA] > 0) & (df[TARGET_FUEL] > 0)]

    # Label-encode categoricals
    for col in CATEGORICAL_FEATURES:
        cats = sorted(df[col].dropna().astype(str).unique().tolist())
        CAT_MAPS[col] = {v: i for i, v in enumerate(cats)}
        df[col] = df[col].astype(str).map(CAT_MAPS[col]).fillna(-1).astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if col in df else 0)

    X      = df[ALL_FEATURES].copy()
    y_eta  = df[TARGET_ETA].values
    y_fuel = df[TARGET_FUEL].values

    return X, y_eta, y_fuel


# ── Model training ────────────────────────────────────────────────────────────

def train_lgb_quantile(X_tr, y_tr, cat_indices):
    models = {}
    for alpha, key in [(0.10,"p10"), (0.50,"p50"), (0.90,"p90")]:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=SEED,
            verbose=-1,
        )
        m.fit(X_tr, y_tr, categorical_feature=cat_indices)
        models[key] = m
    return models


def train_sklearn_quantile(X_tr, y_tr, _cat_indices):
    """Fallback when LightGBM is unavailable — three GBR models at different quantiles."""
    from sklearn.ensemble import GradientBoostingRegressor
    models = {}
    for q, key in [(0.10,"p10"), (0.50,"p50"), (0.90,"p90")]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=150, learning_rate=0.08,
            max_depth=5, random_state=SEED,
        )
        m.fit(X_tr.values, y_tr)
        models[key] = m
    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_models(models, X_test, y_test, baseline_values):
    p50 = models["p50"].predict(X_test)
    mae  = float(mean_absolute_error(y_test, p50))
    rmse = float(np.sqrt(mean_squared_error(y_test, p50)))

    # Naive baseline: median of training set values
    baseline_pred = np.full_like(y_test, float(np.median(baseline_values)))
    b_mae = float(mean_absolute_error(y_test, baseline_pred))

    # Coverage of P10–P90 interval
    p10 = models["p10"].predict(X_test)
    p90 = models["p90"].predict(X_test)
    coverage = float(np.mean((y_test >= p10) & (y_test <= p90)))

    return {"mae": round(mae,3), "rmse": round(rmse,3),
            "baseline_mae": round(b_mae,3), "p10_p90_coverage": round(coverage,3)}


def get_importance(model, feature_names, top_n=10):
    try:
        imp = model.feature_importances_
    except AttributeError:
        return []
    pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v),4)} for f,v in pairs[:top_n]]


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(models, model_dir, feature_names):
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(models, f)
    preprocessing = {
        "features": feature_names,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "cat_maps": CAT_MAPS,
    }
    with open(f"{model_dir}/preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)
    print(f"  Saved artifacts to {model_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAT Operator Copilot — Model Training")
    print("=" * 60)

    print("\nPreparing data ...")
    X, y_eta, y_fuel = prepare_data()
    feature_names = list(X.columns)
    cat_indices   = [feature_names.index(c) for c in CATEGORICAL_FEATURES]

    X_tr, X_te, y_eta_tr, y_eta_te, y_fuel_tr, y_fuel_te = train_test_split(
        X, y_eta, y_fuel, test_size=0.20, random_state=SEED
    )
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    trainer = train_lgb_quantile if LGB_AVAILABLE else train_sklearn_quantile
    algo    = "LightGBM" if LGB_AVAILABLE else "GradientBoostingRegressor (sklearn fallback)"
    print(f"  Algorithm: {algo}")

    print("\nTraining ETA model ...")
    eta_models  = trainer(X_tr, y_eta_tr, cat_indices)
    eta_metrics = evaluate_models(eta_models, X_te, y_eta_te, y_eta_tr)
    print(f"  MAE={eta_metrics['mae']:.2f} min  RMSE={eta_metrics['rmse']:.2f} min  "
          f"Baseline={eta_metrics['baseline_mae']:.2f} min  "
          f"P10-P90 coverage={eta_metrics['p10_p90_coverage']:.1%}")
    save_artifacts(eta_models, "models/eta_model", feature_names)

    print("\nTraining Fuel model ...")
    fuel_models  = trainer(X_tr, y_fuel_tr, cat_indices)
    fuel_metrics = evaluate_models(fuel_models, X_te, y_fuel_te, y_fuel_tr)
    print(f"  MAE={fuel_metrics['mae']:.2f} L  RMSE={fuel_metrics['rmse']:.2f} L  "
          f"Baseline={fuel_metrics['baseline_mae']:.2f} L  "
          f"P10-P90 coverage={fuel_metrics['p10_p90_coverage']:.1%}")
    save_artifacts(fuel_models, "models/fuel_model", feature_names)

    # ── Write metrics report ─────────────────────────────────────────────────
    imp_eta  = get_importance(eta_models["p50"],  feature_names)
    imp_fuel = get_importance(fuel_models["p50"], feature_names)

    metrics = {
        "model_honesty": {
            "training_data": "synthetic",
            "synthetic_flag": True,
            "disclaimer": (
                "All metrics are from synthetic data. "
                "Feature importance indicates predictive contribution, NOT causality. "
                "Never claim these models prove operator performance."
            ),
        },
        "eta_model": {
            "target":         TARGET_ETA,
            "algorithm":      algo,
            "features_used":  feature_names,
            "forbidden_inputs": [TARGET_ETA, TARGET_FUEL],
            "train_samples":  len(X_tr),
            "test_samples":   len(X_te),
            "split":          "80/20 random stratified by seed=42",
            "outputs":        "P10, P50, P90 (quantile regression)",
            "mae_minutes":    eta_metrics["mae"],
            "rmse_minutes":   eta_metrics["rmse"],
            "baseline_mae_minutes": eta_metrics["baseline_mae"],
            "improvement_over_baseline_pct": round(
                (1 - eta_metrics["mae"] / max(eta_metrics["baseline_mae"], 1e-6)) * 100, 1),
            "p10_p90_coverage": eta_metrics["p10_p90_coverage"],
            "top_contributing_features": imp_eta,
            "limitations":    "Trained on synthetic data only.",
        },
        "fuel_model": {
            "target":         TARGET_FUEL,
            "algorithm":      algo,
            "features_used":  feature_names,
            "forbidden_inputs": [TARGET_ETA, TARGET_FUEL],
            "train_samples":  len(X_tr),
            "test_samples":   len(X_te),
            "split":          "80/20 random stratified by seed=42",
            "outputs":        "P10, P50, P90 (quantile regression)",
            "mae_litres":     fuel_metrics["mae"],
            "rmse_litres":    fuel_metrics["rmse"],
            "baseline_mae_litres": fuel_metrics["baseline_mae"],
            "improvement_over_baseline_pct": round(
                (1 - fuel_metrics["mae"] / max(fuel_metrics["baseline_mae"], 1e-6)) * 100, 1),
            "p10_p90_coverage": fuel_metrics["p10_p90_coverage"],
            "top_contributing_features": imp_fuel,
            "limitations":    "Trained on synthetic data only.",
        },
    }

    report_path = f"{REPORTS}/model_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
