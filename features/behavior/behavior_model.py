"""
Feature 03 — Behavioral Fingerprint
Owner: Sruthi (Team Member 1)

Two-stage logic (architecture requirement):
  Stage 1: context → expected behavior
  Stage 2: observed - expected = contextual residual → classify attribution

Attribution categories:
  operator-driven | context-driven | mixed | insufficient_evidence

Rule: coaching is NEVER triggered for low-confidence or context-explained behavior.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.schemas import SessionContext, BehaviorResult
from shared.constants import DATA_SOURCE_SYNTHETIC

DATA_DIR   = "data/synthetic"
MODEL_PATH = "models/behavior_model/behavior_model.pkl"

# Context-only features (NO operator_id — operator identity must not contaminate
# the expected-behavior model, or we can't separate context from operator effect)
CONTEXT_FEATURES = [
    "engine_load_pct",
    "soil_hardness_index",
    "site_slope_deg",
    "rain_mm",
    "visibility_m",
    "temperature_c",
    "travel_distance_km",
    "soil_moisture_pct",
    "workload_enc",       # label-encoded
    "machine_condition_enc",
    "congestion_enc",
    "task_type_enc",
]

WORKLOAD_ENC     = {"light": 0, "medium": 1, "heavy": 2}
COND_ENC         = {"good": 0, "fair": 1, "poor": 2}
CONGESTION_ENC   = {"low": 0, "medium": 1, "high": 2}
TASK_TYPE_ENC    = {t: i for i, t in enumerate(
                    ["excavation","loading","trenching","grading","hauling","stockpiling"])}

# Thresholds for attribution classification
RESIDUAL_CONTEXT_THRESHOLD  = 0.03   # residual below this → context-driven
RESIDUAL_OPERATOR_THRESHOLD = 0.06   # residual above this → likely operator-driven
MIN_CONFIDENCE_FOR_COACHING = 0.72   # coaching requires this confidence minimum
OPERATOR_FRACTION_THRESHOLD = 0.50   # operator fraction above this → operator-driven


class BehaviorModel:
    """
    Learns expected idle_ratio, cycle_efficiency, fuel_efficiency from
    context features alone (no operator identity). At inference time,
    computes residual and classifies attribution.
    """

    def __init__(self):
        self.idle_model   = Ridge(alpha=1.0)
        self.scaler       = StandardScaler()
        self.fitted       = False
        self.pop_mean_idle  = None
        self.pop_std_idle   = None
        self.residual_std   = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, sessions_df: pd.DataFrame):
        """Train context-only expected-idle model on completed session data."""
        df = sessions_df.copy()
        df = df.dropna(subset=["actual_idle_time_min","actual_task_duration_min"])
        df = df[df["actual_task_duration_min"] > 0]

        df["observed_idle_ratio"] = (
            df["actual_idle_time_min"] / df["actual_task_duration_min"]
        ).clip(0, 1)

        df = self._encode(df)
        X  = df[CONTEXT_FEATURES].fillna(df[CONTEXT_FEATURES].median())
        y  = df["observed_idle_ratio"].values

        X_scaled = self.scaler.fit_transform(X)
        self.idle_model.fit(X_scaled, y)

        # Population stats for Z-score and confidence
        self.pop_mean_idle = float(y.mean())
        self.pop_std_idle  = float(y.std()) + 1e-6

        predicted = self.idle_model.predict(X_scaled)
        residuals = y - predicted
        self.residual_std = float(np.std(residuals)) + 1e-6

        self.fitted = True
        print(f"  BehaviorModel fitted — pop_mean_idle={self.pop_mean_idle:.4f}, "
              f"residual_std={self.residual_std:.4f}, n={len(df)}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        observed_idle_ratio: float,
        context_row: dict,
        operator_baseline_idle: float,
        actual_task_duration_min: float = 60.0,
        actual_fuel_used_l: float | None = None,
        actual_cycle_count: int | None = None,
    ) -> BehaviorResult:
        """
        Core analysis function.

        Parameters
        ----------
        observed_idle_ratio     : actual idle_time / task_duration for this session
        context_row             : dict of context fields (workload, congestion, etc.)
        operator_baseline_idle  : operator's personal baseline idle ratio (from passport)
        actual_task_duration_min: duration for cost estimation
        """
        if not self.fitted:
            return self._insufficient(context_row.get("session_id","unknown"))

        # ── Stage 1: context → expected ───────────────────────────────────────
        row_df = self._encode(pd.DataFrame([context_row]))
        X = row_df[CONTEXT_FEATURES].fillna(self.pop_mean_idle)
        X_scaled = self.scaler.transform(X)
        expected_idle = float(self.idle_model.predict(X_scaled)[0])
        expected_idle = np.clip(expected_idle, 0.02, 0.70)

        # ── Stage 2: observed - expected = residual ───────────────────────────
        # UNIT CONTRACT (enforced): both values are idle RATIOS (dimensionless 0–1).
        # operator_residual      = observed_idle_ratio - expected_idle
        # context_explained      = expected_idle - population_mean_idle
        # Both are in the same unit so the training gate ratio is meaningful:
        #   context_share = |context_explained| / (|operator_residual| + |context_explained|)
        # If context_explained is 0.0 while operator_residual is non-trivial,
        # it means expected == pop_mean (edge case), NOT a missing value.
        operator_residual      = observed_idle_ratio - expected_idle
        context_explained      = expected_idle - self.pop_mean_idle   # context's contribution (ratio)
        operator_residual_abs  = abs(operator_residual)
        z_score                = operator_residual / self.residual_std

        # Guard: warn if context_explained is exactly 0 but operator_residual is significant.
        # This would indicate expected == pop_mean, not a code bug, but worth flagging.
        if context_explained == 0.0 and operator_residual_abs > RESIDUAL_CONTEXT_THRESHOLD:
            import warnings
            warnings.warn(
                f"BehaviorModel: context_explained_component=0.0 while "
                f"operator_residual={operator_residual_abs:.4f}. "
                f"Verify expected_idle ({expected_idle:.4f}) != pop_mean ({self.pop_mean_idle:.4f}).",
                stacklevel=2,
            )

        # Fraction of total deviation that is operator-driven
        total_deviation = abs(observed_idle_ratio - operator_baseline_idle) + 1e-6
        operator_fraction = operator_residual_abs / total_deviation

        # ── Classification ────────────────────────────────────────────────────
        attribution, confidence = self._classify(
            operator_residual_abs, z_score, operator_fraction,
            observed_idle_ratio, expected_idle, operator_baseline_idle,
        )

        # ── Cost estimate ─────────────────────────────────────────────────────
        excess_idle_min = max(0.0, operator_residual) * actual_task_duration_min
        cost_fuel_l     = (excess_idle_min / 60) * 5.0 if actual_fuel_used_l is None \
                           else (actual_fuel_used_l * max(0, operator_residual))

        coaching_eligible = (
            attribution == "operator-driven"
            and confidence >= MIN_CONFIDENCE_FOR_COACHING
            and operator_residual > 0   # only when observed > expected (excess idle)
            # negative residual = operator is performing better than expected — never coach that
        )

        session_id = str(context_row.get("session_id", "unknown"))

        return BehaviorResult(
            session_id=session_id,
            attribution=attribution,
            confidence=round(confidence, 3),
            observed_value=round(observed_idle_ratio, 4),
            expected_value=round(expected_idle, 4),
            operator_residual=round(operator_residual, 4),
            context_explained_component=round(context_explained, 4),
            coaching_eligible=coaching_eligible,
            synthetic_flag=True,
        )

    def analyze_from_session_row(self, session_row: pd.Series, operator_baseline_idle: float) -> BehaviorResult:
        """Convenience wrapper for a DataFrame row from task_sessions.csv."""
        obs_idle = float(session_row["actual_idle_time_min"]) / max(1e-3, float(session_row["actual_task_duration_min"]))
        return self.analyze(
            observed_idle_ratio=obs_idle,
            context_row=session_row.to_dict(),
            operator_baseline_idle=operator_baseline_idle,
            actual_task_duration_min=float(session_row["actual_task_duration_min"]),
            actual_fuel_used_l=float(session_row["actual_fuel_used_l"]),
        )

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  BehaviorModel saved to {path}")

    @staticmethod
    def load(path: str = MODEL_PATH) -> "BehaviorModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["workload_enc"]          = df.get("workload",         pd.Series()).map(WORKLOAD_ENC).fillna(1)
        df["machine_condition_enc"] = df.get("machine_condition",pd.Series()).map(COND_ENC).fillna(0)
        df["congestion_enc"]        = df.get("congestion_level", pd.Series()).map(CONGESTION_ENC).fillna(0)
        df["task_type_enc"]         = df.get("task_type",        pd.Series()).map(TASK_TYPE_ENC).fillna(0)
        for col in CONTEXT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _classify(
        self,
        residual_abs: float,
        z_score: float,
        operator_fraction: float,
        observed: float,
        expected: float,
        baseline: float,
    ):
        abs_z = abs(z_score)

        # Very small residual — insufficient evidence or context explains everything
        if residual_abs < RESIDUAL_CONTEXT_THRESHOLD or abs_z < 0.8:
            if residual_abs < 0.01:
                return "insufficient_evidence", 0.45
            return "context-driven", round(0.65 + (1 - min(1, residual_abs / RESIDUAL_CONTEXT_THRESHOLD)) * 0.15, 3)

        # Large residual and operator fraction high → operator-driven
        if residual_abs >= RESIDUAL_OPERATOR_THRESHOLD and operator_fraction > OPERATOR_FRACTION_THRESHOLD:
            confidence = float(np.clip(0.60 + abs_z * 0.08, 0.60, 0.96))
            return "operator-driven", round(confidence, 3)

        # Moderate residual → mixed
        if residual_abs >= RESIDUAL_CONTEXT_THRESHOLD:
            confidence = float(np.clip(0.50 + abs_z * 0.06, 0.50, 0.85))
            return "mixed", round(confidence, 3)

        return "context-driven", 0.70

    def _insufficient(self, session_id: str) -> BehaviorResult:
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


# ── Training entry point ──────────────────────────────────────────────────────

def train_and_save():
    print("Training BehaviorModel ...")
    sessions  = pd.read_csv(f"{DATA_DIR}/task_sessions.csv")
    model = BehaviorModel()
    model.fit(sessions)
    model.save(MODEL_PATH)
    return model


# ── Quick smoke test ──────────────────────────────────────────────────────────

def smoke_test(model: BehaviorModel):
    print("\nSmoke test — two contrasting sessions:")

    base_ctx = dict(
        session_id="SMOKE_001",
        workload="medium", machine_condition="good", congestion_level="low",
        soil_hardness_index=0.4, site_slope_deg=5.0, rain_mm=0.0,
        visibility_m=2000.0, temperature_c=25.0, travel_distance_km=1.0,
        soil_moisture_pct=30.0, engine_load_pct=65.0, task_type="excavation",
    )

    # Context-driven: high congestion → high idle, but context explains it
    ctx_high_cong = {**base_ctx, "session_id": "SMOKE_001", "congestion_level": "high"}
    result_a = model.analyze(
        observed_idle_ratio=0.22,
        context_row=ctx_high_cong,
        operator_baseline_idle=0.12,
        actual_task_duration_min=60,
    )
    print(f"  [Context-driven] attribution={result_a.attribution}, "
          f"confidence={result_a.confidence}, "
          f"observed={result_a.observed_value}, expected={result_a.expected_value}, "
          f"op_residual={result_a.operator_residual}, coaching={result_a.coaching_eligible}")

    # Operator-driven: same context but much higher idle than expected
    result_b = model.analyze(
        observed_idle_ratio=0.38,
        context_row={**base_ctx, "session_id": "SMOKE_002"},
        operator_baseline_idle=0.12,
        actual_task_duration_min=60,
    )
    print(f"  [Operator-driven] attribution={result_b.attribution}, "
          f"confidence={result_b.confidence}, "
          f"observed={result_b.observed_value}, expected={result_b.expected_value}, "
          f"op_residual={result_b.operator_residual}, coaching={result_b.coaching_eligible}")


if __name__ == "__main__":
    # Import via the package so BehaviorModel is pickled as
    # "features.behavior.behavior_model.BehaviorModel", not "__main__.BehaviorModel".
    # The __main__ path would produce a pickle that fails to load from any other
    # entry point (Streamlit, pytest, python -c).
    from features.behavior.behavior_model import train_and_save as _train, smoke_test as _smoke
    m = _train()
    _smoke(m)
