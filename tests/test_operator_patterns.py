import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # behavior model uses repo-relative paths
warnings.filterwarnings("ignore")

from features.behavior import behavior_model as bm


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "OPERATOR_MODEL_DIR", str(tmp_path / "operator_models"))
    monkeypatch.setattr(bm, "_operator_pattern_cache", {})
    return bm.BehaviorModel.load()


@pytest.fixture(scope="module")
def sessions():
    return pd.read_csv(os.path.join(ROOT, "data", "synthetic", "task_sessions.csv"))


def test_pattern_saved_per_operator(model, sessions):
    p = model.learn_operator_specific_patterns("OP1001", sessions)
    assert os.path.exists(os.path.join(bm.OPERATOR_MODEL_DIR, "OP1001.pkl"))
    assert p["n_sessions"] == (sessions.operator_id == "OP1001").sum()
    assert type(p["model"]).__name__ == "LGBMRegressor"
    assert bm.load_operator_pattern("OP1001")["n_sessions"] == p["n_sessions"]


def test_too_few_sessions_saves_nothing(model, sessions):
    few = sessions[sessions.operator_id == "OP1001"].head(bm.MIN_SESSIONS_FOR_OPERATOR_MODEL - 1)
    assert model.learn_operator_specific_patterns("OP1001", few) is None
    assert not os.path.exists(os.path.join(bm.OPERATOR_MODEL_DIR, "OP1001.pkl"))


def test_no_pattern_means_unchanged_analysis(model, sessions):
    row = sessions[sessions.operator_id == "OP1002"].iloc[0]
    with_id = model.analyze_from_session_row(row, 0.12)
    without_id = model.analyze_from_session_row(row.drop(labels=["operator_id"]), 0.12)
    assert with_id == without_id


def test_context_only_fields_never_change(model, sessions):
    """Stage 1 and the residual split stay context-only (D040 / D025 depend on them)."""
    rows = sessions[sessions.operator_id == "OP1001"]
    before = [model.analyze_from_session_row(r, 0.12) for _, r in rows.head(20).iterrows()]
    model.learn_operator_specific_patterns("OP1001", rows)
    after = [model.analyze_from_session_row(r, 0.12) for _, r in rows.head(20).iterrows()]
    for b, a in zip(before, after):
        assert (b.expected_value, b.operator_residual, b.context_explained_component, b.observed_value) == \
               (a.expected_value, a.operator_residual, a.context_explained_component, a.observed_value)


def test_confidence_uses_operator_variance(model, sessions):
    """A consistent operator (small spread) gets a larger |z| and so at least as much confidence."""
    rows = sessions[sessions.operator_id == "OP1027"]
    p = model.learn_operator_specific_patterns("OP1027", rows)
    assert p["shrunk_residual_std"] < model.residual_std
    row = rows.iloc[0]
    pop = model.analyze_from_session_row(row.drop(labels=["operator_id"]), 0.12)
    own = model.analyze_from_session_row(row, 0.12)
    assert own.confidence >= pop.confidence


def test_shrinkage_toward_population(model, sessions):
    rows = sessions[sessions.operator_id == "OP1010"]
    p = model.learn_operator_specific_patterns("OP1010", rows)
    lo, hi = sorted([p["residual_std"], p["population_residual_std"]])
    assert lo <= p["shrunk_residual_std"] <= hi + 1e-6
    assert 0 < p["weight"] < 1


def test_personal_baseline_is_context_aware(model, sessions):
    rows = sessions[sessions.operator_id == "OP1001"]
    model.learn_operator_specific_patterns("OP1001", rows)
    X = model._encode(rows.head(30))[bm.CONTEXT_FEATURES]
    baselines = {round(model._apply_operator_pattern("OP1001", X.iloc[[i]], 0.15, 0.12)[0], 4)
                 for i in range(len(X))}
    assert len(baselines) > 1 and all(0.02 <= b <= 0.70 for b in baselines)
