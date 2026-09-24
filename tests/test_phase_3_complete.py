"""
Phase 3 end-to-end verification for the CAT Operator Copilot demo.

Covers the seven features plus the novel aspects, using the real CSVs in
data/synthetic/ and the real trained models. Nothing under features/, config/
or data/ is modified: anything a feature would write (operator patterns, face
embeddings) goes to pytest's tmp_path.

Runs without Ollama: the LLM client is forced into fallback mode and outbound
non-loopback network access is blocked for the offline test.

Two tests are marked xfail (strict): they document demo-visible gaps that
still exist in develop, so they show up in the summary instead of silently
passing. When one is fixed, its xfail turns into a failure -- remove the marker.

Run: pytest tests/test_phase_3_complete.py -v
"""
import os
import socket
import sys
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # prediction/behavior modules load models via repo-relative paths
warnings.filterwarnings("ignore")

from shared.schemas import AttentionEvent, BehaviorResult, SessionContext  # noqa: E402

DATA = os.path.join(ROOT, "data", "synthetic")


# ── Shared fixtures: real CSVs and models ─────────────────────────────────────

@pytest.fixture(scope="session")
def operators():
    return pd.read_csv(os.path.join(DATA, "operators.csv"))


@pytest.fixture(scope="session")
def machines():
    return pd.read_csv(os.path.join(DATA, "machines.csv"))


@pytest.fixture(scope="session")
def tasks():
    return pd.read_csv(os.path.join(DATA, "tasks.csv"))


@pytest.fixture(scope="session")
def sessions():
    return pd.read_csv(os.path.join(DATA, "task_sessions.csv"))


@pytest.fixture(scope="session")
def safety_events():
    return pd.read_csv(os.path.join(DATA, "safety_events.csv"))


@pytest.fixture(scope="session")
def telemetry():
    return pd.read_csv(os.path.join(DATA, "telemetry.csv"))


@pytest.fixture(scope="session")
def operator_faces():
    return pd.read_csv(os.path.join(DATA, "operator_faces.csv"))


@pytest.fixture(scope="session")
def safety_rules():
    with open(os.path.join(ROOT, "config", "safety_rules.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def behavior_model():
    from features.behavior.behavior_model import BehaviorModel
    return BehaviorModel.load()


@pytest.fixture(scope="session")
def site_data():
    from features.optimization.optimizer import SiteData
    return SiteData.load()


@pytest.fixture(scope="session")
def passport_repo():
    from features.passport.repository import PassportRepository
    return PassportRepository.from_settings()


def behavior_history(model, sessions, operators, operator_id):
    """BehaviorResults for one operator, oldest first, population scoring (no stored patterns)."""
    rows = sessions[sessions.operator_id == operator_id].sort_values("start_timestamp")
    baseline = float(operators.set_index("operator_id").loc[operator_id, "baseline_idle_ratio"])
    return [model.analyze_from_session_row(r.drop(labels=["operator_id"]), baseline)
            for _, r in rows.iterrows()]


@pytest.fixture(autouse=True)
def isolate_operator_patterns(tmp_path, monkeypatch):
    """Operator-pattern files never touch data/; empty store means population scoring."""
    from features.behavior import behavior_model as bm
    monkeypatch.setattr(bm, "OPERATOR_MODEL_DIR", str(tmp_path / "operator_models"))
    monkeypatch.setattr(bm, "_operator_pattern_cache", {})


@pytest.fixture
def offline_llm(monkeypatch):
    """Force the local LLM client into fallback mode regardless of the machine."""
    from features.llm import ollama_client
    from features.safety import safety_guardian
    from features.buddy import buddy
    monkeypatch.setattr(ollama_client, "ollama", None)
    monkeypatch.setattr(safety_guardian, "_llm_client", None)
    monkeypatch.setattr(buddy, "_llm_client", None)


# ══ Part 1: Operator Passport ═════════════════════════════════════════════════

class FakeRecognizer:
    """Deterministic stand-in for DeepFace: a 'frame' is already an embedding vector."""
    model_name = "ArcFace"

    def embed(self, image):
        return list(image)


def _face_vector(index, dim=16, noise=0.0, seed=0):
    v = np.zeros(dim)
    v[index % dim] = 1.0
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, dim)
    return v.tolist()


class TestOperatorPassport:
    FEATURE = "1. Operator Passport"
    COVERAGE = "face login (injected recognizer), authorization, session context"

    def test_biometric_login_flow(self, operator_faces, tmp_path):
        """Registered operators from operator_faces.csv; a close probe authenticates the right one.
        DeepFace/webcam is replaced by an injected recognizer (no face images in the repo)."""
        from features.passport.biometric import (EmbeddingStore, confidence_threshold, identify,
                                                 register_operator)
        store = EmbeddingStore(tmp_path / "faces")
        registered = operator_faces[operator_faces.face_registered].operator_id.tolist()
        assert len(registered) >= 2
        for i, op in enumerate(registered):
            register_operator(op, [_face_vector(i), _face_vector(i, noise=0.05, seed=i)], FakeRecognizer(), store)

        target = registered[1]
        result = identify(_face_vector(1, noise=0.1, seed=99), FakeRecognizer(), store)
        assert result.authenticated and result.operator_id == target
        assert result.confidence >= confidence_threshold()
        assert set(store.registered_operator_ids()) == set(registered)

        stranger = identify(_face_vector(10), FakeRecognizer(), store)   # orthogonal to every registration
        assert not stranger.authenticated and stranger.operator_id is None

    def test_authorization_from_passport_repository(self, passport_repo):
        from features.passport.authorization import (REASON_CERT_STATUS, REASON_MACHINE_TYPE,
                                                     check_authorization)
        as_of = date(2026, 1, 1)
        ok = check_authorization(passport_repo.get_operator("OP1005"), passport_repo.get_machine("EXC001"), as_of)
        assert ok.authorized, ok.reasons                                   # excavator-certified expert

        wrong_type = check_authorization(passport_repo.get_operator("OP1002"),
                                         passport_repo.get_machine("EXC001"), as_of)
        assert not wrong_type.authorized and REASON_MACHINE_TYPE in wrong_type.reasons  # loader-only

        expired = check_authorization(passport_repo.get_operator("OP1008"),
                                      passport_repo.get_machine("EXC001"), as_of)
        assert not expired.authorized and REASON_CERT_STATUS in expired.reasons       # status "expired"

    def test_operator_context_loading(self, passport_repo, operators):
        from features.passport.passport import create_session
        result = create_session("OP1005", "EXC001", passport_repo, task_id="T00001",
                                as_of=datetime(2026, 1, 5, 8, 0))
        assert result.authorized
        ctx = result.session_context
        row = operators.set_index("operator_id").loc["OP1005"]
        assert (ctx.operator_id, ctx.machine_id, ctx.task_id) == ("OP1005", "EXC001", "T00001")
        assert ctx.operator.experience == pytest.approx(row.years_experience)
        assert ctx.operator.machine_skill == row.machine_skill_level
        assert ctx.operator.fatigue_proxy == pytest.approx(row.fatigue_proxy)
        assert ctx.machine.machine_type == "excavator" and ctx.data_quality.synthetic_flag


# ══ Part 2: Safety Guardian ═══════════════════════════════════════════════════

class TestSafetyGuardian:
    FEATURE = "2. Safety Guardian"
    COVERAGE = "CRITICAL/HIGH/MEDIUM/LOW/INFO, YAML thresholds, determinism"

    def test_critical_rules_always_block(self, site_data):
        from features.optimization.optimizer import WeightedScorer
        from features.safety.safety_guardian import context_aware_safety_reasoning, evaluate_record
        cases = {
            "operator_fatigue_critical": {"shift_elapsed_min": 11 * 60},
            "unsafe_ground_slope": {"site_slope_deg": 14.0, "machine_state": "digging"},
            "seatbelt_off_while_moving": {"seatbelt_status": "off", "machine_speed_kmh": 8.0},
            "equipment_malfunction_hydraulic": {"hydraulic_health_score": 0.4},
        }
        stop_actions = {"end_shift_and_rest", "move_to_level_ground", "stop_immediately", "stop_and_inspect"}
        for rule_id, record in cases.items():
            top = evaluate_record({"timestamp": "2026-01-05T10:00:00", **record})[0]
            assert (top.severity, top.rule_id) == ("CRITICAL", rule_id)
            assert top.required_action in stop_actions
            # The reasoning layer can never soften a CRITICAL decision
            enhanced = context_aware_safety_reasoning(top, {"operator_note": "feels fine"})
            assert enhanced["severity"] == "CRITICAL" and not enhanced["decision_changed"]

        # And the optimizer never assigns work into a CRITICAL context: LOC006 slope is 14.9 deg
        scorer = WeightedScorer(site_data)
        task = site_data.tasks[(site_data.tasks.location_id == "LOC006")
                               & site_data.tasks.status.isin(["pending", "active"])].iloc[0]
        op = site_data.operators.iloc[0]
        mc = site_data.machines.iloc[0]
        cand, why = scorer.score(op, mc, task, datetime(2026, 1, 1, 6))
        assert cand is None and why[0] == "safety_block" and "unsafe_ground_slope" in why[1]

    def test_high_and_medium_rules_with_real_context(self, telemetry, machines):
        from features.safety.safety_guardian import evaluate_record
        wet = telemetry[telemetry.rain_mm >= 5.0].iloc[0].to_dict()
        rules = {e.rule_id: e.severity for e in evaluate_record(wet)}
        assert rules.get("weather_heavy_rain") == "HIGH"

        overdue = machines[machines.maintenance_status == "overdue"].iloc[0]
        events = evaluate_record({"machine_id": overdue.machine_id, "maintenance_status": "overdue"})
        assert [(e.rule_id, e.severity) for e in events] == [("maintenance_overdue", "MEDIUM")]
        assert events[0].recommendation  # every rule carries operator-facing guidance

    def test_low_and_info_rules_are_informational(self):
        from features.safety.safety_guardian import evaluate_record
        low = evaluate_record({"congestion_level": "high", "machine_state": "digging"})
        assert [(e.severity, e.required_action) for e in low] == [("LOW", "monitor")]
        info = evaluate_record({"machine_state": "safe_idle", "worker_distance_m": 15.0})
        assert [(e.severity, e.required_action) for e in info] == [("INFO", "none_required")]

    def test_rules_follow_yaml_thresholds(self, safety_rules):
        """Four real rules evaluated exactly at and just inside their YAML thresholds."""
        from features.safety.safety_guardian import evaluate_record
        fat, slope = safety_rules["fatigue"], safety_rules["slope"]
        fuel, cs = safety_rules["fuel"], safety_rules["closing_speed"]

        def fired(rec):
            return {e.rule_id for e in evaluate_record(rec)}

        assert "operator_fatigue_critical" in fired({"shift_elapsed_min": fat["critical_hours"] * 60 + 1})
        assert "operator_fatigue_critical" not in fired({"shift_elapsed_min": fat["critical_hours"] * 60})
        assert "unsafe_ground_slope" in fired({"site_slope_deg": slope["critical_deg"], "machine_state": "grading"})
        assert "unsafe_ground_slope" not in fired({"site_slope_deg": slope["critical_deg"] - 0.1,
                                                   "machine_state": "grading"})
        assert "low_fuel_reserve" in fired({"fuel_level_pct": fuel["low_reserve_pct"] - 0.1})
        assert "low_fuel_reserve" not in fired({"fuel_level_pct": fuel["low_reserve_pct"]})
        env = {"machine_state": "swinging", "worker_distance_m": 8.0}
        assert "worker_in_swing_envelope_with_closing_motion_critical" in fired({**env, "closing_speed_mps": cs["critical_mps"]})
        assert "worker_in_swing_envelope_with_closing_motion_critical" not in fired({**env, "closing_speed_mps": cs["critical_mps"] - 0.01})

    def test_decisions_are_deterministic(self, telemetry, safety_events):
        from features.safety.safety_guardian import evaluate_records, validate_against_logged
        sample = telemetry.sample(300, random_state=1).to_dict("records")
        runs = [[[(e.event_id, e.severity, e.rule_id) for e in evs] for evs in evaluate_records(sample)]
                for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]
        v = validate_against_logged()                    # never less severe than the logged labels
        order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for logged in v["confusion"].index:
            for predicted in v["confusion"].columns:
                if predicted in order and order.index(predicted) < order.index(logged):
                    assert v["confusion"].loc[logged, predicted] == 0


# ══ Part 3: Behavioral Fingerprint ════════════════════════════════════════════

class TestBehavioralFingerprint:
    FEATURE = "3. Behavioral Fingerprint"
    COVERAGE = "residual math, attribution classes, coaching flag, 2-operator contrast"

    def test_residual_calculation(self, behavior_model, sessions):
        row = sessions.iloc[0]
        r = behavior_model.analyze_from_session_row(row.drop(labels=["operator_id"]), 0.12)
        observed = row.actual_idle_time_min / row.actual_task_duration_min
        assert r.observed_value == pytest.approx(observed, abs=1e-4)
        assert r.operator_residual == pytest.approx(r.observed_value - r.expected_value, abs=2e-4)
        assert r.context_explained_component == pytest.approx(
            r.expected_value - behavior_model.pop_mean_idle, abs=2e-4)

    def test_operator_vs_context_driven_attribution(self, behavior_model, sessions):
        sample = sessions.sample(400, random_state=7)
        results = [behavior_model.analyze_from_session_row(r.drop(labels=["operator_id"]), 0.12)
                   for _, r in sample.iterrows()]
        by_attr = pd.Series([r.attribution for r in results]).value_counts()
        assert by_attr.get("operator-driven", 0) > 0 and by_attr.get("context-driven", 0) > 0
        from features.behavior.behavior_model import RESIDUAL_CONTEXT_THRESHOLD, RESIDUAL_OPERATOR_THRESHOLD
        for r in results:
            if r.attribution == "operator-driven":
                assert abs(r.operator_residual) >= RESIDUAL_OPERATOR_THRESHOLD
            if r.attribution == "context-driven":
                assert abs(r.operator_residual) < RESIDUAL_OPERATOR_THRESHOLD

    def test_coaching_eligible_flag(self, behavior_model, sessions):
        """Per-session flag = OPERATOR_LINKED + CONFIDENT + unfavourable; REPEATED is the training gate's job."""
        from features.behavior.behavior_model import MIN_CONFIDENCE_FOR_COACHING
        sample = sessions.sample(400, random_state=11)
        results = [behavior_model.analyze_from_session_row(r.drop(labels=["operator_id"]), 0.12)
                   for _, r in sample.iterrows()]
        eligible = [r for r in results if r.coaching_eligible]
        assert eligible and len(eligible) < len(results)
        for r in eligible:
            assert r.attribution == "operator-driven"
            assert r.confidence >= MIN_CONFIDENCE_FOR_COACHING
            assert r.operator_residual > 0
        assert not any(r.coaching_eligible for r in results if r.operator_residual <= 0)

    def test_two_operators_with_different_idle_patterns(self, behavior_model, sessions, operators):
        ranked = operators.sort_values("baseline_idle_ratio")
        low_op, high_op = ranked.operator_id.iloc[0], ranked.operator_id.iloc[-1]   # OP1003 vs OP1027
        low = behavior_history(behavior_model, sessions, operators, low_op)
        high = behavior_history(behavior_model, sessions, operators, high_op)
        mean = lambda rs, f: float(np.mean([getattr(r, f) for r in rs]))
        assert mean(high, "observed_value") > mean(low, "observed_value") + 0.1
        assert mean(high, "operator_residual") > 0 > mean(low, "operator_residual")
        assert sum(r.coaching_eligible for r in high) > sum(r.coaching_eligible for r in low)


# ══ Predictive Task Intelligence (feature 4) ══════════════════════════════════

class TestPredictiveTaskIntelligence:
    FEATURE = "4. Predictive Task Intelligence"
    COVERAGE = "ETA + fuel P10/P50/P90 from real contexts, dashboard resolves it"

    def test_eta_and_fuel_predictions(self, passport_repo, sessions):
        from features.dashboard import adapters
        from features.prediction.api import predict_combined
        from shared.schemas import SiteContext, TaskContext, WeatherContext
        row = sessions.iloc[0]
        ctx = SessionContext(
            session_id=row.session_id, operator_id=row.operator_id, machine_id=row.machine_id,
            task=TaskContext(task_id=row.task_id, task_type=row.task_type, task_phase=row.task_phase,
                             workload=row.workload),
            site=SiteContext(hardness=row.soil_hardness_index, slope=row.site_slope_deg,
                             congestion=row.congestion_level),
            weather=WeatherContext(rain_mm=row.rain_mm, visibility_m=row.visibility_m,
                                   temperature_c=row.temperature_c, day_night=row.day_night))
        p = predict_combined(ctx)
        assert 0 < p.eta_p10 <= p.eta_p50 <= p.eta_p90
        assert 0 < p.fuel_p10 <= p.fuel_p50 <= p.fuel_p90
        assert 0.4 <= p.confidence <= 0.96 and p.factors
        heavy = predict_combined(SessionContext(**{**ctx.__dict__, "task": TaskContext(
            task_id=row.task_id, task_type=row.task_type, task_phase=row.task_phase, workload="heavy")}))
        light = predict_combined(SessionContext(**{**ctx.__dict__, "task": TaskContext(
            task_id=row.task_id, task_type=row.task_type, task_phase=row.task_phase, workload="light")}))
        assert (heavy.eta_p50, heavy.fuel_p50) != (light.eta_p50, light.fuel_p50)
        assert adapters.resolve("prediction", "predict_combined") is not None
        assert adapters.resolve("behavior", "analyze_behavior") is not None


# ══ Part 4: Attention Manager ═════════════════════════════════════════════════

def _event(eid, etype, severity, state="swinging", action="next_safe_state", reason=None,
           ts="2026-01-05T10:00:00", confidence=0.9):
    return AttentionEvent(eid, etype, severity, "x", action, confidence, state, "", reason or eid, ts)


class TestAttentionManager:
    FEATURE = "5. Attention Manager"
    COVERAGE = "tier ordering, CRITICAL preemption, safe-state release, routing"

    def test_priority_ordering_across_tiers(self):
        from features.attention.attention_manager import AttentionManager
        am = AttentionManager()
        events = [_event("buddy", "buddy", "INFO"), _event("train", "training", "MEDIUM"),
                  _event("nudge", "behavior_nudge", "LOW"), _event("replan", "task_replanning", "MEDIUM"),
                  _event("high", "safety", "HIGH"), _event("crit", "safety", "CRITICAL")]
        assert [e.event_id for e in am.rank(events)] == ["crit", "high", "replan", "nudge", "train", "buddy"]

    def test_critical_safety_always_wins(self):
        from features.attention.attention_manager import AttentionManager
        am = AttentionManager()
        for i in range(am.cfg["max_queue_size"]):
            am.submit(_event(f"t{i}", "training", "LOW", reason=f"r{i}"))
        r = am.submit(_event("crit", "safety", "CRITICAL", action="act_now", confidence=0.3))
        assert r["decision"] == "show_now" and r["route"] == "safety_guardian"
        # Even a duplicate CRITICAL inside the bundle window is shown, never bundled
        assert am.submit(_event("crit2", "safety", "CRITICAL", action="act_now", reason="crit"))["decision"] == "show_now"

    def test_non_urgent_events_wait_for_safe_state_and_route(self):
        from features.attention.attention_manager import AttentionManager
        am = AttentionManager()
        decisions = {e.event_id: am.submit(e) for e in [
            _event("train", "training", "LOW"), _event("replan", "task_replanning", "MEDIUM"),
            _event("buddy", "buddy", "INFO")]}
        assert decisions["train"]["decision"] == "queue" and decisions["replan"]["decision"] == "queue"
        assert decisions["buddy"]["decision"] == "suppress"            # Buddy gated to safe states
        released = am.release("parked")
        assert released[0]["event_id"] == "replan" and released[0]["route"] == "dashboard"
        assert am.release("parked")[0]["route"] == "training_hub"

    @pytest.mark.xfail(strict=True, reason="KNOWN ISSUE: dashboard candidates use operator_state "
                       "'active'/'available' and actionability 'actionable_now'; Attention Manager "
                       "expects machine states and 'act_now', so HIGH safety is queued while working")
    def test_dashboard_candidates_route_correctly(self):
        from features.attention.attention_manager import AttentionManager
        from features.dashboard.attention_candidates import safety_candidate
        ev = safety_candidate({"event_id": "SE1", "severity": "HIGH", "machine_state": "swinging",
                               "confidence": 0.9, "timestamp": "2026-01-05T10:00:00"})
        assert AttentionManager().submit(ev)["decision"] == "show_now"


# ══ Part 5: Task Optimization ═════════════════════════════════════════════════

class TestTaskOptimization:
    FEATURE = "6. Task Optimization"
    COVERAGE = "skill bonus, 5-part cost, expert vs novice, hard constraints"

    def test_skill_bonus_score(self, site_data):
        from features.optimization.optimizer import WeightedScorer
        s = WeightedScorer(site_data)
        assert s.skill_bonus_score({"required_skill_level": "advanced"}, {"task_skill_level": "expert"}) == 0.8
        assert s.skill_bonus_score({"required_skill_level": "advanced"}, {"task_skill_level": "advanced"}) == 1.0
        assert s.skill_bonus_score({"required_skill_level": "advanced"}, {"task_skill_level": None}) == 1.0

    def test_multi_objective_scoring_and_hard_constraints(self, site_data):
        from features.optimization.optimizer import SKILL_RANK, rank_assignments
        result = rank_assignments(data=site_data, pool_size=15)
        assert result["assignments"]
        ops = site_data.operators.set_index("operator_id")
        mcs = site_data.machines.set_index("machine_id")
        tks = site_data.tasks.set_index("task_id")
        assert len({a["operator_id"] for a in result["assignments"]}) == len(result["assignments"])
        for a in result["assignments"]:
            parts = a["cost_breakdown"]
            assert set(parts) == {"time_cost", "fuel_cost", "deadline_risk", "transition_cost",
                                  "safety_condition_risk"}
            assert sum(parts.values()) == pytest.approx(a["total_cost"], abs=0.05)
            assert a["score"] <= a["total_cost"]                   # priority factor >= 1
            assert a["reason"] and a["eta_min"]["p10"] <= a["eta_min"]["p50"] <= a["eta_min"]["p90"]
            op, mc, t = ops.loc[a["operator_id"]], mcs.loc[a["machine_id"]], tks.loc[a["task_id"]]
            assert mc.machine_type == t.required_machine_type
            assert mc.machine_type in op.authorized_machine_types.split(",")
            assert SKILL_RANK[op.machine_skill_level] >= SKILL_RANK[t.required_skill_level]
            assert op.certification_status == "active"

    def test_expert_preferred_over_novice_for_specialised_task(self, site_data, tasks, operators):
        from features.optimization.optimizer import WeightedScorer, _infeasible_reason
        scorer = WeightedScorer(site_data)
        when = datetime(2026, 1, 1, 6)
        task = site_data.tasks[(site_data.tasks.required_skill_level == "advanced")
                               & (site_data.tasks.required_machine_type == "excavator")
                               & site_data.tasks.status.isin(["pending", "active"])
                               & site_data.tasks.dependency_task_id.isna()
                               & ~site_data.tasks.location_id.isin(["LOC006", "LOC008"])].iloc[0]
        mc = site_data.machines[site_data.machines.machine_id == "EXC001"].iloc[0]
        ops = site_data.operators.set_index("operator_id", drop=False)
        expert = ops.loc["OP1005"]                                  # expert, excavator
        novice = ops[(ops.task_skill_level == "novice")].iloc[0]
        advanced = ops[(ops.task_skill_level == "advanced") & ops.authorized_machine_types.str.contains("excavator")
                       & (ops.certification_status == "active")].iloc[0]

        # Novice: hard-excluded from specialised work
        why = _infeasible_reason(novice, mc, task, set(), when)
        assert why is not None and why[0] in {"skill_below_required", "operator_not_authorised"}

        exp_c, _ = scorer.score(expert, mc, task, when)
        adv_c, _ = scorer.score(advanced, mc, task, when)
        assert exp_c is not None and adv_c is not None
        assert any("skill bonus" in r for r in exp_c.reasoning)
        assert not any("skill bonus" in r for r in adv_c.reasoning)
        assert exp_c.breakdown["time_cost"] == pytest.approx(round(exp_c.eta[1] * 0.8, 2), abs=0.02)
        assert adv_c.breakdown["time_cost"] == pytest.approx(round(adv_c.eta[1], 2), abs=0.02)

# ══ Part 6: Training Hub ══════════════════════════════════════════════════════

def _br(residual, context, confidence=0.85, attribution="operator-driven", sid="S"):
    return BehaviorResult(sid, attribution, confidence, 0.2, 0.2 - residual, residual, context,
                          attribution == "operator-driven" and residual > 0, True)


class TestTrainingHub:
    FEATURE = "Training Hub"
    COVERAGE = "3-part gate, favourable residual, escalation ladder, real history"

    def test_gate_requires_repeated_confident_operator_linked(self):
        from features.training.trigger import (REASON_CONTEXT_EXPLAINED, REASON_LOW_CONFIDENCE,
                                               REASON_NOT_REPEATED, evaluate_training_gate)
        good = _br(0.10, 0.02)
        assert evaluate_training_gate([good, good], "OP1", "idle_reduction").triggered          # all three hold
        once = evaluate_training_gate([good], "OP1", "idle_reduction")
        assert not once.triggered and REASON_NOT_REPEATED in once.reasons                       # not REPEATED
        unsure = evaluate_training_gate([_br(0.10, 0.02, confidence=0.69)] * 2, "OP1", "idle_reduction")
        assert not unsure.triggered and REASON_LOW_CONFIDENCE in unsure.reasons                 # not CONFIDENT
        context = evaluate_training_gate([_br(0.04, 0.05)] * 2, "OP1", "idle_reduction")        # share 0.56
        assert not context.triggered and REASON_CONTEXT_EXPLAINED in context.reasons            # not OPERATOR_LINKED
        edge = evaluate_training_gate([_br(0.05, 0.05)] * 2, "OP1", "idle_reduction")           # share exactly 0.50
        assert edge.triggered

    def test_favourable_residual_never_triggers(self, behavior_model, sessions, operators):
        from features.training.trigger import REASON_RESIDUAL_FAVOURABLE, evaluate_training_gate
        synthetic = evaluate_training_gate([_br(-0.12, 0.01, confidence=0.95)] * 5, "OP1", "idle_reduction")
        assert not synthetic.triggered and REASON_RESIDUAL_FAVOURABLE in synthetic.reasons
        # Real data: OP1003 idles less than expected and is never coached
        history = behavior_history(behavior_model, sessions, operators, "OP1003")
        real = evaluate_training_gate(history, "OP1003", "idle_reduction")
        assert not real.triggered and REASON_RESIDUAL_FAVOURABLE in real.reasons

    def test_escalation_ladder(self):
        from features.training.progress import next_escalation_state
        from features.training.trigger import TrainingThresholds, escalation_state_for
        th = TrainingThresholds.from_settings()
        assert [escalation_state_for(n, th) for n in (0, 1, 2, 5)] == \
               ["first_trigger", "repeat", "escalated", "escalated"]
        assert next_escalation_state("first_trigger", False) == "repeat"
        assert next_escalation_state("repeat", False) == "escalated"
        assert next_escalation_state("escalated", False) == "escalated"
        assert next_escalation_state("escalated", True) == "resolved"
        assert next_escalation_state("resolved", False) == "resolved"

    def test_gate_on_real_behavior_history(self, behavior_model, sessions, operators):
        from features.training.trigger import check_training_trigger, evaluate_training_gate
        history = behavior_history(behavior_model, sessions, operators, "OP1027")   # habitual high idle
        decision = evaluate_training_gate(history, "OP1027", "idle_reduction")
        assert decision.triggered and decision.checks["gate_met"]
        assert decision.qualifying_occurrences >= decision.thresholds.min_occurrences
        trigger = check_training_trigger(history, "OP1027", "idle_reduction", prior_trigger_count=1)
        assert trigger.escalation_state == "repeat" and trigger.lesson_id and trigger.confidence >= 0.70


# ══ Part 7: Grounded Buddy ════════════════════════════════════════════════════

class TestGroundedBuddy:
    FEATURE = "7. Grounded AI Buddy"
    COVERAGE = "safe-state gate, retrieval, safety override, conflicts, 3 questions"

    SETTINGS = {"buddy": {"no_attachment_movement_required": True, "max_evidence_age_min": 60,
                          "llm_synthesis": False}}

    @staticmethod
    def _snapshot(row):
        from features.buddy.safe_state import MachineStateSnapshot
        return MachineStateSnapshot(machine_state=row["machine_state"],
                                    attachment_movement=row["attachment_movement"],
                                    arm_speed=row["arm_speed"], bucket_state=row["bucket_state"],
                                    machine_speed_kmh=row["machine_speed_kmh"])

    def test_safe_state_gating_on_real_telemetry(self, telemetry):
        from features.buddy.safe_state import evaluate_safe_state
        idle = telemetry[telemetry.machine_state == "safe_idle"].iloc[0]
        digging = telemetry[(telemetry.machine_state == "digging") & (telemetry.arm_speed > 0)].iloc[0]
        traveling = telemetry[telemetry.machine_state == "traveling"].iloc[0]
        assert evaluate_safe_state(self._snapshot(idle)).allowed
        assert not evaluate_safe_state(self._snapshot(digging)).allowed
        assert not evaluate_safe_state(self._snapshot(traveling)).allowed

    def test_safety_critical_question_uses_authoritative_sources_only(self, safety_events, telemetry):
        """Q1 (safety-critical): answered only from safety_events.csv + approved manual."""
        from features.buddy.buddy import STATUS_DEFERRED_SAFETY_CRITICAL, ask
        from features.buddy.buddy import AUTHORITATIVE_SAFETY_SOURCES
        from features.buddy.retrieval import retrieve
        from features.buddy.safe_state import MachineStateSnapshot
        event = safety_events[safety_events.severity == "CRITICAL"].iloc[0].to_dict()
        tel = telemetry.iloc[0].to_dict()
        as_of = datetime.fromisoformat(event["timestamp"]) + timedelta(minutes=5)
        parked = MachineStateSnapshot(machine_state="parked", attachment_movement="none")
        question = "Is it safe to swing with a worker nearby?"

        evidence = retrieve(question, telemetry_row=tel, safety_events=[event])
        assert any(e.source not in AUTHORITATIVE_SAFETY_SOURCES for e in evidence)   # telemetry retrieved too
        r = ask(question, parked, evidence, as_of=as_of, settings=self.SETTINGS)
        assert r.answered and all(e.source in AUTHORITATIVE_SAFETY_SOURCES for e in r.evidence)
        assert "telemetry" not in r.answer

        no_authority = ask(question, parked, retrieve(question, telemetry_row=tel, manual=()),
                           as_of=as_of, settings=self.SETTINGS)
        assert not no_authority.answered and no_authority.status == STATUS_DEFERRED_SAFETY_CRITICAL
        assert no_authority.deferral_target == "safety_guardian"

    def test_non_critical_question_answered_from_telemetry(self, telemetry):
        """Q2 (non-critical): fuel level from a real telemetry row, cited and quoted."""
        from features.buddy.buddy import ask
        from features.buddy.retrieval import retrieve
        from features.buddy.safe_state import MachineStateSnapshot
        row = telemetry.iloc[100].to_dict()
        question = "What is my fuel level?"
        evidence = retrieve(question, telemetry_row=row, manual=())
        r = ask(question, MachineStateSnapshot(machine_state="parked", attachment_movement="none"),
                evidence, as_of=datetime.fromisoformat(row["timestamp"]) + timedelta(minutes=1),
                settings=self.SETTINGS)
        assert r.answered and str(row["fuel_level_pct"]) in r.answer and "source: telemetry" in r.answer
        assert r.answer_mode == "quoted"

    def test_conflicting_evidence_detected_and_resolved(self, telemetry, machines):
        """Q3 (conflict): real fuel readings that disagree -> authority wins, or defer on a tie."""
        from features.buddy.buddy import STATUS_DEFERRED_CONFLICT, ask
        from features.buddy.evidence import SOURCE_PASSPORT, SOURCE_TELEMETRY, Evidence
        from features.buddy.safe_state import MachineStateSnapshot
        parked = MachineStateSnapshot(machine_state="parked", attachment_movement="none")
        question = "How much fuel is in EXC001?"

        tel = telemetry[telemetry.machine_id == "EXC001"].iloc[0]
        registry = machines.set_index("machine_id").loc["EXC001", "current_fuel_level_pct"]
        assert abs(tel.fuel_level_pct - registry) > 1                 # real disagreement
        as_of = datetime.fromisoformat(tel.timestamp) + timedelta(minutes=1)
        items = [Evidence(SOURCE_TELEMETRY, tel.fuel_level_pct, f"Fuel level is {tel.fuel_level_pct} percent.", tel.timestamp),
                 Evidence(SOURCE_PASSPORT, registry, f"Machine record lists fuel at {registry} percent.", tel.timestamp)]
        r = ask(question, parked, items, as_of=as_of, settings=self.SETTINGS)
        assert r.answered and r.conflict.conflicted and r.conflict.winner.source == SOURCE_TELEMETRY
        assert "disagree" in r.answer.lower()

        # Two telemetry readings (same authority) minutes apart that disagree -> defer
        t = telemetry[telemetry.session_id == "S000457"].sort_values("timestamp")
        t = t.assign(prev=t.fuel_level_pct.shift()).dropna(subset=["prev"])
        pair = t[(t.fuel_level_pct - t.prev).abs() > 0.02 * t.fuel_level_pct].iloc[0]
        tie = [Evidence(SOURCE_TELEMETRY, pair.prev, f"Fuel level is {pair.prev} percent.", pair.timestamp),
               Evidence(SOURCE_TELEMETRY, pair.fuel_level_pct, f"Fuel level is {pair.fuel_level_pct} percent.", pair.timestamp)]
        d = ask(question, parked, tie, as_of=datetime.fromisoformat(pair.timestamp) + timedelta(minutes=1),
                settings=self.SETTINGS)
        assert not d.answered and d.status == STATUS_DEFERRED_CONFLICT


# ══ Novel aspects ═════════════════════════════════════════════════════════════

class TestNovelAspects:
    FEATURE = "Novel aspects"
    COVERAGE = "operator patterns, context attribution, S->B->T loop, offline-first"

    def test_operator_specific_patterns_differ(self, behavior_model, sessions):
        from features.behavior import behavior_model as bm
        high = behavior_model.learn_operator_specific_patterns("OP1027", sessions)
        low = behavior_model.learn_operator_specific_patterns("OP1003", sessions)
        assert high and low
        assert high["residual_mean"] > 0.05 > -0.05 > low["residual_mean"] or \
               high["residual_mean"] - low["residual_mean"] > 0.1
        assert os.path.exists(os.path.join(bm.OPERATOR_MODEL_DIR, "OP1027.pkl"))  # tmp dir, not data/
        assert not os.path.exists(os.path.join(DATA, "operator_models", "OP1027.pkl"))
        # Context-only fields are unchanged by a learned pattern
        row = sessions[sessions.operator_id == "OP1027"].iloc[0]
        with_p = behavior_model.analyze_from_session_row(row, 0.12)
        without = behavior_model.analyze_from_session_row(row.drop(labels=["operator_id"]), 0.12)
        assert (with_p.expected_value, with_p.operator_residual) == (without.expected_value, without.operator_residual)

    def test_same_behavior_different_context_changes_attribution(self, behavior_model, sessions):
        """Identical observed idle is operator-driven on easy ground but not on difficult ground."""
        base = sessions.iloc[0].to_dict()
        easy = {**base, "congestion_level": "low", "workload": "light", "machine_condition": "good",
                "rain_mm": 0.0, "site_slope_deg": 1.0, "soil_hardness_index": 0.2}
        hard = {**base, "congestion_level": "high", "workload": "heavy", "machine_condition": "poor",
                "rain_mm": 12.0, "site_slope_deg": 14.0, "soil_hardness_index": 0.95}
        e_easy = behavior_model.analyze(0.0, easy, 0.12)
        e_hard = behavior_model.analyze(0.0, hard, 0.12)
        assert e_hard.expected_value > e_easy.expected_value           # context raises expectation
        observed = (e_easy.expected_value + e_hard.expected_value) / 2 + 0.05
        r_easy = behavior_model.analyze(observed, easy, 0.12)
        r_hard = behavior_model.analyze(observed, hard, 0.12)
        assert r_easy.operator_residual > r_hard.operator_residual
        assert r_easy.attribution == "operator-driven" and r_easy.coaching_eligible
        assert r_hard.attribution != "operator-driven" and not r_hard.coaching_eligible

    def test_safety_behavior_training_loop(self, behavior_model, sessions, operators, safety_events):
        """One real safety event -> its session's behavior -> operator history -> training -> attention."""
        from features.attention.attention_manager import AttentionManager, from_safety_event, from_training_trigger
        from features.safety.safety_guardian import evaluate_record
        from features.training.trigger import evaluate_training_gate

        op = "OP1027"
        event = safety_events[(safety_events.operator_id == op)
                              & safety_events.severity.isin(["HIGH", "CRITICAL"])].iloc[0]
        # 1. Safety: re-derive the event with the rule engine
        top = evaluate_record(event.to_dict())[0]
        assert top.severity in ("HIGH", "CRITICAL")
        # 2. Behavior: analyze the session the event occurred in
        row = sessions[sessions.session_id == event.session_id].iloc[0]
        baseline = float(operators.set_index("operator_id").loc[op, "baseline_idle_ratio"])
        this_session = behavior_model.analyze_from_session_row(row.drop(labels=["operator_id"]), baseline)
        assert this_session.attribution in ("operator-driven", "mixed")
        # 3. Training: the operator's history up to and including that session
        history = behavior_history(behavior_model, sessions[sessions.start_timestamp <= row.start_timestamp],
                                   operators, op)
        decision = evaluate_training_gate(history, op, "idle_reduction")
        assert decision.triggered
        # 4. Attention: safety preempts; training waits for a safe state
        am = AttentionManager()
        routed = am.submit_many([from_safety_event(top, "swinging"),
                                 from_training_trigger(decision.trigger, "swinging", event.timestamp)])
        assert routed[0]["tier"] in ("critical_safety", "high_safety")
        assert routed[1]["decision"] == "queue" and routed[1]["route"] == "training_hub"
        assert any(r["route"] == "training_hub" for r in am.release("parked") + am.release("parked"))

    def test_offline_first_no_external_calls(self, offline_llm, monkeypatch):
        from features.buddy.buddy import ANSWER_MODE_QUOTED, synthesize_answer_with_llm
        from features.buddy.evidence import SOURCE_TASK_PLAN, SOURCE_TELEMETRY, Evidence
        from features.llm.ollama_client import OllamaClient
        from features.safety.safety_guardian import context_aware_safety_reasoning, evaluate_record

        attempts = []
        real_connect = socket.socket.connect

        def guarded_connect(self, address):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "localhost", "::1"):
                attempts.append(host)
                raise AssertionError(f"external network call to {host}")
            return real_connect(self, address)

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        client = OllamaClient()
        assert not client.is_available(force=True) and client.fallback_mode
        with pytest.raises(ValueError):
            OllamaClient(host="http://api.example.com:11434")

        high = evaluate_record({"timestamp": "2026-01-05T10:00:00", "rain_mm": 8.0})[0]
        enhanced = context_aware_safety_reasoning(high, {"surface_type": "wet"})
        assert enhanced["reasoning_source"] == "rules" and enhanced["severity"] == "HIGH"

        items = [Evidence(SOURCE_TELEMETRY, 60.0, "Fuel level is 60 percent."),
                 Evidence(SOURCE_TASK_PLAN, 60.0, "Plan assumes 60 percent fuel.")]
        answer, mode = synthesize_answer_with_llm(items, "fuel?")
        assert mode == ANSWER_MODE_QUOTED and "[sources: telemetry, task_plan]" in answer
        assert attempts == []


# ══ Demo-readiness: known integration gap ═════════════════════════════════════

class TestDemoIntegration:
    FEATURE = "Dashboard integration"
    COVERAGE = "dashboard adapters, Buddy conflict wording (demo-visible gaps)"

    @pytest.mark.xfail(strict=True, reason="KNOWN ISSUE: Buddy conflict check compares Evidence.value "
                       "across kinds (incident severity 'CRITICAL' vs manual snippet id 'M001'), so every "
                       "safety answer with incident + manual evidence starts 'Sources disagree'")
    def test_safety_answer_not_flagged_as_conflict(self, safety_events):
        from features.buddy.buddy import ask
        from features.buddy.retrieval import retrieve
        from features.buddy.safe_state import MachineStateSnapshot
        event = safety_events[safety_events.severity == "CRITICAL"].iloc[0].to_dict()
        question = "Is it safe to swing with a worker nearby?"
        r = ask(question, MachineStateSnapshot(machine_state="parked", attachment_movement="none"),
                retrieve(question, safety_events=[event]),
                as_of=datetime.fromisoformat(event["timestamp"]) + timedelta(minutes=5),
                settings=TestGroundedBuddy.SETTINGS)
        assert r.answered and not r.conflict.conflicted and "disagree" not in r.answer.lower()

    def test_safety_optimization_attention_resolve(self):
        from features.dashboard import adapters
        assert adapters.resolve("safety", "evaluate_safety") is not None
        assert adapters.resolve("optimization", "generate_plan") is not None
        assert adapters.resolve("attention", "route_event") is not None
