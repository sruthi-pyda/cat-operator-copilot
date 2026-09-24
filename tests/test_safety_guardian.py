import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.safety import safety_guardian as sg
from shared.schemas import SessionContext, MachineContext, TrafficContext, TemporalContext


def rules(record):
    return [e.rule_id for e in sg.evaluate_record(record)]


def test_yaml_loads_and_refs_resolve():
    cfg = sg.load_rules()
    assert len(cfg["rules"]) > 20
    assert {r["severity"] for r in cfg["rules"]} <= set(cfg["severity_order"])


@pytest.mark.parametrize("minutes, expected", [
    (400, None), (481, "operator_fatigue_high"), (600, "operator_fatigue_high"), (601, "operator_fatigue_critical"),
])
def test_fatigue_thresholds(minutes, expected):
    fired = rules({"shift_elapsed_min": minutes})
    fatigue = [r for r in fired if r.startswith("operator_fatigue")]
    assert fatigue == ([expected] if expected else [])


def test_unsafe_slope_only_when_active():
    assert "unsafe_ground_slope" in rules({"site_slope_deg": 13, "machine_state": "digging"})
    assert "unsafe_ground_slope" not in rules({"site_slope_deg": 13, "machine_state": "parked"})


def test_equipment_malfunction_is_critical():
    events = sg.evaluate_record({"hydraulic_health_score": 0.5, "engine_health_score": 0.9})
    assert events[0].severity == "CRITICAL" and events[0].rule_id == "equipment_malfunction_hydraulic"


def test_low_fuel_and_weather_are_high():
    sev = {e.rule_id: e.severity for e in sg.evaluate_record({"fuel_level_pct": 20, "rain_mm": 8})}
    assert sev == {"low_fuel_reserve": "HIGH", "weather_heavy_rain": "HIGH"}


def test_proximity_group_keeps_only_most_severe():
    events = sg.evaluate_record({"machine_state": "swinging", "worker_distance_m": 4, "closing_speed_mps": 2.0})
    grouped = {x["id"] for x in sg.load_rules()["rules"] if x.get("group") == "worker_proximity"}
    proximity = [e for e in events if e.rule_id in grouped]
    assert len(proximity) == 1 and proximity[0].severity == "CRITICAL"


def test_missing_fields_never_fire():
    assert sg.evaluate_record({}) == []


def test_deterministic():
    rec = {"timestamp": "2026-01-01T08:00:00", "machine_state": "traveling", "machine_speed_kmh": 8,
           "seatbelt_status": "off"}
    assert sg.evaluate_record(rec) == sg.evaluate_record(rec)


def test_evaluate_safety_contract():
    ctx = SessionContext(session_id="S1", operator_id="OP1", machine_id="EXC001", timestamp="2026-01-01T08:00:00",
                         machine=MachineContext(machine_id="EXC001", engine_state="safe_idle"),
                         traffic=TrafficContext(worker_proximity=100.0),
                         temporal=TemporalContext(shift_elapsed_min=60))
    ev = sg.evaluate_safety(ctx)
    assert ev.severity == "INFO" and ev.trigger_reason == "no_hazard_detected"

    ctx.temporal.shift_elapsed_min = 650
    ev = sg.evaluate_safety(ctx)
    assert ev.severity == "CRITICAL" and ev.required_action == "end_shift_and_rest" and ev.recommendation


def test_rules_never_under_call_logged_events():
    """Rule engine may be more conservative than logged labels, never less."""
    v = sg.validate_against_logged()
    order = sg.load_rules()["severity_order"]
    conf = v["confusion"]
    for logged in conf.index:
        for predicted in conf.columns:
            if predicted != "NONE" and order.index(predicted) < order.index(logged):
                assert conf.loc[logged, predicted] == 0, (logged, predicted)
