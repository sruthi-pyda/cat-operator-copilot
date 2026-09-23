import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.safety import safety_guardian as sg
from features.llm.ollama_client import GenerationResult


class FakeLLM:
    def __init__(self, text="Rain has softened the ground, so stopping distance is longer.", available=True):
        self.text, self.available, self.calls = text, available, 0

    def generate(self, prompt, system=None, fallback=None, options=None):
        self.calls += 1
        if not self.available:
            return GenerationResult(fallback(prompt), "fallback", None, 0.0, "server down")
        return GenerationResult(self.text, "ollama", "mistral:latest", 5.0)


def event(record):
    return sg.evaluate_record(record)[0]


HIGH = {"timestamp": "2026-01-05T10:00:00", "rain_mm": 8.0}                 # weather_heavy_rain
MEDIUM = {"timestamp": "2026-01-05T10:00:00", "maintenance_status": "overdue"}
CRITICAL = {"timestamp": "2026-01-05T10:00:00", "shift_elapsed_min": 650}   # operator_fatigue_critical


def test_high_gets_llm_reasoning_and_decision_unchanged():
    ev, llm = event(HIGH), FakeLLM()
    r = sg.context_aware_safety_reasoning(ev, {"surface_type": "wet"}, client=llm)
    assert r["reasoning_source"] == "llm" and "stopping distance" in r["reasoning"]
    assert (r["severity"], r["required_action"], r["decision_changed"]) == (ev.severity, ev.required_action, False)
    assert r["reasoning"].startswith(ev.recommendation)


def test_medium_is_enhanced():
    r = sg.context_aware_safety_reasoning(event(MEDIUM), {}, client=FakeLLM())
    assert r["severity"] == "MEDIUM" and r["reasoning_source"] == "llm"


def test_critical_never_calls_llm():
    ev, llm = event(CRITICAL), FakeLLM()
    r = sg.context_aware_safety_reasoning(ev, {"note": "operator says they feel fine"}, client=llm)
    assert llm.calls == 0
    assert r["severity"] == "CRITICAL" and r["required_action"] == "end_shift_and_rest"
    assert r["reasoning_source"] == "rules"


@pytest.mark.parametrize("bad", ["This is a false alarm.", "It is safe to continue operating.",
                                 "You can OVERRIDE this and keep digging."])
def test_contradicting_llm_text_is_discarded(bad):
    ev = event(HIGH)
    r = sg.context_aware_safety_reasoning(ev, {}, client=FakeLLM(text=bad))
    assert r["reasoning_source"] == "rules" and "contradicts rule" in r["llm_note"]
    assert bad not in r["reasoning"]


def test_falls_back_to_rules_when_llm_unavailable():
    ev = event(HIGH)
    r = sg.context_aware_safety_reasoning(ev, {}, client=FakeLLM(available=False))
    assert r["reasoning_source"] == "rules" and ev.recommendation in r["reasoning"]
    assert "unavailable" in r["llm_note"]


def test_long_llm_output_truncated():
    r = sg.context_aware_safety_reasoning(event(HIGH), {}, client=FakeLLM(text="word " * 500))
    llm_part = r["reasoning"][len(event(HIGH).recommendation):]
    assert len(llm_part.strip()) <= sg.load_rules()["llm_reasoning"]["max_chars"]


def test_real_client_without_ollama_falls_back():
    r = sg.context_aware_safety_reasoning(event(HIGH), {"surface_type": "wet"})
    assert r["reasoning_source"] in ("rules", "llm") and r["decision_changed"] is False
