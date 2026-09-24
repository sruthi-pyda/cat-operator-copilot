import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.buddy import buddy as bd
from features.buddy.buddy import ANSWER_MODE_LLM, ANSWER_MODE_QUOTED, ask, synthesize_answer_with_llm
from features.buddy.evidence import (SOURCE_MACHINE_MANUAL, SOURCE_SAFETY_INCIDENT, SOURCE_TASK_PLAN,
                                     SOURCE_TELEMETRY, Evidence)
from features.buddy.safe_state import MachineStateSnapshot
from features.llm.ollama_client import GenerationResult

NOW = datetime(2026, 9, 23, 10, 0, 0)
TS = "2026-09-23T09:50:00"
SAFE_STATES = frozenset({"parked", "safe_idle"})
SETTINGS_ON = {"buddy": {"no_attachment_movement_required": True, "max_evidence_age_min": 60,
                         "llm_synthesis": True}}
PARKED = MachineStateSnapshot(machine_state="parked", attachment_movement="none", machine_speed_kmh=0.0)


class FakeLLM:
    def __init__(self, text="", available=True):
        self.text, self.available, self.prompts = text, available, []

    def generate(self, prompt, system=None, fallback=None, options=None):
        self.prompts.append(prompt)
        if not self.available:
            return GenerationResult(fallback(prompt), "fallback", None, 0.0, "server down")
        return GenerationResult(self.text, "ollama", "mistral:latest", 3.0)


def ev(source, value, content):
    return Evidence(source=source, value=value, content=content, timestamp=TS, confidence=0.9)


FUEL = [ev(SOURCE_TELEMETRY, 60.0, "Fuel level is 60 percent."),
        ev(SOURCE_TASK_PLAN, 60.0, "Planned refuel threshold check: fuel at 60 percent.")]


def _ask(question, evidence, llm, settings=SETTINGS_ON):
    return ask(question, PARKED, evidence, as_of=NOW, settings=settings, safe_states=SAFE_STATES, llm_client=llm)


def test_synthesizes_multiple_agreeing_items_with_citation():
    llm = FakeLLM("You have 60 percent fuel, which matches the plan.")
    r = _ask("how much fuel do I have", FUEL, llm)
    assert r.answered and r.answer_mode == ANSWER_MODE_LLM
    assert r.answer.endswith("[sources: telemetry, task_plan]")
    assert "Fuel level is 60 percent." in llm.prompts[0] and "how much fuel" in llm.prompts[0]


def test_ungrounded_number_falls_back_to_quotes():
    r = _ask("how much fuel do I have", FUEL, FakeLLM("You have 75 percent fuel."))
    assert r.answer_mode == ANSWER_MODE_QUOTED
    assert "75" not in r.answer and "(source: telemetry" in r.answer and "[sources:" in r.answer


def test_llm_unavailable_falls_back_to_direct_quotes():
    r = _ask("how much fuel do I have", FUEL, FakeLLM(available=False))
    assert r.answered and r.answer_mode == ANSWER_MODE_QUOTED
    assert "Fuel level is 60 percent. (source: telemetry" in r.answer
    assert r.answer.endswith("[sources: telemetry, task_plan]")


def test_single_evidence_item_is_quoted_without_llm():
    llm = FakeLLM("anything")
    r = _ask("how much fuel do I have", FUEL[:1], llm)
    assert r.answer_mode == ANSWER_MODE_QUOTED and not llm.prompts


def test_safety_critical_question_never_uses_llm():
    items = [ev(SOURCE_SAFETY_INCIDENT, "stop", "Stop: worker inside swing envelope."),
             ev(SOURCE_MACHINE_MANUAL, "stop", "Stop all swing motion when a worker enters the envelope.")]
    llm = FakeLLM("Just keep going carefully.")
    r = _ask("is it safe to swing now", items, llm)
    assert not llm.prompts and r.answer_mode == ANSWER_MODE_QUOTED
    assert "keep going" not in r.answer


def test_conflicting_sources_keep_authoritative_quote():
    items = [ev(SOURCE_TELEMETRY, 60.0, "Fuel level is 60 percent."),
             ev(SOURCE_TASK_PLAN, 40.0, "Plan assumed 40 percent fuel.")]
    llm = FakeLLM("Fuel is 50 percent.")
    r = _ask("how much fuel do I have", items, llm)
    assert not llm.prompts and "disagree" in r.answer.lower() and r.answer_mode == ANSWER_MODE_QUOTED


def test_synthesis_off_when_setting_missing():
    llm = FakeLLM("You have 60 percent fuel.")
    settings = {"buddy": {"no_attachment_movement_required": True, "max_evidence_age_min": 60}}
    r = _ask("how much fuel do I have", FUEL, llm, settings=settings)
    assert not llm.prompts and r.answer_mode == ANSWER_MODE_QUOTED


def test_to_dict_exposes_answer_mode():
    r = _ask("how much fuel do I have", FUEL, FakeLLM("You have 60 percent fuel."))
    assert r.to_dict(as_of=NOW)["answer_mode"] == ANSWER_MODE_LLM


def test_direct_function_contract():
    answer, mode = synthesize_answer_with_llm(FUEL, "fuel?", client=FakeLLM(available=False))
    assert mode == ANSWER_MODE_QUOTED and answer.endswith("[sources: telemetry, task_plan]")
