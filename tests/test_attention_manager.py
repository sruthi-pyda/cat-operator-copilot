import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.attention import attention_manager as amod
from features.attention.attention_manager import AttentionManager
from shared.schemas import AttentionEvent

TS = "2026-01-05T10:00:00"


def ev(eid, etype, sev, state="swinging", action="next_safe_state", conf=0.9, reason=None, ts=TS):
    return AttentionEvent(eid, etype, sev, "x", action, conf, state, "", reason or eid, ts)


def test_critical_always_shows_even_when_active():
    r = AttentionManager().submit(ev("c", "safety", "CRITICAL", action="act_now"))
    assert r["decision"] == "show_now" and r["route"] == "safety_guardian"


def test_non_safety_queued_while_operating_and_released_in_safe_state():
    am = AttentionManager()
    assert am.submit(ev("t", "training", "LOW"))["decision"] == "queue"
    assert am.release("swinging") == []
    out = am.release("safe_idle")
    assert out[0]["event_id"] == "t" and out[0]["route"] == "training_hub"


def test_buddy_gated_to_safe_states():
    am = AttentionManager()
    assert am.submit(ev("b1", "buddy", "LOW"))["decision"] == "suppress"
    assert am.submit(ev("b2", "buddy", "LOW", state="parked"))["decision"] == "show_now"


def test_ranking_follows_tier_order():
    am = AttentionManager()
    ranked = am.rank([ev("tr", "training", "MEDIUM"), ev("bh", "behavior_nudge", "LOW"),
                      ev("rp", "task_replanning", "LOW"), ev("hs", "safety", "HIGH"),
                      ev("cs", "safety", "CRITICAL")])
    assert [e.event_id for e in ranked] == ["cs", "hs", "rp", "bh", "tr"]


def test_low_confidence_non_safety_suppressed():
    r = AttentionManager().submit(ev("n", "behavior_nudge", "LOW", state="safe_idle", conf=0.2))
    assert r["decision"] == "suppress"


def test_duplicate_within_window_bundled():
    am = AttentionManager()
    am.submit(ev("a", "safety", "MEDIUM", reason="same"))
    assert am.submit(ev("b", "safety", "MEDIUM", reason="same", ts="2026-01-05T10:00:10"))["decision"] == "bundle"


def test_queue_overflow_drops_lowest_ranked():
    am = AttentionManager()
    for i in range(am.cfg["max_queue_size"]):
        am.submit(ev(f"t{i}", "training", "LOW", reason=f"r{i}"))
    r = am.submit(ev("hs", "safety", "HIGH", reason="urgent"))
    assert r["decision"] == "queue" and "dropped_event" in r
    assert am.queue[0].event_id == "hs" and len(am.queue) == am.cfg["max_queue_size"]


def test_route_event_contract_shape():
    amod._default_manager = None
    r = amod.route_event(ev("x", "safety", "CRITICAL", action="act_now"))
    assert {"event_id", "decision", "reason", "queued_events"} <= set(r)
