"""
Attention Manager — central event priority arbiter
Owner: Aneesha (Team Member 2)

Every subsystem produces candidate events; only this module decides what
reaches the operator, when, and through which subsystem. Deterministic —
no LLM involvement.

Tier order (docs/ARCHITECTURE.md, config/settings.yaml attention.priority_order):
  1 critical_safety  2 high_safety  3 task_replanning
  4 behavior_nudge   5 training     6 buddy

Decision policy (show_now | queue | bundle | suppress):
  CRITICAL safety         → show_now, always (preempts everything)
  HIGH safety, act_now    → show_now
  HIGH/MEDIUM safety      → show_now in a safe state, else queue
  LOW safety              → bundle (digest at next safe state)
  INFO safety             → suppress (logged only)
  replanning / behavior / training → show_now only in a safe state with
                            nothing more urgent queued; else queue
  buddy                   → show_now in a safe state, else suppress (Buddy is safe-state gated)
  non-safety confidence below attention.min_confidence → suppress
  duplicate (same tier + reason) within bundle_window_sec → bundle
  queue over max_queue_size → lowest-ranked event is dropped (suppress)

Public API:
  route_event(attention_event)       → dict  (API contract; module-level manager)
  AttentionManager.submit(event)     → dict
  AttentionManager.release(state)    → List[dict]  (queued events now allowed to show)
  AttentionManager.rank(events)      → List[AttentionEvent]
  from_safety_event / from_training_trigger / from_optimization / from_behavior → AttentionEvent
"""

import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.schemas import AttentionEvent, SafetyEvent, TrainingTrigger, BehaviorResult
from shared.constants import SEVERITY_LEVELS

SETTINGS_PATH = os.path.join(ROOT, "config", "settings.yaml")
RULES_PATH    = os.path.join(ROOT, "config", "safety_rules.yaml")

SAFETY, REPLAN, BEHAVIOR, TRAINING, BUDDY = "safety", "task_replanning", "behavior_nudge", "training", "buddy"
ACT_NOW, NEXT_SAFE, INFORMATIONAL = "act_now", "next_safe_state", "informational"
URGENCY_BY_SEVERITY = {"CRITICAL": "immediate", "HIGH": "high", "MEDIUM": "normal", "LOW": "low", "INFO": "low"}
STOP_ACTIONS = {"emergency_stop", "stop_immediately", "stop_or_safe_action"}


def _load_config() -> Dict[str, Any]:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        att = yaml.safe_load(f)["attention"]
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    att["safe_states"] = set(rules["machine_state"]["safe_states"])
    att["critical_always_wins"] = rules["attention_override"]["critical_always_wins"]
    return att


def _parse_ts(ts: Optional[str]) -> datetime:
    try:
        return datetime.fromisoformat(ts) if ts else datetime.now()
    except ValueError:
        return datetime.now()


# ── Adapters: subsystem outputs → AttentionEvent ──────────────────────────────

def from_safety_event(ev: SafetyEvent, operator_state: str) -> AttentionEvent:
    if ev.required_action in STOP_ACTIONS:
        actionability = ACT_NOW
    elif ev.required_action == "none_required":
        actionability = INFORMATIONAL
    else:
        actionability = NEXT_SAFE
    return AttentionEvent(
        event_id=f"AE-{ev.event_id}", event_type=SAFETY, severity=ev.severity,
        urgency=URGENCY_BY_SEVERITY[ev.severity], actionability=actionability,
        confidence=ev.confidence, operator_state=operator_state, decision="",
        reason=ev.recommendation or ev.trigger_reason, timestamp=ev.timestamp,
    )


def from_training_trigger(tt: TrainingTrigger, operator_state: str,
                          timestamp: Optional[str] = None) -> AttentionEvent:
    escalated = "escalat" in (tt.escalation_state or "")
    return AttentionEvent(
        event_id=f"AE-{tt.trigger_event_id}", event_type=TRAINING,
        severity="MEDIUM" if escalated else "LOW",
        urgency="normal" if escalated else "low", actionability=NEXT_SAFE,
        confidence=tt.confidence, operator_state=operator_state, decision="",
        reason=f"Training {tt.lesson_id} for {tt.issue_type} ({tt.escalation_state})",
        timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
    )


def from_optimization(rec: Dict[str, Any], operator_state: str,
                      timestamp: Optional[str] = None) -> AttentionEvent:
    """Accepts a rank_assignments() assignment or a generate_plan() step."""
    late = not rec.get("deadline_met", True)
    at_risk = rec.get("deadline_at_risk", False)
    severity = "HIGH" if late else "MEDIUM" if at_risk else "LOW"
    return AttentionEvent(
        event_id=f"AE-OPT-{rec.get('operator_id')}-{rec.get('task_id')}", event_type=REPLAN,
        severity=severity, urgency=URGENCY_BY_SEVERITY[severity], actionability=NEXT_SAFE,
        confidence=float(rec.get("prediction_confidence", 0.8)), operator_state=operator_state,
        decision="", reason=f"Next task {rec.get('task_id')} on {rec.get('machine_id')}: {rec.get('reason', '')}",
        timestamp=timestamp or rec.get("start") or datetime.now().isoformat(timespec="seconds"),
    )


def from_behavior(br: BehaviorResult, operator_state: str,
                  timestamp: Optional[str] = None) -> AttentionEvent:
    # Non-eligible results (context-driven / low confidence) carry zero
    # confidence so they are suppressed — no raw behavioural blame.
    return AttentionEvent(
        event_id=f"AE-BEH-{br.session_id}", event_type=BEHAVIOR, severity="LOW", urgency="low",
        actionability=NEXT_SAFE if br.coaching_eligible else INFORMATIONAL,
        confidence=br.confidence if br.coaching_eligible else 0.0,
        operator_state=operator_state, decision="",
        reason=f"Behavior nudge ({br.attribution}, residual {br.operator_residual:+.3f})",
        timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
    )


# ── Manager ───────────────────────────────────────────────────────────────────

class AttentionManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or _load_config()
        self.queue: List[AttentionEvent] = []
        self.bundle: List[AttentionEvent] = []
        self.log: List[Dict[str, Any]] = []
        self._recent: Dict[tuple, datetime] = {}

    # -- ranking --

    def tier(self, ev: AttentionEvent) -> str:
        if ev.event_type == SAFETY:
            return "critical_safety" if ev.severity == "CRITICAL" else "high_safety"
        return ev.event_type

    def urgency_key(self, ev: AttentionEvent) -> tuple:
        """Sort key: lower = more urgent. Tier, then severity, then confidence, then age."""
        return (self.cfg["priority_order"].index(self.tier(ev)),
                -SEVERITY_LEVELS.index(ev.severity),
                -ev.confidence,
                _parse_ts(ev.timestamp))

    def rank(self, events: List[AttentionEvent]) -> List[AttentionEvent]:
        return sorted(events, key=self.urgency_key)

    def is_safe_state(self, state: str) -> bool:
        return state in self.cfg["safe_states"]

    # -- decision --

    def _decide(self, ev: AttentionEvent) -> (str, str):
        safe = self.is_safe_state(ev.operator_state)
        tier = self.tier(ev)
        more_urgent_queued = any(self.urgency_key(q) < self.urgency_key(ev) for q in self.queue)

        if tier == "critical_safety" and self.cfg["critical_always_wins"]:
            return "show_now", "CRITICAL safety event always wins"
        if ev.event_type == SAFETY and ev.severity == "HIGH" and ev.actionability == ACT_NOW:
            return "show_now", "HIGH safety event requiring immediate action"

        if ev.event_type != SAFETY and ev.confidence < self.cfg["min_confidence"]:
            return "suppress", f"confidence {ev.confidence:.2f} below {self.cfg['min_confidence']}"

        key = (tier, ev.reason)
        last = self._recent.get(key)
        if last and abs((_parse_ts(ev.timestamp) - last).total_seconds()) <= self.cfg["bundle_window_sec"]:
            return "bundle", f"duplicate within {self.cfg['bundle_window_sec']} s window"

        if ev.event_type == SAFETY:
            if ev.severity == "INFO" or ev.actionability == INFORMATIONAL:
                return "suppress", "informational safety event, logged only"
            if ev.severity == "LOW":
                return "bundle", "LOW safety event added to next safe-state digest"
            if safe:
                return "show_now", f"{ev.severity} safety event; operator in safe state"
            return "queue", f"{ev.severity} safety event held until operator reaches a safe state"

        if ev.event_type == BUDDY:
            if safe:
                return "show_now", "Buddy allowed in safe state"
            return "suppress", "Buddy gated to safe states"

        if not safe:
            return "queue", f"{tier} held: operator is {ev.operator_state}"
        if more_urgent_queued:
            return "queue", f"{tier} held behind more urgent queued events"
        return "show_now", f"{tier}; operator in safe state and nothing more urgent pending"

    def submit(self, event: AttentionEvent) -> Dict[str, Any]:
        decision, reason = self._decide(event)
        event = replace(event, decision=decision, reason=event.reason)
        dropped = None

        if decision == "queue":
            self.queue.append(event)
            self.queue = self.rank(self.queue)
            if len(self.queue) > self.cfg["max_queue_size"]:
                dropped = self.queue.pop()
                self.log.append({"event_id": dropped.event_id, "decision": "suppress",
                                 "reason": "queue full; lowest-ranked event dropped"})
        elif decision == "bundle":
            self.bundle.append(event)

        self._recent[(self.tier(event), event.reason)] = _parse_ts(event.timestamp)
        result = {
            "event_id": event.event_id,
            "decision": decision,
            "reason": reason,
            "route": self.cfg["routes"][self.tier(event)],
            "tier": self.tier(event),
            "queued_events": [q.event_id for q in self.queue],
        }
        if dropped:
            result["dropped_event"] = dropped.event_id
        self.log.append(result)
        return result

    def submit_many(self, events: List[AttentionEvent]) -> List[Dict[str, Any]]:
        """Submit a batch most-urgent-first, so a CRITICAL in the batch is decided before anything else."""
        return [self.submit(e) for e in self.rank(events)]

    def release(self, operator_state: str) -> List[Dict[str, Any]]:
        """
        Operator state changed. In a safe state, surface the single most urgent
        queued event (one attention owner — one thing at a time) plus the
        bundled digest. Returns [] when nothing may be shown.
        """
        if not self.is_safe_state(operator_state):
            return []
        out = []
        if self.queue:
            ev = self.queue.pop(0)
            out.append({"event_id": ev.event_id, "decision": "show_now",
                        "reason": f"released from queue: operator now {operator_state}",
                        "route": self.cfg["routes"][self.tier(ev)], "tier": self.tier(ev),
                        "queued_events": [q.event_id for q in self.queue]})
        if self.bundle:
            digest = self.rank(self.bundle)
            self.bundle = []
            out.append({"event_id": "BUNDLE", "decision": "show_now",
                        "reason": f"digest of {len(digest)} bundled events",
                        "route": "dashboard", "tier": "bundle",
                        "bundled_events": [{"event_id": b.event_id, "reason": b.reason} for b in digest],
                        "queued_events": [q.event_id for q in self.queue]})
        self.log.extend(out)
        return out


_default_manager: Optional[AttentionManager] = None


def get_manager() -> AttentionManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AttentionManager()
    return _default_manager


def route_event(attention_event: AttentionEvent) -> Dict[str, Any]:
    """API contract entry point: route one event through the shared manager."""
    return get_manager().submit(attention_event)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    from features.safety.safety_guardian import evaluate_record
    am = AttentionManager()
    ts = "2026-01-05T10:00:00"

    safety = evaluate_record({
        "timestamp": ts, "session_id": "S_DEMO", "machine_id": "EXC001", "operator_id": "OP1005",
        "machine_state": "swinging", "worker_distance_m": 9.0, "closing_speed_mps": 1.8,
        "shift_elapsed_min": 520, "rain_mm": 6.0, "congestion_level": "high",
    })
    events = [from_safety_event(e, "swinging") for e in safety]
    events.append(from_training_trigger(TrainingTrigger("TR_DEMO", "OP1005", "idle_reduction", "operator-driven",
                                                        0.82, "L003", "first_trigger", True), "swinging", ts))
    events.append(from_optimization({"operator_id": "OP1005", "machine_id": "EXC001", "task_id": "T00010",
                                     "deadline_met": True, "deadline_at_risk": True,
                                     "prediction_confidence": 0.88, "reason": "priority 1"}, "swinging", ts))
    events.append(AttentionEvent("AE-BUDDY-1", BUDDY, "LOW", "low", INFORMATIONAL, 0.9, "swinging", "", "Buddy tip", ts))

    print("--- submit_many (operator swinging) ---")
    for r in am.submit_many(events):
        print(f"  {r['decision']:9s} [{r['tier']:15s} -> {r['route']:15s}] {r['event_id']}: {r['reason']}")
    print(f"  queue: {[q.event_id for q in am.queue]}")
    print("--- release (operator now safe_idle) ---")
    for r in am.release("safe_idle"):
        print(f"  {r['decision']:9s} [{r['tier']:15s} -> {r['route']:15s}] {r['event_id']}: {r['reason']}")


if __name__ == "__main__":
    _smoke_test()
