# API Contract

All features receive a `SessionContext` and return a documented output object.
**Never invent different names for the same field.** Use `machine_id`, not `machineId`, `machine`, `excavator_id`, or `equipment_id`.

## Shared Input: SessionContext

```json
{
  "session_id": "S000123",
  "operator_id": "OP1001",
  "machine_id": "EXC001",
  "task_id": "T00042",
  "timestamp": "2026-09-23T08:15:00",
  "operator": { "...see OperatorContext..." },
  "machine": { "...see MachineContext..." },
  "task": { "...see TaskContext..." },
  "site": { "...see SiteContext..." },
  "traffic": { "...see TrafficContext..." },
  "weather": { "...see WeatherContext..." },
  "temporal": { "...see TemporalContext..." },
  "data_quality": { "source": "synthetic", "synthetic_flag": true, "confidence": 0.85 }
}
```

---

## Feature 04 — Prediction

**Function:** `predict_task(session_context: SessionContext) → PredictionResult`

```json
{
  "session_id": "S000123",
  "eta_p10": 25.0,
  "eta_p50": 31.0,
  "eta_p90": 41.0,
  "fuel_p10": 4.2,
  "fuel_p50": 5.1,
  "fuel_p90": 6.4,
  "confidence": 0.81,
  "factors": ["workload", "soil_hardness", "congestion"],
  "predicted_vs_actual": null,
  "synthetic_flag": true,
  "data_quality_confidence": 0.85
}
```

---

## Feature 02 — Safety

**Function:** `evaluate_safety(session_context: SessionContext) → SafetyEvent`

```json
{
  "event_id": "SE00123",
  "severity": "HIGH",
  "trigger_reason": "worker_inside_swing_envelope_with_closing_motion",
  "required_action": "stop_or_safe_action",
  "confidence": 0.94,
  "timestamp": "2026-09-23T10:15:00",
  "synthetic_flag": true
}
```

Severity levels: `INFO | LOW | MEDIUM | HIGH | CRITICAL`

---

## Feature 03 — Behavior

**Function:** `analyze_behavior(session_context: SessionContext) → BehaviorResult`

```json
{
  "session_id": "S000123",
  "attribution": "context-driven",
  "confidence": 0.78,
  "observed_value": 0.18,
  "expected_value": 0.16,
  "operator_residual": 0.01,
  "context_explained_component": 0.01,
  "coaching_eligible": false,
  "synthetic_flag": true
}
```

Attribution values: `operator-driven | context-driven | mixed | insufficient_evidence`

---

## Feature 05 — Optimization

**Function:** `generate_plan(tasks, session_context: SessionContext) → dict`

```json
{
  "session_id": "S000123",
  "optimized_sequence": ["T00042", "T00017", "T00089"],
  "total_cost": 142.5,
  "cost_breakdown": {
    "time_cost": 90.0,
    "fuel_cost": 22.5,
    "deadline_risk": 15.0,
    "transition_cost": 10.0,
    "safety_condition_risk": 5.0
  },
  "reason": "Priority + deadline ordering with travel cost minimization",
  "synthetic_flag": true
}
```

---

## Attention Manager

**Function:** `route_event(attention_event: AttentionEvent) → dict`

```json
{
  "event_id": "AE00456",
  "decision": "show_now",
  "reason": "CRITICAL safety event always wins",
  "queued_events": []
}
```

Decisions: `show_now | queue | bundle | suppress`

---

## Training Trigger

**Function:** `check_training_trigger(behavior_history, operator_id, issue_type, prior_trigger_count=0) → TrainingTrigger | None`

> **Signature amended 2026-09-23 (Member 3).** The original contract took only
> `behavior_history`. `BehaviorResult` carries neither `operator_id` nor
> `issue_type`, so those must be supplied by the caller; `prior_trigger_count`
> drives the escalation state. `BehaviorResult` itself is unchanged.
>
> `evaluate_training_gate(...)` returns the same decision as a `TriggerDecision`
> with reason codes and per-occurrence checks — use it when the dashboard needs
> to show *why* nothing fired.

```json
{
  "trigger_event_id": "TR00023",
  "operator_id": "OP1001",
  "issue_type": "idle_reduction",
  "attribution_type": "operator-driven",
  "confidence": 0.82,
  "lesson_id": "L003",
  "escalation_state": "first_trigger",
  "synthetic_flag": true
}
```

Training triggers only when issue is repeated AND confidence is high AND not primarily context-explained.

---

## Feature 01 — Operator Passport

**Identify:** `identify(frame, recognizer, store, threshold=None) → IdentificationResult`

```json
{
  "authenticated": true,
  "operator_id": "OP1003",
  "confidence": 0.7418,
  "threshold": 0.7,
  "scores": {"OP1003": 0.7418}
}
```

Below the threshold, `operator_id` is `null` and no session may be opened.

**Open a session:** `create_session(operator_id, machine_id, repository, task_id=None) → SessionResult`

```json
{
  "authorized": true,
  "authorization": {
    "authorized": true,
    "operator_id": "OP1003",
    "machine_id": "EXC001",
    "reasons": [],
    "checks": {
      "machine_type_authorized": true,
      "certification_status_valid": true,
      "certification_not_expired": true
    }
  },
  "session_id": "S000123"
}
```

`session_context` is `null` whenever `authorized` is false. Refusal reason codes:
`machine_type_not_authorized`, `certification_not_valid`, `certification_expired`.

Identity and authorization are separate: a recognised operator can still be refused.

---

## Feature 07 — Grounded Buddy

**Safe-state gate:** `evaluate_safe_state(snapshot) → SafeStateResult`

```json
{
  "allowed": false,
  "reasons": ["machine_state_not_safe", "attachment_movement_active", "arm_in_motion"],
  "checks": {
    "machine_state_safe": false,
    "attachment_stationary": false,
    "arm_stationary": false,
    "bucket_stationary": true,
    "machine_stationary": true
  }
}
```

Unknown machine state or attachment movement **denies**. Safe states come from
`config/safety_rules.yaml`, which Safety owns — the Buddy keeps no private definition.

**Ask:** `ask(question, machine_state, evidence) → BuddyResponse`

```json
{
  "answered": false,
  "status": "deferred_safety_critical",
  "reason": "Safety-critical question with no Safety Guardian or approved manual evidence; deferring.",
  "answer": null,
  "deferral_target": "safety_guardian",
  "evidence": [],
  "conflict": null,
  "synthetic_flag": true
}
```

Statuses: `answered`, `blocked_unsafe_state`, `deferred_no_evidence`,
`deferred_stale_evidence`, `deferred_conflict`, `deferred_safety_critical`.

**Retrieve evidence:** `retrieve(question, session_context=..., telemetry_row=..., safety_events=..., task_plan=..., prediction=..., training_state=...) → tuple[Evidence, ...]`

Returned highest-authority first. Safety-critical questions may only be answered from
`safety_incident` or `machine_manual`.

---

## Attention candidates produced by this slice

`training_candidate`, `buddy_candidate`, `replan_candidate` and `safety_candidate` in
`features/dashboard/attention_candidates.py` return `AttentionEvent` with
`decision = "pending"`. They propose only; the Attention Manager decides. Training and
Buddy are always `deferrable` / `actionable_when_stopped` so they can be held across a
shift rather than competing with active operation.
