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
