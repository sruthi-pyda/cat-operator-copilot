# Technical Review Prep

Everything here is verified against the code. Numbers are from `reports/model_metrics.json`,
`config/safety_rules.yaml` and the test suite — not estimates.

---

## Numbers to know cold

| | |
|---|---|
| Code | ~9,700 lines across `features/`, `app/`, `shared/`, `scripts/` |
| Tests | **423 collected**, 410 pass + 13 skipped (slow tier opt-in via `RUN_SLOW=1`) |
| Safety rules | **33** — 7 CRITICAL, 11 HIGH, 11 MEDIUM, 3 LOW, 1 INFO |
| Prediction features | **18** (12 numeric, 6 categorical) |
| ETA | MAE **8.36 min**, RMSE 13.31, baseline MAE **31.68** → **73.6%** better |
| Fuel | MAE **3.54 L**, RMSE 7.22, baseline MAE **12.04** → **70.6%** better |
| Split | 80/20, seed 42 — 4,000 train / 1,000 test |
| **P10–P90 coverage** | **0.643 (ETA) / 0.666 (fuel)** — should be ~0.80. **Known weakness.** |
| Dataset | 30 operators, 10 machines, 1,500 tasks, 5,000 sessions, 75,000 telemetry rows |

---

## The 60-second architecture answer

> "One shared `SessionContext` — operator, machine, task, site, traffic, weather, temporal,
> data quality — built once per session and passed to every feature. Seven features consume it
> and return documented result objects; nothing imports another feature's internals.
>
> Prediction is LightGBM quantile regression. Behaviour is Ridge, deliberately. Safety and
> authorization are deterministic rule evaluation with no model involved. Optimization is a
> greedy weighted objective over those predictions. The Attention Manager arbitrates what
> actually reaches the operator.
>
> The dashboard resolves each feature by name at call time, so it runs with any subset
> integrated and reports the rest as unavailable rather than faking a value."

---

## Component detail

### Prediction — LightGBM quantile regression
- **Three models per target**, `objective='quantile'` at `alpha` 0.1 / 0.5 / 0.9, 300 estimators.
  Six regressors total. The band is **learned**, not derived from residual spread.
- Confidence is derived from relative band width: `1 − (P90−P10)/P50 × 0.5`, clipped to [0.40, 0.96].
- `predict_task` returns ETA with fuel zeroed; `predict_fuel` the reverse; `predict_combined`
  populates both. The dashboard uses `predict_combined` — a zeroed percentile is rendered as
  "not predicted", never as 0.
- **Why LightGBM:** native quantile objective is the decisive reason; also native categorical
  handling and fast retraining. *Honest:* the architecture named it first and no alternative
  was benchmarked.

### Behaviour — Ridge, not boosted trees
- Predicts *expected* idle ratio from context, then `residual = observed − expected`, split into
  `operator_residual` and `context_explained_component` (both idle ratios — same unit, enforced).
- **Why linear:** the output is subtracted from, not consumed directly. Interpretability beats
  accuracy when the number's job is to be decomposed and defended to an operator.
- `context_share = |context| / (|operator| + |context|)`. Zero gap returns **1.0**, so a session
  with no measurable deviation can never be coached.

### Safety — declarative rule evaluation
- 33 rules in YAML. Each has id, severity, required_action, and AND-ed conditions. Operators:
  `gt gte lt lte eq ne in not_in is_true is_false`. OR is expressed as separate rules.
- Thresholds live only in threshold sections; rules reference them by path (`ref: slope.critical_deg`).
  **No number is hard-coded in Python.**
- **A condition on a missing field never matches** — a sensor dropout cannot manufacture an alarm.
- `group:` de-duplicates: when several proximity rules fire, only the most severe survives.
- **Derived facts:** `machine_active` = set membership in [swinging, loading, digging, traveling,
  grading]; `worker_in_envelope` = machine swinging **and** within a 12 m radius; `shift_hours` =
  `shift_elapsed_min / 60`.

### Optimization — greedy weighted objective
```
total_cost = time_cost (ETA P50 × 1.0)
           + fuel_cost (P50 × 2.0)
           + deadline_risk
           + transition_cost
           + safety_condition_risk
score      = total_cost / priority_factor      (P1 ÷ 2.0 … P5 ÷ 1.0)
```
- **Deadline risk keys off P90**: P50 late → 120 + 5/h; P50 fine but P90 late → 40; else 0.
  Uncertainty feeds the plan, not just the display.
- Transition = `km × 7.5` + 20 attachment swap + 15 blocked route.
- Safety cost: INFO 0, LOW 2, MEDIUM 10, HIGH 30.
- **Hard blocks, not weights:** `if critical: return None` before scoring; 8 h fatigue budget
  ends the plan; unmet dependencies exclude.

### Passport — biometrics
- DeepFace + ArcFace, 512-d embeddings, cosine similarity, threshold 0.70 from config.
- Matching takes the **max** similarity across all enrolled frames for an operator.
- Embeddings stored locally and gitignored; no image in the CSV, nothing uploaded.
- Identity and authorization are separate: recognition returns who; `check_authorization`
  decides what they may run (machine type, cert status, cert expiry — all reported, not just the first).

### Buddy — grounded retrieval
- Sources ranked: safety_incident 100, machine_manual 90, telemetry 70, passport 60, task_plan 55,
  training_state/session 50, prediction 30.
- **Safe-state gate** reads `machine_state.safe_states` from the *Safety Guardian's* config, so the
  two cannot drift. Unknown state or unknown attachment movement **denies**.
- Safety-critical questions are answerable only from safety_incident or machine_manual.
- Conflict: same-source multiples collapse for naturally multi-valued sources (incidents, manual);
  the manual is excluded from value comparison entirely (guidance vs measurement are not rivals);
  a genuine tie at top authority **defers**.
- LLM synthesis is optional, off for safety questions, and **any number not present in the
  evidence discards the answer**. CRITICAL is never sent to the LLM.

### Training gate — four conditions
An occurrence counts only if **all** hold: attribution ∈ {operator-driven, mixed}; confidence ≥ 0.70;
`context_share` ≤ 0.50; and the residual is in the **unfavourable** direction. Fires when ≥ 2
qualifying occurrences in the last 10. **Fires for 5 of 30 operators.**

### Integration
- `adapters.resolve()` imports `features.<x>`, then `features.<x>.api`, and returns None on
  ImportError/AttributeError only — a real exception inside a teammate's function propagates
  rather than being disguised as "not integrated".
- Per-section error boundary: a failure renders a named error inline; the rest of the page survives.

---

## Leakage prevention — be precise here

`task_sessions.csv` holds both context and the four `actual_*` targets. The split happens **in the
loader**, not at point of use:

- `pre_task_context()` removes the four outcome columns
- `outcome()` returns only them
- `build_session_context()` reads solely from the pre-task side, and constructs the context from
  named fields — so even if the filter failed, outcomes cannot ride along
- The model card lists `forbidden_inputs` explicitly
- Tests assert no `actual_*` string appears anywhere in a rendered `SessionContext`

---

## Hard questions — ranked by how much they'd hurt

**1. "Your models are trained on data you generated. Aren't they just learning your generator?"**
This is the sharpest question available. Answer it honestly:
> "Partly, yes. The generator has a known structure and the model can recover it, so the 74%
> improvement over baseline demonstrates the pipeline works end to end — it is not evidence the
> model would transfer to real telemetry. What transfers is the architecture: the leakage
> discipline, quantile outputs, and the decision logic sitting on top. We'd expect to retrain
> from scratch on real fleet data and would not carry these weights across."

**2. "What's your P10–P90 coverage?"**
> "64% and 67%, where calibrated is 80%. The bands are too narrow — the model is overconfident.
> With 4,000 training rows the tails are under-learned. We'd widen the alphas or recalibrate
> before trusting the interval operationally. The point estimate is solid; the interval isn't yet."

**3. "Greedy isn't optimal. Why not a proper solver?"**
> "Correct, it's a bounded local search — top 120 open tasks by priority and deadline, filtered to
> the machine type, then greedy selection. The architecture explicitly said not to spend the time
> on a solver. For a shift of five to eight tasks the gap to optimal is small, and every step's
> reasoning is inspectable, which matters more here than the last few percent."

**4. "How do you know the worker is in the swing path?"**
> "We approximate it: machine state is swinging and the worker is within a 12 m radius. It's not a
> true arc computation — we don't have heading-relative bearing in the telemetry. That's a
> documented simplification, not a claim."

**5. "Is the LLM in the safety path?"**
> "No. The rule decides; the decision object is copied unchanged and `decision_changed` is always
> False. CRITICAL is never sent to the model at all — deliberately, so there's no added latency on
> an emergency. There's also a rejected-phrases list: explanation text containing 'safe to
> continue', 'false alarm' or 'downgrade' is discarded as contradicting the rule."

**6. "Your face threshold is 0.70 — how was it set?"**
> "Calibrated on the accept side only: enrolled frames score 0.88–0.99 leave-one-out. The reject
> side is untested because only one face is enrolled — we have no cross-person distribution. We'd
> set it between the two distributions before trusting it."

**7. "Why Ridge for behaviour when you have LightGBM?"**
> "Different job. Prediction wants accuracy; behaviour wants an expectation you can subtract from
> and explain. A linear model's contribution to the residual is defensible to an operator; a
> boosted ensemble's isn't."

**8. "Two-thirds of a given operator's sessions show authorization refused. Why?"**
> "The generator assigns operators to machines without checking `authorized_machine_types`. The
> system catching it is correct behaviour; the frequency is a data artifact and we'd fix the
> generator, not the check."

---

## Weaknesses to volunteer before they're found

1. **Interval calibration** — 64% coverage vs 80% expected.
2. **Synthetic data** — every row flagged; no real-world validation.
3. **Swing envelope is a radius**, not an arc.
4. **Greedy optimizer** — no optimality guarantee.
5. **Peer-learning similarity score** is synthetic and its derivation is undefined. Softest surface.
6. **Face rejection untested** against a different person.
7. **No time-based history window** in the training gate — it's the last N occurrences, so a stale
   history is bounded by count, not by age.

Volunteering these reads as engineering maturity. Being caught by them reads as the opposite.
