# Implementation Decisions

Decisions that are not explicitly specified in the architecture document are recorded here.

---

## D001 — Project root directory

**Decision:** Project lives at `/CATERPILLAR/` (the working directory), not a nested `cat-operator-copilot/` subdirectory.

**Reason:** Simplifies paths for a hackathon; all teammates clone the same root.

---

## D002 — Python dataclasses for shared schemas

**Decision:** `shared/schemas.py` uses `@dataclass` rather than Pydantic models.

**Reason:** No extra dependency for the MVP. If validation becomes important, migrate to Pydantic.

---

## D003 — Synthetic data seed

**Decision:** `SEED = 42` as specified in the architecture document.

**Reason:** Reproducibility; same dataset every time unless seed is intentionally changed.

---

## D004 — Face recognition library

**Decision:** DeepFace + ArcFace backend.

**Reason:** Architecture document recommends this. Do not train a custom face model.

**Fallback:** If DeepFace installation fails, use the simplest stable local pretrained option available.

---

## D005 — Optimization approach

**Decision:** Greedy/priority-based search for the MVP (not a full OR-Tools solver).

**Reason:** Architecture document explicitly says do not spend most of the hackathon on a sophisticated solver.

---

## D006 — P10/P50/P90 uncertainty

**Decision:** Approximate percentiles using quantile regression or ± residual std from the gradient boosting model.

**Reason:** Native quantile regression in LightGBM/XGBoost is straightforward and avoids extra infrastructure.

---

_Add new decisions here as they arise. Do not silently make assumptions._

---

## D007 — Safety Guardian thresholds (fatigue, slope, equipment, weather, fuel)

**Decision:** All thresholds live in `config/safety_rules.yaml`; rules are declarative (`rules:` list) and reference thresholds by dotted path. Fatigue = hours into shift (`shift_elapsed_min / 60`): >10 h CRITICAL, >8 h HIGH. Unsafe slope ≥ 12° on an active machine is CRITICAL (10–12° MEDIUM, 8–10° LOW). Equipment malfunction = engine or hydraulic health score < 0.60. Low fuel < 25 % is HIGH.

**Reason:** The spec named the categories but not the numbers. Slope and health limits are conservative placeholders sized to the synthetic data range (slope 1.4°–14.9°, health 0.69–1.0) and should be reviewed against real CAT machine specs.

---

## D008 — Rules are more conservative than logged safety_events labels

**Decision:** The YAML (not the synthetic labels) is authoritative. On `safety_events.csv` the rules agree with 72 % of labels and never assign a lower severity. Differences: seatbelt-off-while-moving is CRITICAL (YAML) vs. HIGH (labels); worker ≤ 10 m of an active machine is HIGH vs. MEDIUM; worker ≤ 20 m of a travelling machine is MEDIUM vs. LOW.

**Reason:** Under-calling a hazard is worse than over-calling. `tests/test_safety_guardian.py` enforces the never-under-call property.

---

## D009 — SafetyEvent gains optional fields

**Decision:** `shared/schemas.SafetyEvent` gained `recommendation`, `session_id`, `operator_id`, `machine_id`, `rule_id`, `evidence`, all with defaults.

**Reason:** Downstream (Attention Manager, dashboard) needs the recommendation text and traceability to the rule. Defaults keep existing constructors valid.

---

## D010 — Optimizer objective is a weighted sum, CRITICAL safety is a hard block

**Decision:** Cost = time + fuel + deadline risk + transition + safety risk (weights in `config/settings.yaml` `optimization`), divided by a priority factor to rank. Any CRITICAL Safety Guardian finding makes a candidate infeasible rather than expensive. Plans stop at the 8 h fatigue budget from `safety_rules.yaml`.

**Reason:** A weighted sum keeps the API-contract cost breakdown additive and explainable. A product (time × fuel × risk) collapses to zero when any factor is zero and hides which factor drove the choice. Safety must never be traded off against cost.

---

## D011 — Attention Manager policy

**Decision:** CRITICAL safety always shows; HIGH safety with a stop action shows immediately; other safety events show in a safe state or queue; LOW safety is bundled; INFO is suppressed. Replanning, behavior and training show only in a safe state with nothing more urgent queued. Buddy is suppressed outside safe states. `release()` surfaces one queued event at a time plus a digest of bundled events.

**Reason:** One attention owner; minimise operator distraction while machine is active.
