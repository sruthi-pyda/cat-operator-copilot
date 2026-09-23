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

## D007 — BehaviorResult unit contract and context_share helper

**Decision:** `operator_residual` and `context_explained_component` in `BehaviorResult` are both
**idle ratios** (dimensionless, 0–1). They are set together in a single return statement so one
can never be populated without the other. A `context_share()` method is provided on `BehaviorResult`
for the Training Hub gate so teammates don't reimplement the ratio math independently.

**Reason:** Raised by Team Member 2 (Aneesha). The failure mode is a silent-zero:
if `context_explained_component` defaults to 0 while `operator_residual` is non-zero, the gate
formula `|context| / (|operator| + |context|)` = 0, meaning context is never credited,
and operators working difficult ground get incorrectly coached. A runtime warning is emitted if
`context_explained == 0.0` and `operator_residual > threshold` so the bug would surface in testing.

**Training gate rule (owned by Team Member 2):**
```python
if result.context_share() > 0.5:
    # context explains more than half the gap — do NOT trigger coaching
```

---

_Add new decisions here as they arise. Do not silently make assumptions._
