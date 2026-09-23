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

## D007 — Project-local virtual environment

**Decision:** Dependencies install into `.venv/` at the project root (gitignored), not into the global interpreter.

**Reason:** This machine runs Python 3.13 with NumPy 2.3 used by other projects. DeepFace pulls TensorFlow, which commonly pins `numpy<2`; a global install could break unrelated work. Isolation also makes the team's installs reproducible.

**How to use:** `.venv/Scripts/python.exe -m pytest` on Windows, `.venv/bin/python -m pytest` elsewhere.

---

## D008 — `shared/config.py` settings loader

**Decision:** Added `shared/config.py` with `load_settings()`, `load_yaml()` and `resolve_path()`.

**Reason:** Passport, Buddy and Dashboard all need `config/settings.yaml`, and thresholds must stay out of Python per the architecture. Purely additive — it defines no new schema and renames no field, so the API contract is unchanged.

---

## D009 — `authorized_machine_types` delimiter

**Superseded 2026-09-23 — no action needed.** The original decision asked Member 1 to emit `|`, on the assumption that a comma would collide with CSV parsing.

Checking the generated file, `authorized_machine_types` is written as a *quoted* field (`"hauler,grader,loader"`). That is valid CSV: the quoting protects the internal commas, `csv.DictReader` returns the whole string, and the reader's comma fallback already splits it correctly.

**Current state:** the reader accepts `|`, `;` and `,` and strips a surrounding `[...]`, so it handles the generator's output as-is. The generator does not need to change.

---

## D010 — Absent certification expiry is not treated as expired

**Decision:** An empty `certification_expiry` passes the expiry check; the `certification_status` check still applies independently.

**Reason:** A missing date is missing data, not evidence of expiry. Refusing on it would turn a data-quality gap into a false authorization failure. Status remains the authoritative signal.

---

## D011 — Session ID generation

**Decision:** `create_session()` generates `S` + six digits derived from a UUID4 when no `session_id` is supplied; callers may pass an explicit ID.

**Reason:** Matches the `S000123` shape in the API contract without needing a shared counter. Collisions are possible in principle and acceptable for the MVP; the replay harness passes explicit IDs.

---

## D012 — Test fixtures precede the synthetic dataset

**Decision:** Passport tests read small hand-written CSVs in `tests/fixtures/` rather than `data/synthetic/`.

**Reason:** Lets Member 3 develop and test before the generator lands, and keeps tests fast, deterministic and independent of a regenerated dataset. Integration against the real dataset happens in Phase 3.

---

## D013 — DeepFace verified on Python 3.13; no fallback needed

**Decision:** Keep DeepFace + ArcFace (D004). The fallback path is not required.

**Evidence:** Installed and executed in the project venv on Python 3.13.4 — TensorFlow 2.21.0, NumPy 2.5.3, `DeepFace.represent(..., model_name="ArcFace")` returned a 512-dimension embedding. Notably TensorFlow 2.21 does **not** force a NumPy 1.x downgrade, so the venv keeps NumPy 2.x.

**Note:** `tf-keras` must be installed alongside DeepFace under Keras 3.

---

## D014 — UTF-8 console encoding required on Windows

**Decision:** Run Python with `PYTHONIOENCODING=utf-8` when DeepFace is involved.

**Reason:** DeepFace's logger prints emoji; the default Windows `cp1252` console encoding raises `UnicodeEncodeError` and kills the process. This is an environment quirk, not a code defect, so it is not worked around in application code.

---

## D015 — Buddy reads its safe-state list from `config/safety_rules.yaml`

**Decision:** `features/buddy/safe_state.py` reads `machine_state.safe_states` from the Safety Guardian's config rather than keeping its own list.

**Reason:** The architecture forbids the Buddy from making safety decisions. Letting it hold a private definition of "safe" would let the two drift apart. Safety owns the definition; the Buddy obeys it.

---

## D016 — Unknown machine state denies Buddy interaction

**Decision:** If `machine_state` or `attachment_movement` is missing, the safe-state gate refuses.

**Reason:** Absent telemetry is not evidence of safety. Deny-by-default is the only defensible direction for a gate that decides whether to distract an operator.

---

## D017 — Safety-critical questions may only be answered from safety sources

**Decision:** When a question is classified safety-critical, the Buddy discards every source except Safety Guardian incidents and approved manual snippets. If none remain, it defers to the Safety Guardian.

**Reason:** Enforces "the Buddy may explain a safety event but never makes the safety decision". Classification is keyword-based and deliberately broad — a false positive costs one deferral, a false negative would let the Buddy answer a safety question on its own authority.

---

## D018 — Buddy answers are quoted, not generated

**Decision:** `_compose_answer()` assembles the response from the `content` already carried by the winning evidence item, plus a source citation.

**Reason:** The architecture forbids the LLM inventing an operating instruction. Quoting rather than generating makes that structural instead of a prompt-level request.

---

## D019 — Training trigger gate

**Decision:** An occurrence counts only if all three hold: `attribution ∈ {operator-driven, mixed}`, `confidence >= min_confidence`, and `context_share <= max_context_share`, where `context_share = |context_explained_component| / (|operator_residual| + |context_explained_component|)`. A trigger fires when qualifying occurrences reach `min_occurrences` inside the recent window.

**Thresholds** (`training:` block of `config/settings.yaml`): window 10, min occurrences 2, min confidence 0.70, max context share 0.50, quiz pass 0.80, improvement target 10%, escalate after 2.

**Reason:** Implements "repeated AND confident AND not primarily context-explained". Critically, failures that break *different* checks never sum into a trigger — each occurrence must satisfy all three by itself.

---

## D020 — `check_training_trigger` signature amended

**Decision:** Takes `(behavior_history, operator_id, issue_type, prior_trigger_count=0)` rather than `behavior_history` alone. `docs/API_CONTRACT.md` is updated accordingly.

**Reason:** `BehaviorResult` carries neither `operator_id` nor `issue_type`, so the original signature was not implementable. `BehaviorResult` itself is unchanged, so no other feature is affected.

---

## D021 — `mixed` attribution is coachable

**Decision:** `mixed` occurrences can count toward a trigger; the context-share check decides. `attribution_type` on the emitted trigger is `operator-driven` only when every qualifying occurrence was.

**Reason:** Part of a mixed gap is operator-linked. Excluding `mixed` outright would make the gate nearly unreachable; the context-share test is the better instrument.

---

## D022 — A zero gap fails closed

**Decision:** When `operator_residual` and `context_explained_component` are both 0, `context_share` is 1.0, so the occurrence does not qualify.

**Reason:** No measured gap means no evidence against the operator. Failing closed keeps the benefit of the doubt with the operator.

---

## D023 — `resolved` added as a fourth escalation state

**Decision:** States are `first_trigger → repeat → escalated`, plus terminal `resolved`. Passing the quiz does **not** resolve an issue; only a follow-up measurement that meets the improvement target does.

**Reason:** The loop is closed by measured change, not by completing a lesson.

---

## D024 — A zero before-metric is reported as not comparable

**Decision:** `compare_metrics()` with `before == 0` returns `improvement_pct=0.0`, `comparable=False`, reason `baseline_metric_is_zero`.

**Reason:** There is no percentage improvement over a zero baseline. The dashboard shows "no baseline to compare" rather than a fabricated number or a crash.

---

## D025 — Integration requirement for the Behavioral Fingerprint (Member 1)

**Requirement, not a decision.** The training gate's operator protection depends on `BehaviorResult.context_explained_component` being **populated and on the same scale as `operator_residual`**.

If `context_explained_component` is left at its default of 0 while `operator_residual` is populated, `context_share` becomes 0.0 and the "difficult site" protection silently disappears — every repeated, confident anomaly would become coachable. This fails open, which is the wrong direction.

**Action for Member 1:** confirm both components are emitted in the same units. Flag to Member 3 if they are not.

---

## D026 — `attachment_movement` is a boolean in the generated telemetry

**Decision:** The Buddy's safe-state gate accepts both a boolean (`False` = not moving) and the worded form implied by the documented schema. `false`, `0`, `no` and `off` join the stationary vocabulary.

**Reason — this was a live defect.** The gate originally expected only words (`"none"`, `"idle"`). Against the real telemetry, `False` fell through as "not a known stationary value" and was read as *movement*, so the Buddy would have been blocked on all 75,000 rows and demo Test 5 would have failed. Verified after the fix: enabled on 20,959 / 20,959 `safe_idle` rows and blocked on 100% of `digging`, `loading`, `swinging`, `traveling` and `grading`.

---

## D027 — `bucket_state` is a configuration, not a motion

**Decision:** `MOVING_BUCKET_STATES` covers only unambiguous motion (`digging`, `dumping`, `curling`, `lifting`, `swinging`). `loading`, `closed` and `open` do not block.

**Reason:** The generated telemetry uses `closed` / `open` / `loading` to describe the bucket itself, and `loading` co-occurs with `attachment_movement=False` and `arm_speed=0` in a third of `safe_idle` rows. Treating a configuration as motion would block genuinely safe moments while adding no safety: actual motion is already carried by `attachment_movement` and `arm_speed`.

---

## D028 — Synthetic data is taken from Member 1's branch without merging

**Decision:** The dataset is copied into the working tree with a path-scoped checkout and left **untracked** on `feature/saanvi-passport-ui`:

```
git checkout origin/feature/sruthi-data-models -- data/synthetic/
git reset -q data/synthetic/
```

**Reason:** The team keeps branches separate. Merging would pull Member 1's code and history into this branch, and committing the data here would duplicate a 21 MB telemetry file and set up conflicts when both branches reach `develop`. The data stays owned by the branch that generates it. Re-run the two commands above to refresh after Member 1 regenerates.

**Consequence:** tests must not depend on `data/synthetic/`. They use `tests/fixtures/` instead (D012); findings from the real data are pinned as unit tests with the real vocabulary values.

---

## D031 — Expired demo certificates are intentional

**Resolved 2026-09-23 (was OPEN-01).** `OP1002`'s `certification_expiry` of 2025-09-04 is deliberate, not a data defect. The team wants an operator whose certificate has lapsed so the demo can show authorization **refusing** a session rather than only ever succeeding.

**Consequence:** `OP1002` authenticates by face but cannot open a session, and the dashboard shows the refusal with reason `certification_expired`. Demo the successful path with an operator whose certificate is current. The generator should not be "fixed".

---

## D029 — Dashboard runs on port 8502

**Decision:** The Streamlit dashboard uses port 8502, not Streamlit's default 8501.

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m streamlit run app/ui/dashboard.py --server.port 8502
```

**Reason:** An unrelated Streamlit app already occupies 8501 on the development machine. Teammates on a clean machine can use either port; 8502 is what `.claude/launch.json` and the docs specify so everyone sees the same thing.

---

## D030 — Unintegrated features render a notice, never a placeholder

**Decision:** When a teammate's feature is absent, the dashboard prints "Not integrated yet" with the owning member. It never shows a zero, a dash-with-units, or a sample number in a metric tile.

**Reason:** A plausible-looking ETA is indistinguishable from a real one on screen, and during a demo nobody checks. An empty state is honest; a placeholder is a lie that looks like a result.

---

_Add new decisions here as they arise. Do not silently make assumptions._
