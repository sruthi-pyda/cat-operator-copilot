# Member 3 (Saanvi / OP1003) — Build Plan

Owns: Operator Passport + biometric, Training Hub, Grounded Buddy, Task Planning Dashboard, UI integration.
Branch: `feature/saanvi-passport-ui`.

Status key: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Step 0 — Environment (highest risk, do first)

- [x] Project-local `.venv` (Python 3.13)
- [x] Install base deps + Streamlit
- [x] Attempt DeepFace install; verify import and a single embedding call — **works** (TF 2.21, NumPy 2.5.3, ArcFace 512-dim). See D013.
- [x] No fallback needed; `PYTHONIOENCODING=utf-8` required on Windows (D014)
- [ ] Verify webcam capture works — **needs the user to run it**, it opens the laptop camera

**Why first:** DeepFace pulls TensorFlow, the one dependency that can consume hours. Fail fast.
**Why a venv:** TensorFlow commonly pins `numpy<2`; a global install would break other projects on this machine.

## Step 1 — Passport core (no biometric, no synthetic data needed)

- [x] `features/passport/repository.py` — load operators/machines, fixture fallback
- [x] `features/passport/authorization.py` — machine-type authorization + certification validity
- [x] `features/passport/passport.py` — `load_passport()`, `create_session()` → `SessionContext`
- [x] Tests: unknown operator, unauthorized machine type, expired certification, happy path (24 tests, mutation-checked)

Passport supplies context to other features. It does **not** produce a standalone operator score.

## Step 2 — Biometric

- [x] `features/passport/biometric.py` — `register_operator()`, `identify()`
- [x] Registration: webcam frames → embedding under `data/face_registrations/` (gitignored); CSV stores the *path*, never the image
- [x] Login returns `{authenticated, operator_id, confidence}`; threshold from `config/settings.yaml`
- [x] Below threshold → no authorized session is created
- [x] Tests mock the recognition backend — no test needs a webcam
- [x] `features/passport/registration.py` — webcam CLI (`register` / `login`)
- [ ] **Needs the three of you present:** run `register` for OP1001, OP1002, OP1003 and calibrate the threshold against the real scores

## Step 3 — Training Hub

- [x] `features/training/trigger.py` — gate: repeated AND confidence ≥ threshold AND not primarily context-explained (D019)
- [x] `features/training/content/` — 4 lessons, 4 quizzes, 3 scenarios as YAML
- [x] `features/training/progress.py` — quiz scoring, before/after metric, escalation state
- [x] Tests: 50 training tests; mutation-verified that removing the context-share check breaks "hard site does not get the operator coached"

## Step 4 — Grounded Buddy

- [x] `features/buddy/safe_state.py` — parked OR verified safe idle, **and** no arm/bucket/attachment movement
- [x] `features/buddy/evidence.py` — retrieval over approved sources only; every item tagged source / timestamp / freshness / synthetic_flag / confidence / source_agreement
- [x] `features/buddy/conflict.py` — detect conflict → authoritative source → else defer
- [x] `features/buddy/buddy.py` — question → safe-state gate → retrieve → authority → freshness → conflict → answer or safe deferral
- [x] Tests 4, 5, 6 from the instruction doc (31 tests)
- [ ] Safety-critical uncertainty defers to Safety Guardian / approved manual. The Buddy never makes a safety decision and never invents an operating instruction.

## Step 5 — Task Planning Dashboard

- [x] `features/dashboard/adapters.py` — ports for `predict_task`, `evaluate_safety`, `analyze_behavior`, `generate_plan`, `route_event`; returns an explicit *unavailable* state when a teammate's module is absent (never fabricated numbers)
- [x] `features/dashboard/data.py` — loads the tables and enforces the pre-task / outcome split
- [x] `app/ui/dashboard.py` — Streamlit single shift home screen, verified in a browser
- [x] Dashboard calls the shared APIs; it does not import teammates' internal model logic

Run it:

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m streamlit run app/ui/dashboard.py --server.port 8502
```

Port 8502, not Streamlit's default 8501 — the data_repair project's app already uses 8501 on this machine (D029).

## Step 6 — Integration

- [ ] Wire real teammate modules in place of stubs
- [ ] Telemetry replay drives the dashboard
- [ ] End-to-end demo run

---

## Standing constraints for this slice

- No ground-truth leakage: actuals are only ever shown as *outcomes* next to predictions, never fed into a prediction input.
- Safety is deterministic and owned by Member 2. The Buddy may explain a safety event; it can never override or replace the decision.
- All displayed data carries its `synthetic_flag`; synthetic values are never presented as real CAT telemetry.
- Cross-feature calls go through the documented schemas in `docs/API_CONTRACT.md`.
