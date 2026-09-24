# Demo Script — CAT Operator Copilot

Demo: 24 Sept 2026, 11:00 IST · 10–15 min live + 2 min selling points.
Every number below was verified on the integrated system. Nothing here is aspirational.

---

## Pre-flight (do this by 10:30, not 10:55)

- [ ] **Merge `develop` → `main` and tag** (Task 3.3, Aneesha):
      `git checkout main && git merge origin/develop --no-ff -m "Phase 3: merge develop into main for the demo" && git push origin main`
      `git tag -a v1.0-hackathon -m "CAT Operator Copilot v1.0" && git push origin v1.0-hackathon`
- [ ] `git checkout main && git pull` on the demo laptop
- [ ] `.venv\Scripts\python.exe -m pytest -q` → expect **358 passed, 1 skipped**
- [ ] **Re-register the face in the demo room's lighting** —
      `$env:PYTHONIOENCODING="utf-8"; .venv\Scripts\python.exe -m features.passport.registration register --operator-id OP1001 --frames 8 --countdown 15`
      Then test login twice. Lighting at login is what decides success, not enrolment quality.
- [ ] Start the dashboard: `.venv\Scripts\python.exe -m streamlit run app/ui/dashboard.py --server.port 8502`
- [ ] Set sidebar to **operator OP1001**, **session S000689**, and leave it there
- [ ] Close other Streamlit instances (port clashes)
- [ ] Optional: start Ollama. Everything works without it — Buddy answers are quoted instead of conversational.

**Operator IDs (changed — see D040):** Saanvi = OP1001, Sruthi = OP1002, Aneesha = OP1003.
Only OP1001 has a registered face, and only OP1001's training loop fires.

---

## The demo session

**S000689** — the only session that has everything at once:

| | |
|---|---|
| operator / machine | OP1001 on EXC010 (hauler) — **authorized** |
| shift elapsed | 57 min (leaves budget, so the plan is full) |
| plan | **5 steps**, 11 tasks excluded with real reasons |
| safety | **HIGH** at 07:10:46 — worker 6.0 m, closing 0.843 m/s, swinging |

Backup for the authorization-refused beat: **S000158** (OP1001 on an excavator → refused).

---

## Run of show

### 1. Login — biometric (Saanvi) · 1 min
Run the login command. Face → `{authenticated: true, operator_id: OP1001, confidence: 0.xx}`.

> "Recognition runs entirely on this laptop. No image leaves the machine — we store a face
> embedding, never a photo, and the folder is gitignored. Below the configured threshold, no
> session is created at all."

**Point out:** identity and authorization are separate. Being recognised doesn't mean being cleared.

### 2. Shift status + Safety (Saanvi) · 1 min
Top strip: operator, machine, task, AUTHORIZED, safety badge.

> "Safety sits above everything else on this screen, and it's quiet — one green line — until
> something earns attention."

Then the HIGH banner: worker 6.0 m, closing 0.843 m/s, swinging, action `stop_or_safe_action`.

> "This is the point of the whole safety design. Distance did not decide this. Elsewhere in the
> same dataset a worker at 6.5 m with a stationary machine is logged INFO and needs no action.
> What made this HIGH is motion, swing path and closing speed together."

### 3. Plan + why (Aneesha) · 2 min
Ordered sequence with **NOW** on step 1, then the Next Task card: priority, deadline, start,
ETA, fuel, confidence.

> "The order is a weighted trade-off across time, fuel, deadline risk, travel and safety —
> not fuel alone."

**Open "Excluded from the plan (11)".** This is the strongest optimizer moment:

> "Four tasks refused outright on `CRITICAL safety: unsafe_ground_slope`. Others on the 8-hour
> fatigue budget, one waiting on a dependency. Safety and fatigue are hard blocks here — the
> optimizer cannot trade them away against time or fuel, however cheap that would be."

### 4. Prediction (Sruthi) · 1.5 min
The P10–P90 range bars.

> "Not a point estimate. A band, with the P50 marked and a confidence value. Held-out
> performance is **MAE 8.21 minutes for ETA and 3.55 L for fuel**, against a naive
> same-task-type median baseline."

**Say plainly:** feature importances are contributions to the prediction, not causes.

### 5. Live operation — replay (Saanvi) · 2 min
Drag the **Replay position** slider through the shift.

> "This is the recorded shift played forward. Watch the Buddy field: available at safe idle,
> blocked the moment the machine starts working. That verdict is evaluated live from each
> telemetry row, not replayed."

### 6. Behavioural fingerprint (Sruthi) · 1.5 min
Observed vs expected-under-context, with the gap split.

> "Observed idle 0.255 against 0.160 expected for these conditions. The model splits that gap
> into an operator-linked part and a part the site, machine and weather explain. We never say
> the operator was bad — we say observed behaviour differs from expectation under this context."

### 7. Training Hub — the closed loop (Saanvi) · 2 min
Gate tab, real decision:

> "**7 of 10 qualifying occurrences, lesson L003, confidence 0.749, first trigger.** That's the
> real Behavioral Fingerprint output, not a simulation."

Open the per-session detail.

> "An occurrence only counts if it's operator-linked, confident, not primarily explained by
> context, and not in the operator's favour — all four. An operator on hard ground never gets
> coached for the ground. Neither does one who beat expectation."

Then Lesson tab → take two quiz questions → show scoring.

> "Passing the quiz doesn't close the issue. Only a measured follow-up improvement does."

### 8. Buddy (Saanvi) · 1.5 min
Machine state `safe_idle` → "Buddy enabled".

Ask, in this order:
1. `what is my next task` → answers from the session, cited
2. `is it safe to swing right now` → answers from the approved manual

Then switch machine state to **swinging**:

> "Blocked: machine_state_not_safe, attachment_movement_active, arm_in_motion. The Buddy is not
> available while the machine is working — being merely 'not travelling' isn't enough."

**Do not ask** "how do I refuel" — known cosmetic issue, it prefixes "sources disagree".

### 9. End of shift (Sruthi) · 1 min
Predicted vs actual.

> "The honest test is whether the actual fell inside the P10–P90 band, not how close P50 was."

---

## Selling points (2 min)

1. **Deterministic, auditable safety.** Rules in YAML, not model output. The LLM can explain a
   safety event; it never decides one. CRITICAL is never sent to the LLM at all.
2. **Context beats single factors.** Proximity alone is not danger; idle alone is not a bad
   operator; the plan is not fuel-minimisation.
3. **Uncertainty, not point estimates.** P10/P50/P90 everywhere a prediction appears.
4. **Coaching that refuses to blame.** Four independent conditions before training fires, and it
   fires for 5 of 30 operators — it discriminates rather than rubber-stamps.
5. **Privacy-first biometrics.** Local embeddings, no images stored, nothing uploaded.
6. **Honest empty states.** Anything not integrated says so and names its owner. No placeholder
   numbers anywhere — a plausible ETA on screen is indistinguishable from a real one.

---

## Hard questions — honest answers

**"Is this real CAT data?"**
No. Everything is synthetic, seed 42, and every row carries `synthetic_flag`. It's labelled on
screen. We'd rather show a working decision chain on marked synthetic data than imply access we
don't have.

**"Why do so many sessions show authorization refused?"**
The generator assigns operators to machines without checking `authorized_machine_types`, so
about two-thirds of OP1001's sessions are on machines they aren't cleared for. The system
catching that is correct behaviour; the frequency is a data artifact, not a design claim.

**"Why is the plan empty on some sessions?"**
The fatigue rule. Those sessions are 8+ hours into an 8-hour budget, so the optimizer refuses
every task. We deliberately did not tune that away.

**"Is the LLM making safety decisions?"**
No. The rule decision is copied unchanged, `decision_changed` is always False, CRITICAL never
reaches the LLM, and Buddy synthesis is disabled for any safety-critical question. If the model
emits a number that isn't in the evidence, the answer is discarded and we quote instead.

**"How do you know the predictor isn't seeing the answer?"**
The four `actual_*` columns are split out at the loader, so a `SessionContext` physically cannot
carry them, and the prediction functions never read them. There's a test asserting it.

**"What's the peer-learning similarity score?"**
Synthetic, and its derivation isn't specified. It's the softest surface in the project — we show
the score next to every technique rather than hiding it, and only approved, anonymised examples
appear. (Don't oversell this one.)

**"Is the face threshold validated?"**
On the accept side, yes — well-lit frames score 0.88–0.99 against a 0.70 threshold. The reject
side is untested, because only one face is enrolled. Honest answer: we'd calibrate against a
cross-person distribution before trusting it.

---

## If something breaks

- A section fails → it shows a named error inline and **the rest of the page still renders**.
  Say so; it's deliberate. Keep going.
- Face login fails → it's the lighting. Move on and use the sidebar operator selector.
- Ollama down → Buddy still answers, quoted from evidence. Nothing to explain.
