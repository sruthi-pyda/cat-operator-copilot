# Dashboard Walkthrough — exactly what to do and say

All seven features, in order, with the click and the line for each.
Sidebar **Jump to a scenario** drives everything — never hunt for a session ID live.

---

## Setup (before anyone is watching)

**Terminal 1** — start the dashboard, wait for it to finish loading once:
```
cd C:\Users\hp\Desktop\projects\cat-operator-copilot
.venv\Scripts\python.exe -m streamlit run app/ui/dashboard.py --server.port 8502
```
Open `http://localhost:8502`. **First load is ~25 seconds.** Click every tab once so
nothing is cold. Then return to the top.

**Terminal 2** — keep this ready for the login:
```
cd C:\Users\hp\Desktop\projects\cat-operator-copilot
$env:PYTHONIOENCODING="utf-8"
```

---

## STEP 1 · Login — Feature 1, Operator Passport (part 1)

**Do:** In Terminal 2, run:
```
.venv\Scripts\python.exe -m features.passport.registration login --countdown 12
```
Look at the camera. Wait for the result line.

**Say:**
> "The operator signs in with their face. Recognition runs entirely on this laptop — we store
> a mathematical embedding, never a photograph, and nothing is uploaded anywhere. If confidence
> falls below the configured threshold, no session is created at all."

**Point at the output:** `{'authenticated': True, 'operator_id': 'OP1001', 'confidence': 0.9x}`

> "Recognised as OP1001. But being recognised is not the same as being cleared to operate —
> those are two separate checks, and I'll show you the second one failing shortly."

---

## STEP 2 · The shift screen — Feature 1, Operator Passport (part 2)

**Do:** Switch to the browser. Scenario dropdown is already on
**`Authorization — Cleared to operate`**. Point at the top strip.

**Say:**
> "Here's the Passport loaded into the shift screen: the operator, the machine — a hauler —
> the task, the session, and the authorization result. The Passport checks three things
> before a session opens: is this operator rated for this machine type, is their certificate
> valid, and has it expired."

> "Everything else on this page is built on that session context. One operator, one machine,
> one task, shared by every feature."

*(If asked about skills or baselines: "They're loaded into the session context and used by the
behaviour model — you'll see the operator's personal baseline drive the coaching decision
later. We surface the decision rather than the raw profile.")*

---

## STEP 3 · Safety Guardian — Feature 2 · **the centrepiece**

**Do:** Scenario → **`Safety — CRITICAL — worker in the swing path`**.
Let the red banner sit for two seconds before speaking.

**Say:**
> "Worker at 10.4 metres. Closing speed 1.87 metres a second. Machine swinging.
> Critical — emergency stop."

*(Same operator you logged in as. The plan is empty on this one because they're eight hours
into the shift — if anyone notices, that's the fatigue rule, and it's worth saying: "no further
tasks are assigned to this operator today.")*

**Do:** Scenario → **`Safety — INFO — a nearby worker that is not a hazard`**.
Scroll to Live operation → Event log.

**Say:**
> "Now a worker at a comparable distance — but the machine is stationary and they're outside
> the movement envelope. The system logs it and says **no action required**."

> "Same proximity. Opposite answer. A distance-threshold alarm would have fired on both, and
> by the end of the week the operator would ignore it. What decides this is motion, swing path
> and closing speed together — and it's all deterministic rules in a YAML file you can audit.
> No model is involved in that call."

---

## STEP 4 · Authorization refusing — Feature 1 closing the loop

**Do:** Scenario → **`Authorization — Refused — certification lapsed`**.

**Say:**
> "Same system, different operator. The face was recognised, but the certificate expired —
> so no session opens at all."

**Do:** Scenario → **`Authorization — Refused — not rated for this machine`**.

**Say:**
> "And here: valid certificate, wrong machine type. Two different refusals for two different
> reasons — not one hard-coded check."

---

## STEP 5 · Optimal Task Sequencing — Feature 5

**Do:** Scenario → **`Planning — Full shift plan, every deadline met`**. Point at the Plan panel.

**Say:**
> "The optimizer has sequenced the shift — five tasks, every deadline met, with the current
> one marked NOW. Next to it, that task in detail: priority, deadline, start time, predicted
> duration and fuel, and the model's confidence."

> "The ordering balances time, fuel, deadline risk, travel between sites and safety risk. Not
> fuel alone — that's the easy objective and the wrong one."

**Do:** Expand **"Excluded from the plan"**.

**Say:**
> "More interesting is what it refused. Four tasks blocked on a critical safety finding —
> unsafe ground slope. Others blocked by the eight-hour fatigue budget. One waiting on an
> unfinished dependency."

> "Safety and fatigue are hard blocks. The optimizer cannot trade them away against time or
> fuel, however cheap that would be. That's the difference between a constraint and a weight."

---

## STEP 6 · Predictive Task Intelligence — Feature 4

**Do:** Same scenario. Point at the Prediction panel on the right.

**Say:**
> "Duration and fuel as ranges — P10 to P90, with the P50 marked. Held-out error is 8.2
> minutes on duration and 3.6 litres on fuel, measured against a naive same-task-type baseline."

> "A single number hides the risk. The band is what tells you whether you can commit to a
> deadline."

*(If asked about the contributing factors list: "Those are contributions to the prediction.
Not causes.")*

---

## STEP 7 · Behavioral Fingerprint — Feature 3

**Do:** Scenario → **`Safety — HIGH — closing motion inside the envelope`**.
Scroll to Live operation → **Behavioural fingerprint**.

**Say:**
> "Observed idle ratio 0.255. Expected, under *these* conditions, 0.160. The model builds the
> expectation from the machine, the site, the weather and the task — then splits the gap into
> the part linked to the operator and the part the context already explains."

> "We never say the operator performed badly. We say observed behaviour differs from what was
> expected under this operating context. That's the only claim the data supports."

**Do:** Drag the **Replay position** slider through the shift.

**Say:**
> "This is the recorded shift replayed. Watch the Buddy field change as the machine moves
> between idle and active work — that verdict is computed live from each telemetry row."

---

## STEP 8 · Attention Manager — the cross-cutting component

**Do:** **Attention queue** tab.

**Say:**
> "Every subsystem can propose something. Only the Attention Manager decides what actually
> reaches the operator. Each candidate carries its severity, urgency, how actionable it is,
> and the operator's current state."

> "Training and Buddy are always deferrable and only actionable once stopped — so a lesson can
> wait a whole shift rather than interrupting a swing. Critical safety always wins."

---

## STEP 9 · Operator Training Hub — Feature 6

**Do:** **Training Hub** tab → **Gate**.

**Say:**
> "This operator did trigger coaching — seven of the last ten sessions qualified, confidence
> 0.75, and a lesson was assigned."

**Do:** Expand **per-session detail**.

**Say:**
> "But look at the gate. An occurrence only counts if it's repeated, confident, operator-linked,
> not explained by context — and not in the operator's favour either. All four conditions.
> Across thirty operators this fires for five. It discriminates rather than rubber-stamps."

> "An operator working hard ground never gets coached for the ground."

*(Optional contrast — scenario `Behaviour & Training — No coaching — a different operator, gate
holds`. Say plainly that this is a different operator: "Same gate, different operator, and here
it declines to coach." Skip it if you're short on time.)*

**Do:** **Lesson** tab → scroll the lesson → answer two quiz questions → **Submit answers**.

**Say:**
> "Lesson, scenario, quiz — then a measured follow-up. Passing the quiz doesn't close the
> issue. Only the metric actually moving does."

**Do:** **Peer techniques** tab.

**Say:**
> "And how other operators handled comparable work — anonymised, approved examples only, with
> the similarity score shown so you can judge whether it transfers."

---

## STEP 10 · Grounded Buddy — Feature 7

**Do:** Scenario → **`Buddy — Buddy answers at safe idle`** → **Operating Buddy** tab.
Confirm it says "Buddy enabled".

**Do:** Type **`what is my next task`** → Ask.

**Say:**
> "It answers from evidence and cites the source. It doesn't generate operating advice — it
> repeats what a trusted source already says."

**Do:** Type **`is it safe to swing right now`** → Ask.

**Say:**
> "A safety question can only be answered from a Safety Guardian incident or an approved
> manual. Everything else defers."

**Do:** Change **Machine state** to **swinging**.

**Say:**
> "Blocked — machine state not safe, attachment moving, arm in motion. The Buddy is
> unavailable while the machine is working. Not merely 'not driving' — genuinely idle, with
> nothing moving."

---

## STEP 11 · End of shift + close

**Do:** **End of shift** tab.

**Say:**
> "And afterwards, predicted against actual. The honest test isn't how close the P50 was —
> it's whether the actual landed inside the P10–P90 band."

**Close:**
> "Everything here runs on synthetic data, and every row is flagged as synthetic — we're not
> claiming access to real fleet telemetry. What's real is the decision logic: when to
> interrupt, when to stay quiet, when to refuse a task, and when to admit the model doesn't
> know. That's the part that would still be right if you swapped our data for yours."

---

## Do NOT

- Don't ask the Buddy **"how do I refuel"** — known cosmetic issue, it prefixes "sources disagree".
- Don't browse sessions manually — several produce an empty plan because the operator is past
  the fatigue budget.
- Don't debug live. If a section errors it shows a named error and the rest of the page keeps
  working — say that's deliberate and move on.
