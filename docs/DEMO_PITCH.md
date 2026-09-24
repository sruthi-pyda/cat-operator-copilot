# The Pitch — CAT Operator Copilot

Companion to `DEMO_SCRIPT.md`. That file is the checklist; this one is what you **say**.

Read the opening and closing out loud once before you present. Everything else you can
improvise around, but those two decide how the whole thing lands.

---

## The one-sentence version

> "An operator copilot that decides *what to tell you, when* — using the full operating
> context rather than any single number, and refusing to let a language model anywhere near
> a safety decision."

## The 30-second version

> "Most equipment software throws alerts at operators. Proximity alarm. Idle alarm. Low fuel.
> Operators learn to ignore them, because most are wrong.
>
> We built the opposite. Every decision here uses the whole context — the machine, the site,
> the weather, the task, the operator's own history. A worker ten metres away with the machine
> parked is logged and ignored. A worker ten metres away with the boom swinging toward them is
> critical. Same distance, opposite answer.
>
> And the parts that must never be wrong — safety, authorization — are deterministic rules you
> can read in a config file. The AI explains them. It never decides them."

---

## Why this framing wins

Do not lead with "we used LightGBM and DeepFace." Every team has models. Lead with
**judgement**: the system knows what *not* to say. That is the hard part and almost nobody
demonstrates it.

Three claims to land, in priority order:

1. **Context over single factors.** Distance alone isn't danger. Idle alone isn't a bad
   operator. Speed alone isn't efficiency.
2. **Deterministic where it matters.** Safety and authorization are rules, auditable, in YAML.
   The LLM is downstream of the decision, never upstream.
3. **Honest under uncertainty.** Ranges not point estimates. Empty states that say "not
   integrated" instead of showing a plausible fake number. A coaching gate that mostly says no.

---

## Run of show — what to open, what to say

Open the dashboard at `http://localhost:8502`. It opens on **OP1001 / S000146**.
Use the sidebar **Jump to a scenario** dropdown to move between beats — never hunt for a
session ID live.

### Beat 0 · Face login (terminal) · 45 sec

Run the login command before switching to the browser.

> "The operator logs in with their face. Recognition runs entirely on this laptop — we store a
> mathematical embedding, never a photo, and nothing is uploaded. Below the confidence
> threshold, no session is created at all."

**The line that matters:**
> "Being recognised is not the same as being cleared. Those are two separate checks, and I'll
> show you the second one failing in a moment."

---

### Beat 1 · The shift screen · 45 sec
*Scenario: `Authorization — Cleared to operate` (opens here by default)*

> "This is the whole shift on one screen. Who's operating, on what machine, cleared or not,
> and the current safety state — before anything else."

Point at the safety line: a single quiet green **CLEAR**.

> "Safety sits above everything, and right now it's one quiet line. That's deliberate. A
> system that shouts constantly gets ignored."

---

### Beat 2 · The plan, and why · 90 sec
*Same scenario*

> "The optimizer has sequenced this operator's shift: five tasks, every deadline met. The
> order balances time, fuel, deadline risk, travel between sites and safety — not fuel alone."

**Open "Excluded from the plan".** This is your strongest optimizer moment.

> "More interesting is what it refused. Four tasks blocked on a critical safety finding —
> unsafe ground slope. Others blocked by the eight-hour fatigue budget. One waiting on a
> dependency.
>
> Safety and fatigue are **hard blocks** here. The optimizer cannot trade them away against
> time or fuel, however cheap that would be. That's the difference between a constraint and a
> weight."

---

### Beat 3 · Prediction with uncertainty · 45 sec
*Same scenario*

> "Duration and fuel, as ranges. P10 to P90 with the P50 marked. Held-out error is 8.2 minutes
> on duration, 3.6 litres on fuel, against a naive same-task-type baseline.
>
> A single number would hide the risk. The band is the useful part — it's what tells you
> whether to commit to a deadline."

If asked about the factor list: *"Those are contributions to the prediction. Not causes."*

---

### Beat 4 · Safety, in context · 90 sec — **the centrepiece**
*Scenario: `Safety — CRITICAL — worker in the swing path`*

Let the red banner land before speaking.

> "Worker at 9.1 metres. Closing speed 2.05 metres per second. Machine swinging. Critical —
> emergency stop."

Now switch to *`Safety — INFO — a nearby worker that is not a hazard`*.

> "And here — a worker at a comparable distance, machine stationary, outside the movement
> envelope. The system logs it and says **no action required**.
>
> Same proximity. Opposite answer. A distance-threshold alarm would have fired on both, and
> the operator would have learned to ignore it by the end of the week. Motion, swing path and
> closing speed together decide this — and it's all rules in a YAML file you can audit."

**If asked "is that AI?"** — *"No. That's deterministic. A language model can explain the
event afterwards, but it never makes or changes the call, and critical events never reach it."*

---

### Beat 5 · Authorization refusing · 45 sec
*Scenario: `Authorization — Refused — certification lapsed`*

> "Same system, different operator. Face recognised — but the certificate expired, so no
> session opens."

Then *`Authorization — Refused — not rated for this machine`*.

> "And here, a valid certificate but the wrong machine type. Two different refusals, two
> different reasons. Not one hard-coded check."

---

### Beat 6 · Behaviour without blame · 60 sec
*Scenario: `Safety — HIGH — closing motion inside the envelope` (OP1001)*

Scroll to the behavioural fingerprint.

> "Observed idle 0.255. Expected, for *these* conditions, 0.160. The model splits that gap
> into the part linked to the operator and the part explained by the site, the machine and the
> weather.
>
> We never say the operator was bad. We say observed behaviour differs from what was expected
> under this context — which is the only thing the data actually supports."

---

### Beat 7 · Coaching that mostly says no · 75 sec
*Same scenario → Training Hub tab → Gate*

> "This operator did trigger coaching: seven of ten recent sessions qualified, confidence
> 0.75, lesson assigned."

Open the per-session detail.

> "But look at the gate. An occurrence only counts if it's repeated, confident, operator-linked,
> and **not** explained by context — and not in the operator's favour either. All four.
>
> Across thirty operators this fires for five. It discriminates rather than rubber-stamps. An
> operator working hard ground never gets coached for the ground."

Then Lesson tab, answer two quiz questions, submit.

> "Lesson, quiz, and then a measured follow-up. Passing the quiz doesn't close the issue — only
> the metric moving does."

---

### Beat 8 · The Buddy, and when it shuts up · 60 sec
*Scenario: `Buddy — Buddy answers at safe idle`*

Ask **"what is my next task"** → answers, cited.

> "It answers from evidence and cites the source. It doesn't generate operating advice."

Ask **"is it safe to swing right now"** → answers from the approved manual.

Now change machine state to **swinging**.

> "Blocked — machine state not safe, attachment moving, arm in motion. The Buddy is unavailable
> while the machine is working. Not merely 'not driving' — actually idle, with nothing moving."

---

### Beat 9 · Close · 30 sec

> "Everything you've seen runs on synthetic data, and every row is flagged as synthetic —
> we're not claiming access to real fleet telemetry.
>
> What's real is the decision logic: when to interrupt, when to stay quiet, when to refuse a
> task, and when to admit the model doesn't know. That's the part that would still be right
> if you swapped our data for yours."

---

## Anticipated questions — short answers

**"How is this different from existing telematics?"**
Telematics reports. This decides — and more importantly decides what *not* to surface. The
Attention Manager ranks every candidate event against the operator's current state, so training
never interrupts a swing.

**"Why should I trust the safety logic?"**
Because you can read it. It's a YAML rule file, not a model. Every event names the rule that
fired and the required action.

**"Isn't the AI making safety calls?"**
No. The rule decides, the decision is copied unchanged, and critical events are never sent to
the model at all. Where it does phrase an answer, any number not present in the evidence causes
the answer to be discarded.

**"Is the data real?"**
No, and we say so on screen. Synthetic, seed 42, every row flagged.

**"What doesn't work yet?"**
The peer-learning similarity score is synthetic and its derivation isn't defined — it's the
softest part. Face recognition is calibrated on the accept side but we haven't tested rejection
against a cross-person distribution. We'd rather tell you that than let you find it.

---

## Delivery notes

- **Let the CRITICAL banner sit for two seconds before you talk.** The contrast with the INFO
  case is the strongest moment in the demo; don't rush it.
- **Say "refuses" and "blocks" out loud.** Judges remember a system that says no.
- **If something errors**, the page shows a named error and everything else keeps rendering —
  say that's deliberate and move on. Don't debug live.
- **Don't oversell peer learning.** If asked, be straight that it's the weakest surface.
- The honest answers land better than polished ones. This build's real strength is that it
  knows what it doesn't know.
