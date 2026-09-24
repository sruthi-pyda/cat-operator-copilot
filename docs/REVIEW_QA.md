# Reviewer Q&A Bank

Every answer verified against the code. Where something is weak, the honest answer is given —
a reviewer who catches a hidden weakness discounts everything else you said.

---

## 1. Architecture

**Q. Walk me through the system.**
One `SessionContext` is built per session — operator, machine, task, site, traffic, weather,
temporal state, data quality. Every feature receives that same object and returns a documented
result. Seven features: Passport (identity + clearance), Safety Guardian, Behavioural Fingerprint,
Predictive Task Intelligence, Optimal Task Sequencing, Training Hub, Grounded Buddy. Plus a
cross-cutting Attention Manager that decides what actually reaches the operator. The dashboard is
the single shift screen that composes them.

**Q. How do the features talk to each other?**
Through documented function signatures in `docs/API_CONTRACT.md`, never by importing each other's
internals. The dashboard resolves each one by name at call time — `features.<x>` then
`features.<x>.api`. If a feature isn't present, it reports "not integrated" with the owner's name.

**Q. Why a shared context object rather than each feature loading what it needs?**
Three people built these in parallel. Without one normalised context you get three different names
for the same field and silent disagreements — one feature reading `machine_id`, another
`equipment_id`. It also means a feature can't quietly reach for data it shouldn't have, which is
how the leakage rule is enforced structurally rather than by discipline.

**Q. What happens if one feature crashes?**
Each dashboard section is wrapped in an error boundary. A failure renders a named error in place —
which feature, which exception — and every other section still renders. Streamlit would otherwise
halt the whole script and silently drop everything below the fault.

**Q. Why Streamlit?**
It's a demo surface, not a product. Streamlit gets a working interactive UI in Python without a
frontend stack, which is the right trade for a hackathon. For production this would be an API with
a real frontend; the features are already separated from the UI, so that swap is mechanical.

**Q. How would you scale this?**
The features are pure functions over a context object, so they parallelise naturally. The real
bottlenecks are the per-candidate model calls in the optimizer (already cached within a run) and
loading 75,000 telemetry rows per page (would become a query, not a CSV read).

---

## 2. Data

**Q. Where does the data come from?**
It's synthetic, generated with seed 42. 30 operators, 10 machines, 1,500 tasks, 5,000 sessions,
75,000 telemetry rows, ~1,500 safety events. Every row carries `synthetic_flag` and the dashboard
says so on screen. We have no access to real CAT fleet telemetry and don't claim any.

**Q. Is the synthetic data realistic?**
It's relational, not random — fuel rises with load and idle, duration rises with soil hardness,
slope, congestion and rain and falls with operator skill, safety events depend on motion plus
proximity plus closing speed rather than distance alone. But it is a model of reality written by
us, which is a real limitation (see the generator question below).

**Q. How do you prevent target leakage?**
`task_sessions.csv` contains both the pre-task context and four outcome columns
(`actual_task_duration_min`, `actual_fuel_used_l`, `actual_idle_time_min`, `actual_cycle_count`).
The split happens in the loader: `pre_task_context()` strips them, `outcome()` returns only them,
and `build_session_context()` reads solely from the pre-task side and pulls named fields — so even
if the filter failed, an outcome couldn't ride along. The model card lists `forbidden_inputs`
explicitly, and a test asserts no `actual_*` string appears in a rendered context.

**Q. What's your train/test split?**
80/20 with seed 42 — 4,000 training sessions, 1,000 held out. All reported metrics are on the
held-out set.

---

## 3. Prediction

**Q. What model, and why?**
LightGBM. Gradient boosting is the strongest family for tabular data at this scale — 18 mixed
numeric and categorical features, 5,000 rows, with interactions that matter (hard soil costs more
on a worn machine). The decisive reason for LightGBM specifically is a native quantile objective,
which we need for P10/P50/P90.

**Q. Explain the quantile regression.**
Not one model with error bars. Three models per target, each trained with an asymmetric quantile
loss: `alpha=0.1`, `0.5`, `0.9`. Each learns that percentile directly, so the band is learned rather
than inferred from residual spread. Six regressors total across ETA and fuel, 300 estimators each.

**Q. How accurate is it?**
ETA: MAE 8.36 minutes, RMSE 13.31, against a naive same-task-type-median baseline of 31.68 — a
73.6% improvement. Fuel: MAE 3.54 L against 12.04, a 70.6% improvement. Held out, 1,000 sessions.

**Q. What's your P10–P90 coverage?**
64% for ETA, 67% for fuel, where a calibrated interval would be about 80%. **The bands are too
narrow — the model is overconfident.** With 4,000 training rows the tails are under-learned. We'd
widen the alphas or apply conformal calibration before trusting the interval operationally. The
point estimate is sound; the interval isn't yet.

**Q. Is feature importance telling you causality?**
No, and the model card says so explicitly. It's predictive contribution. Visibility ranks highly
partly because it correlates with time of day and weather, not because reducing visibility causes
duration on its own.

**Q. Why not a neural network?**
5,000 rows. Deep learning needs far more data and doesn't beat gradient boosting on tabular
problems at this scale. It would also lose the native quantile objective.

**Q. How do you know it isn't overfitting?**
Held-out evaluation on 1,000 sessions never seen in training, and the gap to the naive baseline is
large and consistent across both targets. We didn't run k-fold, which would be the next step.

**Q. What features does it use?**
18: soil hardness, slope, rain, visibility, temperature, travel distance, engine load, soil
moisture, recent idle ratio, recent load ratio, recent incident count, years of experience, plus
categoricals — workload, machine condition, congestion, task type, task phase, day/night.

---

## 4. Behaviour

**Q. How do you separate operator behaviour from circumstances?**
Two stages. First predict the *expected* value for these conditions — machine, site, weather, task.
Then `residual = observed − expected`, decomposed into an operator-linked part and a
context-explained part. Both are idle ratios, same unit, populated together so one can't exist
without the other.

**Q. Why Ridge rather than your better model?**
Different job. The prediction model's number is consumed directly, so accuracy wins. The behaviour
model's number is *subtracted from* and then explained to an operator, so interpretability wins. A
linear contribution is defensible; a boosted ensemble's isn't.

**Q. What if the operator/context split is wrong?**
Then coaching decisions are wrong, which is why the gate needs four conditions rather than trusting
the split alone. We also enforce a unit contract — both components must be the same unit, or the
ratio between them is meaningless. There's a runtime warning if the context component is zero while
the operator residual isn't, because that silently disables the protection.

**Q. How do you avoid blaming operators?**
The system never claims an operator performed badly. The only claim available is that observed
behaviour differs from what was expected under the current operating context. And a gap the context
explains never becomes coaching.

---

## 5. Safety

**Q. How are the safety rules defined?**
33 rules in `config/safety_rules.yaml`. Each has an id, severity, required action, and a list of
conditions that must all match. Operators are `gt gte lt lte eq ne in not_in is_true is_false`. OR
is expressed as a separate rule. The Python file is an evaluator — it contains no thresholds.

**Q. How is severity assessed?**
It's declared per rule, not computed. Order is INFO < LOW < MEDIUM < HIGH < CRITICAL. Rules can
share a `group`, and within a group only the most severe firing rule survives — one worker produces
one alert, not four.

**Q. Give me a concrete example of context beating a single factor.**
A worker 20 m from an active machine is MEDIUM. The same worker at the same distance with the
machine stationary and outside the movement envelope is INFO, action `none_required`. The INFO rule
literally requires `machine_active: is_false`. Distance never decides on its own.

**Q. Why not use ML for safety?**
Because you can't audit it and you can't explain a refusal to a regulator. Every event names the
rule that fired and the action required. A model that's 99% right on safety is 1% catastrophic.

**Q. What if a sensor fails?**
A condition whose field is missing or null never matches. Missing data cannot fire a rule, so a
dropout produces silence rather than a false alarm. The Buddy's gate goes the other way —
deny-by-default — because there, absent data shouldn't grant permission.

**Q. How do you determine the swing envelope?**
Approximated: machine state is `swinging` and the worker is within a 12 m radius. It's not a true
arc — the telemetry has no heading-relative bearing. A documented simplification, not a claim.

**Q. How do you avoid alarm fatigue?**
Three ways. Severity bands so most events are INFO/LOW and logged rather than surfaced; group
de-duplication; and the Attention Manager, which holds anything below high safety until the operator
is in a state to receive it.

**Q. Does the LLM touch safety?**
No. It can add an explanation to an already-made decision. Severity, required action and trigger
reason are copied unchanged, `decision_changed` is always False, and CRITICAL is never sent to the
model at all — deliberately, so there's no latency on an emergency. There's also a rejected-phrases
list: text containing "safe to continue", "false alarm", "downgrade" and similar is discarded.

---

## 6. Optimization

**Q. What algorithm?**
Greedy sequential selection. At each step it scores every feasible remaining task from the current
location, clock and attachment, picks the lowest score, advances the clock, repeats until the
fatigue budget would be exceeded.

**Q. What's the objective?**
`time_cost + fuel_cost + deadline_risk + transition_cost + safety_condition_risk`, then divided by
a priority factor (P1 ÷ 2.0 down to P5 ÷ 1.0). Time is ETA P50 × 1.0; fuel is P50 × 2.0; transition
is km × 7.5 plus 20 for an attachment swap and 15 for a blocked route; safety cost is INFO 0, LOW 2,
MEDIUM 10, HIGH 30.

**Q. How does uncertainty enter the plan?**
Deadline risk keys off P90, not P50. A task whose P50 finishes on time but whose P90 doesn't is
priced as "at risk" at 40. A P50 that misses is 120 plus 5 per hour late. So the prediction's
uncertainty changes the ordering, not just the display.

**Q. Is it optimal?**
No. It's a bounded local search — top 120 open tasks by priority and deadline, filtered to the
machine type, then greedy. The architecture explicitly said not to build a solver. For a five-to-
eight task shift the gap is small, and every step's reasoning is inspectable, which matters more
here.

**Q. How are hard constraints handled?**
Not as costs. A CRITICAL safety finding returns before scoring happens. The 8-hour fatigue budget
ends the plan. An unmet dependency excludes the task, though a dependency satisfied earlier in the
same plan counts. That's the difference between a constraint and a weight — no amount of time or
fuel saving can buy past them.

**Q. What triggers replanning?**
A material change in the operating context — rain, visibility, congestion, soil hardness, slope,
machine condition — each with a noise threshold so sensor jitter doesn't count. The dashboard names
the reason rather than just showing a new order.

---

## 7. Attention Manager

**Q. What does it do?**
Every subsystem can propose an event. Only the Attention Manager decides what reaches the operator
now. Each candidate carries severity, urgency, actionability, confidence and the operator's current
state, and the arbiter returns show_now / queue / bundle / suppress with a reason.

**Q. Why does that matter?**
It's the difference between a system that assists and one that nags. Training and Buddy candidates
are always deferrable and only actionable once stopped, so a lesson can wait an entire shift rather
than interrupting a swing. Critical safety always wins.

---

## 8. Biometrics

**Q. What's the recognition stack?**
DeepFace with ArcFace, producing 512-dimension embeddings, compared by cosine similarity against a
0.70 threshold from config. We didn't train a face model — that was explicitly out of scope.

**Q. How is privacy handled?**
Everything is local. We store embeddings, not photographs; the directory is gitignored; nothing is
uploaded. Only three real people are enrolled — every other operator in the dataset is synthetic
with no biometric data at all.

**Q. How was the threshold chosen, and is it validated?**
Partly. On the accept side, enrolled frames score 0.88–0.99 leave-one-out, comfortably above 0.70.
**The reject side is untested** — with one face enrolled there's no cross-person distribution. We'd
set the threshold between the two distributions before trusting it.

**Q. What about spoofing — a photo of someone?**
No liveness detection. That's a real gap for production and out of scope here. We'd add a
challenge-response or a depth sensor.

**Q. What happens if recognition fails?**
No session is created. Below the threshold the result carries `operator_id: null`, so there's
nothing to open a session with.

**Q. Is recognition the same as authorization?**
No, and that separation is deliberate. Recognition says who; `check_authorization` decides what they
may operate — machine type clearance, certification status, certification expiry, each reported
separately so you see every failure, not just the first.

---

## 9. Buddy

**Q. Is this just a chatbot?**
No. It retrieves from approved sources and quotes them with a citation. The default answer is
assembled from text a trusted source already carries. It doesn't generate operating advice.

**Q. How do you stop it hallucinating?**
Four ways. It can only draw from eight approved sources. Answers are quoted, not generated, by
default. When optional LLM phrasing is enabled, the model gets *only* the evidence, and **any number
it emits that isn't in the evidence causes the answer to be discarded** and the quoted version used.
And citations are attached by code, not by the model.

**Q. What's the safe-state gate?**
The Buddy is only available when the machine is parked or in verified safe idle **and** nothing is
moving — no attachment movement, no arm speed, no machine speed. "Not travelling" is explicitly not
enough. Unknown state denies: absent telemetry isn't evidence of safety.

**Q. Where does it get its definition of "safe"?**
From the Safety Guardian's own config file, not its own list. Otherwise the two would drift apart.

**Q. How does it handle disagreeing sources?**
Sources are ranked by authority — safety incident 100, approved manual 90, telemetry 70, down to
prediction 30. A genuine disagreement is stated, then resolved by the authoritative source, or
deferred if two sources tie at the top. Naturally multi-valued sources (several safety events,
several manual snippets) aren't treated as self-contradicting, and the manual is excluded from value
comparison entirely — guidance and a sensor reading answer different questions.

**Q. Is this RAG?**
It's structured retrieval rather than embedding search — the architecture explicitly preferred that
for the MVP. Sources are keyed by topic keywords, which is auditable and has no index to go stale.

---

## 10. Training Hub

**Q. When does coaching trigger?**
Four conditions, all required per occurrence: attribution is operator-driven or mixed; confidence
≥ 0.70; the context-explained share is ≤ 0.50; and the residual is in the unfavourable direction.
It fires when at least 2 qualifying occurrences appear in the last 10.

**Q. Why that fourth condition?**
Because `context_share` compares magnitudes, so an operator who *beat* expectation looked identical
to one who fell short. Before we caught it, the gate assigned an idle-reduction lesson to an
operator who idled *less* than predicted. Every test used positive residuals, so nothing caught it
until integration with the real model.

**Q. Does it actually discriminate, or does it fire for everyone?**
5 of 30 operators. We ran it across the full set.

**Q. How is improvement measured?**
A before metric against an after metric, with an explicit direction (lower is better for idle, cycle
time, fuel). Passing the quiz doesn't close the issue — only a measured follow-up meeting the
improvement target does. A zero before-metric reports "not comparable" rather than inventing a
percentage.

---

## 11. Engineering practice

**Q. How do you know any of this works?**
423 tests. More usefully: we mutation-tested the test suite — deliberately broke the authorization
check, the safe-state gate, the training gate's context rule and the leakage filter, and confirmed
tests failed each time. A test that can't fail isn't evidence.

**Q. What broke during integration?**
Several things that each passed their own tests. Replay passed `None` as a session context, harmless
until Safety was wired up, then it crashed a whole section. Telemetry stored `attachment_movement`
as a boolean, not a word — the Buddy would have been blocked on all 75,000 rows. Safety events were
attached to every question, so "how much fuel is left" answered with a collision alert. And the
coaching-direction bug. All of them only appear when the pieces meet.

**Q. How did three people work in parallel without breaking each other?**
Feature branches with a shared contract agreed first, integrated on `develop`. Cross-feature calls
go through documented signatures, so a teammate's internals can change without breaking anyone. The
adapter layer means the dashboard runs with any subset integrated.

**Q. What would you do differently?**
Agree the *shapes* as well as the names up front — the boolean-versus-word telemetry field and the
dict-versus-list exclusions both cost time. And write integration tests earlier; every bug that
mattered lived in the seams, not the units.

**Q. Is this production ready?**
No, and the honest list is short: interval calibration, no liveness detection on the biometrics, the
swing envelope is a radius not an arc, greedy rather than optimal sequencing, and every model is
trained on synthetic data. What is production-shaped is the decision logic and the separation
between deterministic safety and everything else.

---

## 12. The hardest question

**Q. Your models are trained on data you generated. Aren't they just learning your own generator?**

> "Partly, yes — and that's the right thing to be sceptical about. The generator has a known
> structure, and a gradient booster can recover it. So the 74% improvement over baseline shows the
> pipeline works end to end. It is *not* evidence the model would transfer to real telemetry.
>
> What transfers is the architecture: the leakage discipline, the quantile outputs, the decision
> logic on top, and the rule layer that doesn't depend on any model at all. We'd retrain from
> scratch on real fleet data and wouldn't carry these weights across."

Don't fight this one. Conceding it precisely is more convincing than defending it.
