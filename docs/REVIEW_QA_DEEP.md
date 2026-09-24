# Deep Technical Q&A

Harder follow-ups than `REVIEW_QA.md`. Implementation, statistics and systems level.
Verified against the code. Where the honest answer is "that's a flaw", it says so.

---

## FIRST — a leakage issue you should raise yourself

**Q. Your #2 fuel feature is `recent_idle_ratio`. Where does that come from?**

This is the sharpest finding in the project and you should volunteer it.

In the generator, `recent_idle_ratio` is set to **this session's** `idle_r` plus σ=0.01 noise. And
`idle_r` appears directly in the fuel equation:

```python
idle_r = op_idle[op] * cong_f[cong] + N(0, 0.02)
fuel   = op_fuel[op] * dur/60 * tf * terrain_f * (1 + idle_r * 0.5) * ... 
recent_idle_ratio = idle_r + N(0, 0.01)      # ~99% correlated with the above
```

So the fuel model is partly predicting fuel from a near-copy of a term in its own generating
equation — and that quantity isn't known before the task starts. Despite the name it is not a
historical average.

> "It's soft leakage. Not the forbidden target column — our hard rule on `actual_fuel_used_l` holds —
> but a feature that shouldn't be available pre-task and is ~99% correlated with a multiplicative
> term in the target. It inflates the fuel model's apparent skill. The fix is either to make the
> field genuinely historical in the generator, or drop it from the fuel feature set and re-measure.
> We'd expect the fuel MAE to worsen and that number would be the honest one."

**The ETA model is cleaner:** `dur` is computed *before* `idle_r` and doesn't use it. Its #2 feature
`recent_load_ratio` is derived from workload, which is already a legitimate input — that's
redundancy, not leakage.

---

## Statistics and ML

**Q. Why does quantile loss actually give you a percentile?**
Pinball loss weights errors asymmetrically: under-prediction costs `α`, over-prediction costs
`1−α`. At α=0.9, being under is nine times more expensive, so the minimiser sits where 90% of mass
falls below. The optimum of that loss *is* the quantile.

**Q. What if P10 comes out above P50? Quantile crossing.**
Three independently trained models can cross. We clamp in `_predict_with_percentiles`:
`p10 = max(0.1, p10); p50 = max(p10, p50); p90 = max(p50, p90)`. That's a post-hoc monotonicity
fix, not a principled one — proper approaches are joint quantile estimation or isotonic
rearrangement.

**Q. Your split is random. Isn't this temporal data?**
Fair criticism. Sessions span January to October and we split randomly at 80/20, so the model can
see later sessions while predicting earlier ones. For a deployment you'd split temporally — train
on the first N months, test on the rest — because that's the real prediction problem. Random split
likely flatters the metrics.

**Q. How would you calibrate the intervals properly?**
Conformal prediction is the cleanest: hold out a calibration set, measure the empirical quantile of
conformity scores, and widen the interval by that amount to get a distribution-free coverage
guarantee. Cheaper alternative: widen alphas to 0.05/0.95 and re-measure empirical coverage.

**Q. Why report MAE rather than MAPE or R²?**
MAE is in the unit the operator cares about — minutes and litres — and is robust to the long right
tail on duration. MAPE would blow up on short tasks. We report RMSE alongside so you can see the
tail penalty: 13.3 vs 8.4 on ETA means a few large misses.

**Q. What exactly is the naive baseline?**
Median duration for the same task type, per the architecture's specification. It's a real baseline,
not a constant — it captures "trenching takes longer than grading" and nothing else.

**Q. n_estimators=300 with no early stopping — how do you know that's right?**
We don't. There's no validation-based early stopping and no hyperparameter search; 300 was a
reasonable default and the held-out numbers were good enough. Proper tuning would use early
stopping on a validation fold.

**Q. How are categoricals encoded?**
`cat_maps` in `preprocessing.json` — integer codes saved at training time and reapplied at
inference, so encoding can't drift between train and serve. Six categorical features.

**Q. Multicollinearity — visibility, dust and day/night are correlated.**
Yes: `dust = high if vis < 200 else medium if vis < 500 else low` in the generator, so dust is a
deterministic function of visibility. For trees that's harmless for prediction but it splits and
dilutes the importance scores, which is one reason not to read importance as causality.

**Q. How is Ridge's alpha chosen for the behaviour model?**
That's Member 1's component. It's a regularisation strength; if not cross-validated it's a default.
Worth being straight that we didn't tune it.

**Q. How would you detect data drift in production?**
Monitor the feature distributions against the training distribution (PSI or KS per feature), and
monitor realised coverage of the P10–P90 band — if coverage drifts away from nominal, the model has
gone stale even if MAE looks fine.

**Q. What's your error distribution — is it biased?**
Not measured. We report MAE and RMSE but no residual analysis by segment. A proper review would
check bias by task type, operator and machine condition, because an average that's fine overall can
hide systematic error on a subgroup.

**Q. Safety events are rare. Is that a class imbalance problem?**
Not for us, because safety isn't a classifier — it's rule evaluation. Imbalance would matter if we'd
learned safety from data, which is exactly one of the reasons we didn't.

**Q. Could you predict safety events instead of ruling them?**
You could, and you shouldn't. A learned model gives you a probability you can't audit and can't
explain to a regulator, and the cost of a false negative is a person. Rules are inspectable, and
every event names the rule that fired.

---

## Safety engineering

**Q. Two CRITICAL rules fire at once — what happens?**
Both produce events unless they share a `group`. Within a group the highest severity wins with YAML
order as the tiebreak. Across groups they coexist, which is correct — a seatbelt violation and a
proximity violation are separate facts needing separate actions.

**Q. How do you test 33 rules?**
Condition-level tests plus fixtures that exercise each severity band. The more useful check is that
a condition on a missing field never matches — that's the property preventing sensor dropout from
firing alarms, and it's tested directly.

**Q. What's your false negative rate?**
Unmeasurable here, and that's the honest answer. There's no ground-truth incident set — the
"events" in our data were generated by rules similar to the ones evaluating them. Validating recall
needs real incident data with labelled outcomes.

**Q. Isn't testing rules against data generated by rules circular?**
Largely yes. It proves the evaluator implements the rules, not that the rules are right. Rule
correctness is a domain-expert question, not a data question.

**Q. Fail-safe or fail-operational?**
Both, deliberately, in different places. The Safety Guardian fails *silent* — missing data never
fires a rule, so a dropout doesn't spam alarms. The Buddy's gate fails *closed* — unknown state
denies interaction. The asymmetry is intentional: absent data is not evidence of danger, but it's
also not permission.

**Q. Any debouncing or hysteresis?**
Only the seatbelt grace period (5 s) in config. Otherwise each telemetry row is evaluated
independently, so a value oscillating across a threshold would flap. Production would need
hysteresis bands or an N-of-M confirmation.

**Q. Why is CRITICAL excluded from the LLM path?**
Latency and risk. An emergency shouldn't wait on a language model, and there's no upside — the
action is already determined. It's a config list (`llm_reasoning.enabled_severities: [HIGH, MEDIUM]`),
not a code branch.

**Q. What stops the LLM contradicting the rule on a HIGH event?**
A rejected-phrases list. Output containing "safe to continue", "no action", "false alarm",
"downgrade", "override" and similar is discarded and the rule's own text used. Crude but effective,
and it fails toward the rule.

---

## Optimizer

**Q. What's the complexity?**
O(steps × candidates) scoring calls. With a 120-task pool and ~5 steps that's up to 600 evaluations,
each needing a prediction and a safety check — which is why both are cached per run, keyed on the
fields that actually affect the result rather than on IDs.

**Q. Is the prediction cache key correct?**
It keys on `(task_id, machine_condition, years_experience, baseline_idle_ratio, weather timestamp)`
— the things the features depend on, not the operator or machine ID. If two operators share
experience and baseline they share a cache entry, which is correct because the feature vector is
identical.

**Q. Why divide by priority instead of multiplying cost?**
Same ordering, different scaling. Division by a factor >1 makes high-priority tasks cheaper rather
than making low-priority ones arbitrarily expensive, which keeps the cost components readable in
the breakdown.

**Q. How were the weights chosen?**
By hand, in units of roughly machine-minutes — fuel at 2.0 per litre, transition at 7.5 per km
(≈8 km/h travel), attachment swap 20 minutes. They're documented assumptions, not fitted. A real
deployment would derive them from actual cost data.

**Q. Greedy can paint itself into a corner. Example?**
Yes — taking a cheap nearby task first can strand you far from a high-priority deadline. Lookahead
or beam search would help. With five to eight tasks the exposure is small, but the limitation is
real.

**Q. What about dependency cycles?**
A cycle means neither task is ever feasible, so both stay excluded with "waiting on dependency".
It terminates safely but doesn't detect or report the cycle as such.

**Q. Why filter the candidate pool at all?**
Cost. Scoring every open task means a model call each. I tried widening the pool and filtering by
machine type first — it made plans take 87 seconds and didn't change the output, so I reverted it.

**Q. Is the plan deterministic?**
Yes, given the same data and clock. No randomness in scoring; models are fixed artefacts.

---

## Buddy and LLM

**Q. Can the evidence prompt-inject the model?**
In principle yes — evidence text goes into the prompt. Mitigations are structural rather than
prompt-based: the model only phrases, citations are appended by code, any ungrounded number
discards the output, and safety questions never reach the model. A crafted manual snippet could
still influence wording, but the manual is a curated file, not user input.

**Q. Where does the number-grounding check fall short?**
It only checks numbers. A model could introduce an ungrounded *qualitative* claim — "it's safe to
proceed" — and pass. That's partly covered by the rejected-phrases list on the safety side, and by
synthesis being disabled for safety questions entirely, but it's a real gap for non-safety answers.

**Q. Why a local model rather than an API?**
Privacy and determinism of availability. Nothing about an operator's shift leaves the machine, and
there's no external dependency to fail mid-shift. It falls back to quoting when Ollama isn't
running, which is the default state.

**Q. Why keyword retrieval rather than embeddings?**
The architecture explicitly preferred a simple structured layer for the MVP. It's auditable — you
can see exactly why a snippet matched — and there's no index to rebuild or go stale. The cost is
recall: a paraphrase that shares no keywords retrieves nothing, and the system then defers rather
than guessing.

**Q. Two keyword bugs you hit — what were they?**
`danger` was a keyword on the emergency snippet, so *any* safety question returned emergency-stop
guidance regardless of topic. And `fuel` was on the refuelling snippet, so "how much fuel is left"
returned a refuelling procedure instead of the gauge reading. Both were over-broad keywords, and
both produced confident-looking answers to questions nobody had asked.

**Q. How do you know a deferral is correct rather than just a retrieval miss?**
You don't, entirely. A deferral is safe but not necessarily right — if retrieval missed relevant
evidence, the Buddy defers when it could have answered. We chose that direction deliberately, but
it means recall failures are invisible.

---

## Biometrics

**Q. Why cosine similarity rather than Euclidean?**
ArcFace is trained with an angular margin loss, so its embedding space is angular — identity is
encoded in direction, not magnitude. Cosine is the metric the model was optimised for.

**Q. How many enrolment frames, and why does that matter?**
Six. Matching takes the max across all of them, so more frames covering different poses and lighting
raises recall. During enrolment we caught two frames containing a *different face* — 0.97 to each
other, 0.09 to the operator — which would have let that person authenticate. We removed them.

**Q. How would you set the threshold properly?**
Collect same-person and different-person similarity distributions, plot FAR and FRR against
threshold, and pick the operating point for your cost asymmetry — for equipment access you'd favour
a low false-accept rate. We only have the same-person side.

**Q. What's the cost of a false accept versus a false reject here?**
Asymmetric. A false reject is friction — the operator re-scans. A false accept lets an unauthorised
or uncertified person start a machine. That argues for a higher threshold than we've set, and it's
another reason the reject side needs measuring.

---

## Systems and engineering

**Q. Why is the first page load 25 seconds?**
TensorFlow and the ArcFace graph initialise, LightGBM artefacts unpickle, and 75,000 telemetry rows
are read from CSV. Subsequent interactions are ~12s because Streamlit caches the frames but re-runs
the script top to bottom on every widget change.

**Q. How would you make it fast?**
Move telemetry to Parquet or a database with predicate pushdown, cache the model objects in
`st.cache_resource` rather than reloading, and precompute the per-session aggregates the dashboard
recomputes on each rerun.

**Q. Why pin scikit-learn and lightgbm exactly?**
The models are pickled estimators. Unpickling across versions raises `InconsistentVersionWarning`
and sklearn states it may produce invalid results — but the model still loads and still returns
numbers, so the failure is silent. Two people could demo the same model, get different predictions,
and see no error. We hit exactly this: models trained on 1.4.2, environment running 1.9.1.

**Q. Pickle is unsafe — isn't that a risk?**
Yes. Unpickling executes arbitrary code, so these artefacts are only as trustworthy as their source.
For production you'd use ONNX or LightGBM's own text format, which don't execute anything on load.

**Q. Tell me about the pickle bug you hit.**
The behaviour model was trained by running its own file as a script, so the class was recorded as
`__main__.BehaviorModel`. It loaded fine for the person who trained it and failed for everyone else
with `AttributeError`. Fix needed no code change — retrain via an import so the class gets its real
module path.

**Q. How do you version models?**
We don't, beyond git. There's no model registry, no hash of the training data recorded with the
artefact. `reports/model_metrics.json` acts as a model card but isn't linked to a specific artefact
hash.

**Q. Is anything thread-safe?**
Not audited. Streamlit runs one script per session; the module-level caches in the optimizer and the
lazily-initialised Ollama client are shared state that would need review under concurrency.

**Q. How did you test that your tests work?**
Mutation testing by hand: broke the certification-expiry check, the safe-state gate, the training
gate's context rule, and the leakage filter, and confirmed each produced the expected failures, then
reverted. A green suite that can't fail is not evidence.

**Q. What's the slow test tier?**
Tests that call the optimizer and prediction models take ~6s each, so they're behind `RUN_SLOW=1`.
423 collected; 410 run by default, 13 skipped.

**Q. You rewrote tests when features integrated. Isn't that fitting tests to code?**
Fair challenge, and it depends what changed. Those tests asserted teammates' features were *absent* —
a fact about a moment in time, not a contract. They failed for succeeding. I rewrote them to assert
that the adapter reports what's genuinely importable, which is true before and after integration.
That's fixing a bad test, not weakening a good one.

**Q. What's the single biggest risk in this codebase?**
The synthetic data. Every model is trained on a process we wrote, and the `recent_idle_ratio` issue
above shows the generator can leak structure into features in ways that aren't obvious. The
rule-based layers don't have that exposure, which is part of why safety is rules.
