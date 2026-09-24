"""Curated demo scenarios — a shortlist of sessions worth showing.

There are 5,000 sessions. Most demonstrate nothing in particular, some produce
an empty plan, and hunting for a good one live is how a demo goes wrong. This
module names twenty that each show one thing clearly.

**Every scenario is anchored to a real session and its claims are asserted in
tests/demo_scenarios.py.** Nothing here is aspirational: if the data changes so
that a session stops showing what it promises, the tests fail rather than the
demo. That is the whole point of the file — a scenario list nobody verifies is
worse than no list, because it is trusted.

The `expect` fields are the verifiable part. `show` and `say` are guidance for
the presenter and are deliberately not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

CATEGORY_SAFETY = "Safety"
CATEGORY_AUTH = "Authorization"
CATEGORY_PLAN = "Planning"
CATEGORY_PREDICT = "Prediction"
CATEGORY_TRAINING = "Behaviour & Training"
CATEGORY_BUDDY = "Buddy"
CATEGORY_CONDITIONS = "Conditions"

CATEGORY_ORDER = (
    CATEGORY_SAFETY, CATEGORY_AUTH, CATEGORY_PLAN, CATEGORY_PREDICT,
    CATEGORY_TRAINING, CATEGORY_BUDDY, CATEGORY_CONDITIONS,
)


@dataclass(frozen=True)
class DemoScenario:
    key: str
    title: str
    category: str
    session_id: str
    operator_id: str
    show: str                      # where to look / what to click
    say: str                       # the point being made
    expect: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "category": self.category,
            "session_id": self.session_id, "operator_id": self.operator_id,
            "show": self.show, "say": self.say, "expect": dict(self.expect),
        }


# Verified against the dataset on 2026-09-24. `expect` keys:
#   worst_severity     highest severity in that session's safety_events
#   authorized         whether the Passport opens a session for this pairing
#   refusal_reason     expected authorization reason code when refused
#   min_plan_steps     plan must contain at least this many steps
#   plan_empty         the optimizer must refuse every task
#   deadlines_met      exact "met/total" string from the plan
SCENARIOS: tuple[DemoScenario, ...] = (
    # --- Safety -------------------------------------------------------------
    # OP1001, deliberately: the audience watches a face login as OP1001, so the
    # critical beat must be that same operator. OP1001's CRITICAL sessions are
    # all late in the shift, hence the empty plan -- which is worth saying out
    # loud rather than avoiding, since the fatigue rule is the reason.
    DemoScenario(
        key="critical-proximity",
        title="CRITICAL — worker in the swing path",
        category=CATEGORY_SAFETY,
        session_id="S004374", operator_id="OP1001",
        show="Safety banner at the top of the page.",
        say="Motion, swing path and closing speed made this critical. Distance alone did not.",
        expect={"worst_severity": "CRITICAL", "authorized": True, "plan_empty": True},
    ),
    DemoScenario(
        key="high-swing-envelope",
        title="HIGH — closing motion inside the envelope",
        category=CATEGORY_SAFETY,
        session_id="S000689", operator_id="OP1001",
        show="Safety banner: worker 6.0 m, closing 0.843 m/s, machine swinging.",
        say="Required action comes from the rule that fired, not from a model.",
        expect={"worst_severity": "HIGH", "authorized": True, "min_plan_steps": 1},
    ),
    DemoScenario(
        key="harmless-proximity",
        title="INFO — a nearby worker that is not a hazard",
        category=CATEGORY_SAFETY,
        session_id="S000019", operator_id="OP1001",
        show="Live operation event log: worker_outside_envelope_machine_stationary.",
        say="The system says this needs no action. A proximity alarm would have cried wolf.",
        expect={"worst_severity": "INFO", "authorized": True, "min_plan_steps": 1},
    ),
    DemoScenario(
        key="safety-blocks-plan",
        title="Safety as a hard block, not a cost",
        category=CATEGORY_SAFETY,
        session_id="S000689", operator_id="OP1001",
        show="Plan → 'Excluded from the plan' → CRITICAL safety: unsafe_ground_slope.",
        say="The optimizer cannot trade a critical safety finding away against time or fuel.",
        expect={"authorized": True, "min_plan_steps": 1},
    ),

    # --- Authorization ------------------------------------------------------
    DemoScenario(
        key="authorized-clean",
        title="Cleared to operate",
        category=CATEGORY_AUTH,
        session_id="S000146", operator_id="OP1001",
        show="Identity strip: AUTHORIZED.",
        say="Machine type, certification status and expiry all checked before a session opens.",
        expect={"authorized": True, "min_plan_steps": 3, "deadlines_met": "5/5"},
    ),
    DemoScenario(
        key="refused-machine-type",
        title="Refused — not rated for this machine",
        category=CATEGORY_AUTH,
        session_id="S000054", operator_id="OP1001",
        show="Identity strip: REFUSED, with the reason beneath.",
        say="Recognised is not the same as cleared. The face matched; the clearance did not.",
        expect={"authorized": False, "refusal_reason": "machine_type_not_authorized"},
    ),
    DemoScenario(
        key="refused-expired-cert",
        title="Refused — certification lapsed",
        category=CATEGORY_AUTH,
        session_id="S002093", operator_id="OP1002",
        show="Identity strip: REFUSED, reason certification_expired.",
        say="A lapsed certificate stops the session outright. Deliberate, not a data error.",
        expect={"authorized": False, "refusal_reason": "certification_expired"},
    ),

    # --- Planning -----------------------------------------------------------
    DemoScenario(
        key="full-plan",
        title="Full shift plan, every deadline met",
        category=CATEGORY_PLAN,
        session_id="S000146", operator_id="OP1001",
        show="Plan panel: ordered sequence with NOW, then the Next Task card.",
        say="Ordered on time, fuel, deadline risk, travel and safety together.",
        expect={"authorized": True, "min_plan_steps": 3, "deadlines_met": "5/5"},
    ),
    DemoScenario(
        key="deadline-risk",
        title="Plan carrying deadline risk",
        category=CATEGORY_PLAN,
        session_id="S000303", operator_id="OP1001",
        show="Plan list: steps flagged past deadline or at risk.",
        say="Deadline pressure is priced into the order, and shown rather than hidden.",
        expect={"authorized": True, "min_plan_steps": 3},
    ),
    DemoScenario(
        key="fatigue-stops-work",
        title="Fatigue budget refuses more work",
        category=CATEGORY_PLAN,
        session_id="S000108", operator_id="OP1001",
        show="Empty plan; exclusions all read 'exceeds 8 h shift budget'.",
        say="This operator is past the shift budget, so nothing is assigned. Not a bug.",
        expect={"authorized": True, "plan_empty": True},
    ),
    DemoScenario(
        key="many-exclusions",
        title="Why most tasks were not chosen",
        category=CATEGORY_PLAN,
        session_id="S000495", operator_id="OP1001",
        show="Plan → Excluded: dependencies, fatigue and safety blocks side by side.",
        say="The refusals are as informative as the plan.",
        expect={"authorized": True, "min_plan_steps": 3},
    ),

    # --- Prediction ---------------------------------------------------------
    DemoScenario(
        key="uncertainty-bands",
        title="ETA and fuel as ranges, not points",
        category=CATEGORY_PREDICT,
        session_id="S000146", operator_id="OP1001",
        show="Prediction panel: P10–P90 track with P50 marked.",
        say="A point estimate hides the risk. The band is the useful part.",
        expect={"authorized": True, "min_plan_steps": 3},
    ),
    DemoScenario(
        key="predicted-vs-actual",
        title="Did the actual land inside the band?",
        category=CATEGORY_PREDICT,
        session_id="S000019", operator_id="OP1001",
        show="End of shift tab.",
        say="The honest test is coverage of the interval, not how close P50 was.",
        expect={"authorized": True, "min_plan_steps": 1},
    ),

    # --- Behaviour and training --------------------------------------------
    DemoScenario(
        key="training-fires",
        title="Coaching triggered, with evidence",
        category=CATEGORY_TRAINING,
        session_id="S000689", operator_id="OP1001",
        show="Training Hub → Gate: 7 of 10 qualifying, lesson L003.",
        say="Repeated, confident, operator-linked and not explained by context — all four.",
        expect={"authorized": True},
    ),
    DemoScenario(
        key="training-refuses",
        title="No coaching — a different operator, gate holds",
        category=CATEGORY_TRAINING,
        session_id="S001007", operator_id="OP1003",
        show="Training Hub → Gate: reasons listed for not firing.",
        # Necessarily a different operator: OP1001's gate fires, so showing it
        # hold requires someone whose pattern does not qualify. Say so.
        say="A different operator. Same gate, and here it declines to coach.",
        expect={"authorized": True},
    ),
    DemoScenario(
        key="peer-techniques",
        title="How others handled similar work",
        category=CATEGORY_TRAINING,
        session_id="S000146", operator_id="OP1001",
        show="Training Hub → Peer techniques.",
        say="Anonymised, approved only, and the similarity score is always shown.",
        expect={"authorized": True},
    ),

    # --- Buddy --------------------------------------------------------------
    DemoScenario(
        key="buddy-available",
        title="Buddy answers at safe idle",
        category=CATEGORY_BUDDY,
        session_id="S000146", operator_id="OP1001",
        show="Buddy tab, state safe_idle. Ask: what is my next task.",
        say="Answers are quoted from a trusted source and cited, never generated.",
        expect={"authorized": True},
    ),
    DemoScenario(
        key="buddy-blocked",
        title="Buddy blocked while the machine works",
        category=CATEGORY_BUDDY,
        session_id="S000689", operator_id="OP1001",
        show="Buddy tab, switch machine state to swinging.",
        say="Not travelling is not enough. Attachment movement blocks it too.",
        expect={"authorized": True},
    ),
    DemoScenario(
        key="buddy-defers",
        title="Buddy defers a safety question",
        category=CATEGORY_BUDDY,
        session_id="S000146", operator_id="OP1001",
        show="Buddy tab. Ask: is it dangerous near the river.",
        say="Not covered by the approved manual, so it defers to the Safety Guardian.",
        expect={"authorized": True},
    ),

    # --- Conditions ---------------------------------------------------------
    DemoScenario(
        key="degraded-visibility",
        title="Degraded visibility and dust",
        category=CATEGORY_CONDITIONS,
        session_id="S000019", operator_id="OP1001",
        show="Conditions strip: visibility 140 m, dust high.",
        say="Conditions feed the prediction and the plan, not just the display.",
        expect={"authorized": True, "min_plan_steps": 1},
    ),
)


# The dashboard opens here. It must be an OP1001 scenario: OP1001 is the only
# registered face, so opening on anyone else contradicts the login that just
# happened. "full-plan" is also the cleanest first impression -- authorized,
# every deadline met, nothing red.
DEFAULT_SCENARIO_KEY = "full-plan"


def default_scenario() -> DemoScenario:
    return scenario_by_key(DEFAULT_SCENARIO_KEY) or SCENARIOS[0]


def default_label_index() -> int:
    """Index into ["— browse —"] + scenario_labels(), so the selectbox opens here."""
    labels = scenario_labels()
    target = f"{default_scenario().category} — {default_scenario().title}"
    return labels.index(target) + 1 if target in labels else 0


def all_scenarios() -> tuple[DemoScenario, ...]:
    return SCENARIOS


def scenario_by_key(key: str) -> Optional[DemoScenario]:
    return next((s for s in SCENARIOS if s.key == key), None)


def scenarios_by_category() -> dict[str, list[DemoScenario]]:
    grouped: dict[str, list[DemoScenario]] = {c: [] for c in CATEGORY_ORDER}
    for scenario in SCENARIOS:
        grouped.setdefault(scenario.category, []).append(scenario)
    return {c: items for c, items in grouped.items() if items}


def scenario_labels() -> list[str]:
    """Selectbox labels, grouped by category in presentation order."""
    return [f"{s.category} — {s.title}" for s in SCENARIOS]


def scenario_for_label(label: str) -> Optional[DemoScenario]:
    return next((s for s in SCENARIOS if f"{s.category} — {s.title}" == label), None)
