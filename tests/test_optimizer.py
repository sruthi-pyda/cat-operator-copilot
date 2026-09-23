import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # prediction API loads models via repo-relative paths
warnings.filterwarnings("ignore")

from features.optimization import optimizer as opt
from shared.schemas import SessionContext


@pytest.fixture(scope="module")
def data():
    return opt.SiteData.load()


@pytest.fixture(scope="module")
def ranking(data):
    return opt.rank_assignments(data=data, pool_size=10)


def test_assignments_respect_hard_constraints(data, ranking):
    ops = data.operators.set_index("operator_id")
    mcs = data.machines.set_index("machine_id")
    tasks = data.tasks.set_index("task_id")
    assert ranking["assignments"]
    for a in ranking["assignments"]:
        op, mc, t = ops.loc[a["operator_id"]], mcs.loc[a["machine_id"]], tasks.loc[a["task_id"]]
        assert mc["machine_type"] == t["required_machine_type"]
        assert mc["machine_type"] in op["authorized_machine_types"].split(",")
        assert opt.SKILL_RANK[op["machine_skill_level"]] >= opt.SKILL_RANK[t["required_skill_level"]]
        assert op["certification_status"] == "active"
        assert not any(f.startswith("CRITICAL") for f in a["safety_flags"])


def test_each_resource_used_once_and_ranked_by_score(ranking):
    a = ranking["assignments"]
    for key in ("operator_id", "machine_id", "task_id"):
        assert len({x[key] for x in a}) == len(a)
    assert [x["score"] for x in a] == sorted(x["score"] for x in a)


def test_cost_breakdown_sums_to_total(ranking):
    for a in ranking["assignments"]:
        assert abs(sum(a["cost_breakdown"].values()) - a["total_cost"]) < 0.05
        assert set(a["cost_breakdown"]) == {"time_cost", "fuel_cost", "deadline_risk",
                                            "transition_cost", "safety_condition_risk"}


def test_generate_plan_contract_and_fatigue_budget(data, ranking):
    first = ranking["assignments"][0]
    ctx = SessionContext(session_id="S_T", operator_id=first["operator_id"], machine_id=first["machine_id"])
    plan = opt.generate_plan(None, ctx, data=data)
    assert {"session_id", "optimized_sequence", "total_cost", "cost_breakdown", "reason", "synthetic_flag"} <= set(plan)
    assert plan["optimized_sequence"]
    assert plan["shift_minutes_planned"] <= 8 * 60


def test_plan_respects_dependencies(data):
    tasks = data.tasks
    dep_task = tasks[tasks["status"].isin(["pending", "active"]) & tasks["dependency_task_id"].notna()
                     & (tasks["required_machine_type"] == "excavator")].iloc[0]
    ctx = SessionContext(session_id="S_D", operator_id="OP1005", machine_id="EXC001")
    plan = opt.generate_plan([dep_task["task_id"], dep_task["dependency_task_id"]], ctx, data=data)
    seq = plan["optimized_sequence"]
    if dep_task["task_id"] in seq:
        dep_done = tasks.set_index("task_id").loc[dep_task["dependency_task_id"], "status"] == "completed"
        assert dep_done or seq.index(dep_task["dependency_task_id"]) < seq.index(dep_task["task_id"])


@pytest.fixture(scope="module")
def scorer(data):
    return opt.WeightedScorer(data)


@pytest.mark.parametrize("specialized_only, operator_level, required, expected", [
    (True,  "expert",   "advanced",     0.8),   # specialised task, expert -> bonus
    (True,  "advanced", "advanced",     1.0),   # specialised task, not expert -> neutral
    (True,  "expert",   "novice",       1.0),   # ordinary task -> neutral (experts kept for specialised work)
    (True,  "advanced", "intermediate", 1.0),
    (False, "advanced", "intermediate", 0.8),   # literal rule: meets requirement -> bonus
    (False, "novice",   "intermediate", 1.0),   # below requirement -> neutral
    (False, "advanced", "advanced",     1.0),   # specialised still needs an expert
    (True,  None,       "advanced",     1.0),   # missing skill -> neutral
    (True,  "guru",     "advanced",     1.0),   # unknown skill -> neutral
])
def test_skill_bonus_score(scorer, monkeypatch, specialized_only, operator_level, required, expected):
    monkeypatch.setitem(scorer.cfg["skill_bonus"], "specialized_only", specialized_only)
    task = {"required_skill_level": required, "priority": 3}
    assert scorer.skill_bonus_score(task, {"task_skill_level": operator_level}) == expected


def test_skill_bonus_accepts_level_string(scorer):
    assert scorer.skill_bonus_score({"required_skill_level": "advanced"}, "expert") == 0.8


def test_calculate_total_score_applies_bonus_to_time_cost_only(scorer):
    breakdown = {"time_cost": 100.0, "fuel_cost": 20.0, "deadline_risk": 0.0,
                 "transition_cost": 10.0, "safety_condition_risk": 0.0}
    task = {"required_skill_level": "advanced", "priority": 1}
    adj, total, score, mult = scorer.calculate_total_score(breakdown, task, {"task_skill_level": "expert"})
    assert (adj["time_cost"], adj["fuel_cost"], total, mult) == (80.0, 20.0, 110.0, 0.8)
    assert score == round(110.0 / 2.0, 2)                              # priority 1 factor
    _, total_neutral, _, mult_neutral = scorer.calculate_total_score(breakdown, task, {"task_skill_level": "advanced"})
    assert (total_neutral, mult_neutral) == (130.0, 1.0)


def test_ranking_prefers_experts_for_specialised_tasks(data):
    """The bonus must move specialised work to experts, not push it out of the plan."""
    ops = data.operators.set_index("operator_id")
    tasks = data.tasks.set_index("task_id")

    def experts_on_specialised(multiplier):
        cfg = opt.load_settings()["skill_bonus"]
        old, cfg["time_cost_multiplier"] = cfg["time_cost_multiplier"], multiplier
        try:
            r = opt.rank_assignments(data=data, pool_size=60)
        finally:
            cfg["time_cost_multiplier"] = old
        return sum(tasks.loc[a["task_id"], "required_skill_level"] == "advanced"
                   and ops.loc[a["operator_id"], "task_skill_level"] == "expert"
                   for a in r["assignments"])

    assert experts_on_specialised(0.8) > experts_on_specialised(1.0)
