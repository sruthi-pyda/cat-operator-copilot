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
