"""Tests for the Operator Passport (Feature 01).

`AS_OF` is fixed so certification-expiry tests do not silently change meaning as
the real date moves past the fixture expiry dates.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from features.passport.authorization import (
    REASON_CERT_EXPIRED,
    REASON_CERT_STATUS,
    REASON_MACHINE_TYPE,
    check_authorization,
)
from features.passport.passport import (
    UnknownMachineError,
    UnknownOperatorError,
    create_session,
    load_passport,
)
from features.passport.repository import PassportRepository, _parse_bool, _parse_date, _parse_list

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = datetime(2026, 9, 23, 8, 0, 0)


@pytest.fixture
def repo() -> PassportRepository:
    return PassportRepository(FIXTURES / "operators.csv", FIXTURES / "machines.csv")


# --- parsing -----------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("excavator|loader", ("excavator", "loader")),
        ("excavator;loader", ("excavator", "loader")),
        ("excavator, loader", ("excavator", "loader")),
        ("['excavator', 'loader']", ("excavator", "loader")),
        ("excavator", ("excavator",)),
        ("", ()),
        (None, ()),
    ],
)
def test_parse_list_accepts_each_agreed_delimiter(raw, expected):
    assert _parse_list(raw) == expected


def test_parse_date_handles_date_and_timestamp_and_junk():
    assert _parse_date("2027-12-31") == date(2027, 12, 31)
    assert _parse_date("2027-12-31T08:00:00") == date(2027, 12, 31)
    assert _parse_date("") is None
    assert _parse_date("not-a-date") is None


def test_parse_bool_defaults_synthetic_flag_to_true_when_missing():
    assert _parse_bool("", default=True) is True
    assert _parse_bool("false", default=True) is False


# --- repository --------------------------------------------------------------

def test_repository_loads_operators_and_machines(repo):
    assert repo.get_operator("OP1003").name == "Saanvi"
    assert repo.get_machine("EXC001").machine_type == "excavator"
    assert repo.get_operator("OP9999") is None


def test_only_registered_operators_are_reported_as_face_registered(repo):
    registered = [op.operator_id for op in repo.face_registered_operators()]
    assert registered == ["OP1003"]


def test_missing_table_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_synthetic_data"):
        PassportRepository(tmp_path / "nope.csv", tmp_path / "also_nope.csv")


# --- authorization -----------------------------------------------------------

def test_authorized_operator_passes_every_check(repo):
    result = check_authorization(
        repo.get_operator("OP1003"), repo.get_machine("EXC001"), as_of=AS_OF.date()
    )
    assert result.authorized is True
    assert result.reasons == ()
    assert all(result.checks.values())


def test_unauthorized_machine_type_is_refused(repo):
    result = check_authorization(
        repo.get_operator("OP1004"), repo.get_machine("EXC001"), as_of=AS_OF.date()
    )
    assert result.authorized is False
    assert REASON_MACHINE_TYPE in result.reasons


def test_expired_certification_is_refused(repo):
    result = check_authorization(
        repo.get_operator("OP1005"), repo.get_machine("EXC001"), as_of=AS_OF.date()
    )
    assert result.authorized is False
    assert REASON_CERT_EXPIRED in result.reasons
    assert result.checks["machine_type_authorized"] is True


def test_suspended_certification_is_refused(repo):
    result = check_authorization(
        repo.get_operator("OP1006"), repo.get_machine("EXC001"), as_of=AS_OF.date()
    )
    assert result.authorized is False
    assert REASON_CERT_STATUS in result.reasons


def test_absent_expiry_is_not_treated_as_expired(repo):
    result = check_authorization(
        repo.get_operator("OP1007"), repo.get_machine("EXC001"), as_of=AS_OF.date()
    )
    assert result.authorized is True


def test_every_failing_check_is_reported_not_just_the_first(repo):
    result = check_authorization(
        repo.get_operator("OP1006"), repo.get_machine("DZR001"), as_of=AS_OF.date()
    )
    assert set(result.reasons) == {REASON_MACHINE_TYPE, REASON_CERT_STATUS}


# --- session creation --------------------------------------------------------

def test_authorized_session_carries_operator_and_machine_context(repo):
    result = create_session("OP1003", "EXC001", repo, task_id="T00042", as_of=AS_OF)
    assert result.authorized is True
    context = result.session_context
    assert context is not None
    assert context.operator_id == "OP1003"
    assert context.machine_id == "EXC001"
    assert context.task_id == "T00042"
    assert context.operator.authorization == "authorized"
    assert context.operator.personal_baseline["baseline_idle_ratio"] == 0.14
    assert context.machine.machine_type == "excavator"
    assert context.data_quality.synthetic_flag is True


def test_refused_operator_gets_no_session_context(repo):
    result = create_session("OP1005", "EXC001", repo, as_of=AS_OF)
    assert result.authorized is False
    assert result.session_context is None
    assert REASON_CERT_EXPIRED in result.authorization.reasons


def test_unknown_operator_and_machine_raise(repo):
    with pytest.raises(UnknownOperatorError):
        create_session("OP9999", "EXC001", repo, as_of=AS_OF)
    with pytest.raises(UnknownMachineError):
        create_session("OP1003", "EXC999", repo, as_of=AS_OF)


def test_load_passport_returns_baseline_without_scoring_the_operator(repo):
    operator = load_passport("OP1003", repo)
    assert operator.baseline.idle_ratio == 0.14
    # The Passport supplies context; it must not invent a performance score.
    assert not any("score" in f for f in vars(operator))


def test_live_machine_state_is_not_invented_from_the_machine_table(repo):
    """engine_state/speed/heading/load come from telemetry, not the machine CSV."""
    context = create_session("OP1003", "EXC001", repo, as_of=AS_OF).session_context
    assert context.machine.engine_state is None
    assert context.machine.speed is None
    assert context.machine.heading is None
    assert context.machine.load is None


def test_session_context_carries_no_task_outcome_fields(repo):
    """No-leakage guard: actuals must never ride along in a pre-task context."""
    context = create_session("OP1003", "EXC001", repo, as_of=AS_OF).session_context
    flat = repr(context)
    for leaked in ("actual_task_duration", "actual_fuel_used", "actual_idle_time", "actual_cycle"):
        assert leaked not in flat
