"""Operator Passport: identity + context backbone for every other feature.

Flow (biometric identification happens upstream, in biometric.py):

    operator_id -> load profile -> authorization check -> SessionContext

An unauthorized operator never receives a session context. Live machine values
(engine state, speed, heading, load) are deliberately left unset here: they come
from telemetry, not from the machine table, and the Passport does not invent them.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from features.passport.authorization import AuthorizationResult, check_authorization
from features.passport.repository import MachineRecord, OperatorRecord, PassportRepository
from shared.constants import DATA_QUALITY_CONFIDENCE_DEFAULT
from shared.schemas import (
    DataQuality,
    MachineContext,
    OperatorContext,
    SessionContext,
)


class UnknownOperatorError(KeyError):
    pass


class UnknownMachineError(KeyError):
    pass


@dataclass(frozen=True)
class SessionResult:
    """Outcome of a session attempt.

    `session_context` is None whenever `authorized` is False.
    """

    authorized: bool
    authorization: AuthorizationResult
    session_context: Optional[SessionContext] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "authorization": self.authorization.to_dict(),
            "session_id": self.session_context.session_id if self.session_context else None,
        }


def generate_session_id() -> str:
    return f"S{uuid.uuid4().int % 1_000_000:06d}"


def load_passport(operator_id: str, repository: PassportRepository) -> OperatorRecord:
    operator = repository.get_operator(operator_id)
    if operator is None:
        raise UnknownOperatorError(f"No operator record for {operator_id!r}")
    return operator


def build_operator_context(
    operator: OperatorRecord, authorization: AuthorizationResult
) -> OperatorContext:
    return OperatorContext(
        operator_id=operator.operator_id,
        experience=operator.years_experience,
        machine_skill=operator.machine_skill_level,
        task_skill=operator.task_skill_level,
        certifications=[operator.certification_status] if operator.certification_status else [],
        authorization="authorized" if authorization.authorized else "refused",
        personal_baseline=operator.baseline.to_dict(),
        recent_workload=operator.recent_workload_hours,
        fatigue_proxy=operator.fatigue_proxy,
    )


def build_machine_context(machine: MachineRecord) -> MachineContext:
    return MachineContext(
        machine_id=machine.machine_id,
        machine_model=machine.machine_model,
        machine_type=machine.machine_type,
        machine_condition=machine.machine_condition,
        engine_hours=machine.engine_hours,
        attachment=machine.attachment_type,
        fuel_level=machine.current_fuel_level_pct,
        maintenance_indicators=machine.maintenance_indicators(),
    )


def create_session(
    operator_id: str,
    machine_id: str,
    repository: PassportRepository,
    task_id: Optional[str] = None,
    as_of: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> SessionResult:
    operator = load_passport(operator_id, repository)
    machine = repository.get_machine(machine_id)
    if machine is None:
        raise UnknownMachineError(f"No machine record for {machine_id!r}")

    now = as_of or datetime.now()
    authorization = check_authorization(operator, machine, as_of=now.date())
    if not authorization.authorized:
        return SessionResult(authorized=False, authorization=authorization)

    timestamp = now.isoformat(timespec="seconds")
    synthetic = operator.synthetic_flag or machine.synthetic_flag
    context = SessionContext(
        session_id=session_id or generate_session_id(),
        operator_id=operator.operator_id,
        machine_id=machine.machine_id,
        task_id=task_id,
        timestamp=timestamp,
        operator=build_operator_context(operator, authorization),
        machine=build_machine_context(machine),
        data_quality=DataQuality(
            source=operator.data_source,
            timestamp=timestamp,
            synthetic_flag=synthetic,
            confidence=DATA_QUALITY_CONFIDENCE_DEFAULT,
        ),
    )
    return SessionResult(authorized=True, authorization=authorization, session_context=context)
