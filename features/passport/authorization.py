"""Deterministic authorization check for the Operator Passport.

Answers one question: may this operator run this machine right now?

Authorization is independent of identity. The biometric step establishes *who*
the operator is; this module decides *what they are cleared for*, so a
successfully recognised operator can still be refused a session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from features.passport.repository import MachineRecord, OperatorRecord

VALID_CERTIFICATION_STATUSES = frozenset({"valid", "active", "current"})

REASON_MACHINE_TYPE = "machine_type_not_authorized"
REASON_CERT_STATUS = "certification_not_valid"
REASON_CERT_EXPIRED = "certification_expired"


@dataclass(frozen=True)
class AuthorizationResult:
    authorized: bool
    operator_id: str
    machine_id: str
    reasons: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "authorized": self.authorized,
            "operator_id": self.operator_id,
            "machine_id": self.machine_id,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
        }


def check_authorization(
    operator: OperatorRecord,
    machine: MachineRecord,
    as_of: Optional[date] = None,
) -> AuthorizationResult:
    """Every check must pass; each failure is reported with a stable reason code."""
    as_of = as_of or date.today()

    machine_type_ok = machine.machine_type in operator.authorized_machine_types
    status_ok = operator.certification_status.strip().lower() in VALID_CERTIFICATION_STATUSES
    # An absent expiry date is treated as not-expired; the status check still applies.
    not_expired = operator.certification_expiry is None or operator.certification_expiry >= as_of

    reasons = []
    if not machine_type_ok:
        reasons.append(REASON_MACHINE_TYPE)
    if not status_ok:
        reasons.append(REASON_CERT_STATUS)
    if not not_expired:
        reasons.append(REASON_CERT_EXPIRED)

    return AuthorizationResult(
        authorized=not reasons,
        operator_id=operator.operator_id,
        machine_id=machine.machine_id,
        reasons=tuple(reasons),
        checks={
            "machine_type_authorized": machine_type_ok,
            "certification_status_valid": status_ok,
            "certification_not_expired": not_expired,
        },
    )
