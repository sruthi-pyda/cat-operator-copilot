"""Safe-state gate for the Grounded AI Operating Buddy (Feature 07).

The Buddy is an assistant, never a safety authority. This gate decides whether
it may interact with the operator at all.

Two rules from the architecture, both required:

  1. the machine is parked OR in a *verified* safe idle, and
  2. there is no active arm / bucket / attachment movement.

Merely "not travelling" is explicitly not enough -- a stationary machine with a
swinging arm is not a safe moment to start a conversation.

The list of safe machine states is read from `config/safety_rules.yaml`, which
the Safety Guardian owns. The Buddy deliberately does not keep its own
definition of "safe": it obeys the safety configuration rather than competing
with it.

**Unknown state denies.** If machine state or attachment movement is missing,
the gate refuses. Absent telemetry is not evidence of safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from shared.config import SAFETY_RULES_PATH, load_settings, load_yaml

# Telemetry vocabularies: which reported values mean "this part is not moving".
# These map raw telemetry strings to motion, not to a tunable threshold.
STATIONARY_ATTACHMENT_MOVEMENTS = frozenset(
    {"none", "no_movement", "idle", "static", "stationary", "stowed", "parked"}
)
MOVING_BUCKET_STATES = frozenset(
    {"digging", "dumping", "curling", "lifting", "loading", "swinging", "carrying"}
)

REASON_STATE_UNKNOWN = "machine_state_unknown"
REASON_STATE_NOT_SAFE = "machine_state_not_safe"
REASON_ATTACHMENT_UNKNOWN = "attachment_movement_unknown"
REASON_ATTACHMENT_MOVING = "attachment_movement_active"
REASON_ARM_MOVING = "arm_in_motion"
REASON_BUCKET_MOVING = "bucket_in_motion"
REASON_MACHINE_MOVING = "machine_in_motion"


@dataclass(frozen=True)
class MachineStateSnapshot:
    """Live telemetry describing what the machine is doing right now."""

    machine_state: Optional[str] = None
    attachment_movement: Optional[str] = None
    arm_speed: Optional[float] = None
    bucket_state: Optional[str] = None
    machine_speed_kmh: Optional[float] = None


@dataclass(frozen=True)
class SafeStateResult:
    allowed: bool
    reasons: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
        }


def _normalise(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def load_safe_states(safety_rules: Optional[dict] = None) -> frozenset[str]:
    """The Safety Guardian's own list of safe machine states."""
    rules = safety_rules if safety_rules is not None else load_yaml(SAFETY_RULES_PATH)
    states = rules.get("machine_state", {}).get("safe_states", [])
    return frozenset(_normalise(s) for s in states if _normalise(s))


def evaluate_safe_state(
    snapshot: MachineStateSnapshot,
    safe_states: Optional[frozenset[str]] = None,
    settings: Optional[dict] = None,
) -> SafeStateResult:
    safe_states = safe_states if safe_states is not None else load_safe_states()
    settings = settings if settings is not None else load_settings()
    require_no_attachment_movement = settings.get("buddy", {}).get(
        "no_attachment_movement_required", True
    )

    reasons: list[str] = []

    state = _normalise(snapshot.machine_state)
    state_known = state is not None
    state_safe = state_known and state in safe_states
    if not state_known:
        reasons.append(REASON_STATE_UNKNOWN)
    elif not state_safe:
        reasons.append(REASON_STATE_NOT_SAFE)

    attachment = _normalise(snapshot.attachment_movement)
    attachment_known = attachment is not None
    attachment_still = attachment_known and attachment in STATIONARY_ATTACHMENT_MOVEMENTS
    if require_no_attachment_movement:
        if not attachment_known:
            reasons.append(REASON_ATTACHMENT_UNKNOWN)
        elif not attachment_still:
            reasons.append(REASON_ATTACHMENT_MOVING)

    arm_still = not (snapshot.arm_speed is not None and snapshot.arm_speed > 0)
    if not arm_still:
        reasons.append(REASON_ARM_MOVING)

    bucket = _normalise(snapshot.bucket_state)
    bucket_still = bucket is None or bucket not in MOVING_BUCKET_STATES
    if not bucket_still:
        reasons.append(REASON_BUCKET_MOVING)

    machine_still = not (
        snapshot.machine_speed_kmh is not None and snapshot.machine_speed_kmh > 0
    )
    if not machine_still:
        reasons.append(REASON_MACHINE_MOVING)

    return SafeStateResult(
        allowed=not reasons,
        reasons=tuple(reasons),
        checks={
            "machine_state_safe": state_safe,
            "attachment_stationary": attachment_still,
            "arm_stationary": arm_still,
            "bucket_stationary": bucket_still,
            "machine_stationary": machine_still,
        },
    )
