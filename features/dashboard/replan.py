"""Describes what changed in the operating context, for the replan banner.

Ownership boundary: deciding *whether* to replan is `should_replan()` in the
Optimization feature (Member 2). This module does not decide anything -- it only
puts a reason in words so the dashboard can say **why** a plan changed instead
of silently showing a new order.

    Plan changed
    Reason: rain increased + congestion increased

Until the optimizer lands, the banner reports the context change alone and says
plainly that no plan has been recalculated. It never implies a reordering that
nobody computed.

Per-field minimum deltas exist so sensor jitter does not read as a change: a
0.1 mm difference in rain is noise, 2 mm is weather.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

DIRECTION_UP = "increased"
DIRECTION_DOWN = "decreased"
DIRECTION_CHANGED = "changed"


@dataclass(frozen=True)
class WatchedField:
    key: str
    label: str
    getter: Callable[[Any], Any]
    min_delta: Optional[float] = None
    order: Optional[tuple[str, ...]] = None
    unit: str = ""


def _safe(getter: Callable[[Any], Any], context: Any) -> Any:
    """Sub-contexts are optional -- a Passport session has no site or weather."""
    try:
        return getter(context)
    except AttributeError:
        return None


WATCHED_FIELDS: tuple[WatchedField, ...] = (
    WatchedField("rain_mm", "rain", lambda c: c.weather.rain_mm, min_delta=0.5, unit="mm"),
    WatchedField("visibility_m", "visibility", lambda c: c.weather.visibility_m,
                 min_delta=100.0, unit="m"),
    WatchedField("temperature_c", "temperature", lambda c: c.weather.temperature_c,
                 min_delta=3.0, unit="C"),
    WatchedField("dust_level", "dust", lambda c: c.weather.dust_level,
                 order=("low", "medium", "high")),
    WatchedField("congestion", "congestion", lambda c: c.traffic.congestion,
                 order=("low", "medium", "high")),
    WatchedField("soil_hardness_index", "soil hardness", lambda c: c.site.hardness,
                 min_delta=0.05),
    WatchedField("site_slope_deg", "slope", lambda c: c.site.slope, min_delta=1.0, unit="deg"),
    WatchedField("machine_condition", "machine condition",
                 lambda c: c.machine.machine_condition,
                 order=("poor", "fair", "good", "excellent")),
)


@dataclass(frozen=True)
class ConditionChange:
    key: str
    label: str
    previous: Any
    current: Any
    direction: str

    def describe(self) -> str:
        return f"{self.label} {self.direction}"

    def detail(self) -> str:
        return f"{self.label} {self.direction} ({self.previous} to {self.current})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "previous": self.previous,
            "current": self.current,
            "direction": self.direction,
        }


def _compare(field: WatchedField, previous: Any, current: Any) -> Optional[ConditionChange]:
    if previous is None or current is None or previous == current:
        return None

    if field.order is not None:
        order = field.order
        text_previous, text_current = str(previous).lower(), str(current).lower()
        if text_previous in order and text_current in order:
            direction = (
                DIRECTION_UP if order.index(text_current) > order.index(text_previous)
                else DIRECTION_DOWN
            )
        else:
            direction = DIRECTION_CHANGED
        return ConditionChange(field.key, field.label, previous, current, direction)

    try:
        delta = float(current) - float(previous)
    except (TypeError, ValueError):
        return ConditionChange(field.key, field.label, previous, current, DIRECTION_CHANGED)

    if field.min_delta is not None and abs(delta) < field.min_delta:
        return None
    return ConditionChange(
        field.key, field.label, previous, current,
        DIRECTION_UP if delta > 0 else DIRECTION_DOWN,
    )


def describe_context_change(previous: Any, current: Any) -> tuple[ConditionChange, ...]:
    """Every watched field that moved by more than its noise threshold."""
    if previous is None or current is None:
        return ()
    changes = []
    for field in WATCHED_FIELDS:
        change = _compare(field, _safe(field.getter, previous), _safe(field.getter, current))
        if change is not None:
            changes.append(change)
    return tuple(changes)


def replan_reason(changes: Sequence[ConditionChange]) -> str:
    """The one-line reason shown under 'Plan changed'."""
    if not changes:
        return "no material change in operating context"
    return " + ".join(change.describe() for change in changes)
