"""Telemetry replay -- drives a recorded session forward as if it were live.

A real machine is not needed for the demo: this feeds the stored telemetry rows
of one session in order, and at each step reports what the system would have
decided at that moment.

Each step carries the Buddy's safe-state verdict (a real evaluation, not a
recording) and any safety event the dataset recorded in that interval. Features
that are not integrated yet are reported as unavailable rather than skipped, so
the replay never implies a decision nobody made.

Safety events are matched to the step whose interval contains them -- an event
belongs to the telemetry row it happened during, not the nearest one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import pandas as pd

from features.buddy.safe_state import MachineStateSnapshot, SafeStateResult, evaluate_safe_state
from features.dashboard import adapters
from features.dashboard.data import DashboardData


@dataclass(frozen=True)
class ReplayStep:
    index: int
    timestamp: str
    machine_state: str
    safe_state: SafeStateResult
    worker_distance_m: Optional[float] = None
    closing_speed_mps: Optional[float] = None
    safety_events: tuple[dict[str, Any], ...] = ()
    safety_evaluation: Optional[adapters.AdapterResult] = None

    @property
    def buddy_available(self) -> bool:
        return self.safe_state.allowed

    def summary(self) -> str:
        """One timeline line, in the shape the architecture's demo script uses."""
        clock = self.timestamp[11:19] if len(self.timestamp) >= 19 else self.timestamp
        parts = [f"{clock}  {self.machine_state:<10}"]
        parts.append("buddy:on " if self.buddy_available else "buddy:off")
        if self.worker_distance_m is not None:
            parts.append(f"worker {self.worker_distance_m:>5.1f}m")
        if self.closing_speed_mps:
            parts.append(f"closing {self.closing_speed_mps:>4.1f}m/s")
        for event in self.safety_events:
            parts.append(f"** {event.get('severity')} {event.get('trigger_reason')}")
        return "  ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "machine_state": self.machine_state,
            "buddy_available": self.buddy_available,
            "buddy_reasons": list(self.safe_state.reasons),
            "worker_distance_m": self.worker_distance_m,
            "closing_speed_mps": self.closing_speed_mps,
            "safety_events": [dict(e) for e in self.safety_events],
            "safety_evaluation": (
                self.safety_evaluation.to_dict() if self.safety_evaluation else None
            ),
        }


def _optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


@dataclass
class TelemetryReplay:
    data: DashboardData
    session_id: str
    _telemetry: Optional[pd.DataFrame] = field(default=None, repr=False)
    _events: Optional[pd.DataFrame] = field(default=None, repr=False)

    def telemetry(self) -> pd.DataFrame:
        if self._telemetry is None:
            rows = self.data.telemetry(self.session_id)
            self._telemetry = rows.sort_values("timestamp").reset_index(drop=True)
        return self._telemetry

    def events(self) -> pd.DataFrame:
        if self._events is None:
            self._events = self.data.safety_events(self.session_id)
        return self._events

    def _events_between(self, start: str, end: Optional[str]) -> tuple[dict[str, Any], ...]:
        events = self.events()
        if events.empty:
            return ()
        window = events[events.timestamp >= start]
        if end is not None:
            window = window[window.timestamp < end]
        return tuple(window.to_dict(orient="records"))

    def steps(self, limit: Optional[int] = None) -> Iterator[ReplayStep]:
        rows = self.telemetry()
        if limit is not None:
            rows = rows.head(limit)
        timestamps = rows.timestamp.tolist()

        for index, row in enumerate(rows.itertuples(index=False)):
            snapshot = MachineStateSnapshot(
                machine_state=row.machine_state,
                attachment_movement=row.attachment_movement,
                arm_speed=_optional_float(row.arm_speed),
                bucket_state=row.bucket_state,
                machine_speed_kmh=_optional_float(row.machine_speed_kmh),
            )
            next_timestamp = timestamps[index + 1] if index + 1 < len(timestamps) else None
            yield ReplayStep(
                index=index,
                timestamp=str(row.timestamp),
                machine_state=str(row.machine_state),
                safe_state=evaluate_safe_state(snapshot),
                worker_distance_m=_optional_float(getattr(row, "worker_distance_m", None)),
                closing_speed_mps=_optional_float(getattr(row, "closing_speed_mps", None)),
                safety_events=self._events_between(str(row.timestamp), next_timestamp),
                safety_evaluation=adapters.get_safety(None),
            )

    def run(self, limit: Optional[int] = None, delay_sec: float = 0.0) -> list[ReplayStep]:
        collected = []
        for step in self.steps(limit=limit):
            collected.append(step)
            if delay_sec:
                time.sleep(delay_sec)
        return collected
