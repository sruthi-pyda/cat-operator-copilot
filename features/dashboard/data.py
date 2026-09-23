"""Loads the synthetic tables the dashboard displays.

**Leakage boundary.** `task_sessions.csv` holds both the pre-task context and the
post-task outcomes (`actual_task_duration_min`, `actual_fuel_used_l`,
`actual_idle_time_min`, `actual_cycle_count`). Those four are the prediction
targets, so they are split out here rather than filtered at the point of use:

    pre_task_context(session_id)  -> everything a model may see, actuals removed
    outcome(session_id)           -> only the actuals, for after-the-fact display

`build_session_context()` reads only from the pre-task side, so a SessionContext
handed to any feature cannot carry an outcome. The dashboard shows actuals only
beside an already-made prediction, clearly as what happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from features.passport.repository import PassportRepository
from shared.config import load_settings, resolve_path
from shared.schemas import (
    DataQuality,
    MachineContext,
    OperatorContext,
    SessionContext,
    SiteContext,
    TaskContext,
    TemporalContext,
    TrafficContext,
    WeatherContext,
)

# The prediction targets. Never an input, never part of a SessionContext.
ACTUAL_COLUMNS = (
    "actual_task_duration_min",
    "actual_fuel_used_l",
    "actual_idle_time_min",
    "actual_cycle_count",
)


class MissingDatasetError(FileNotFoundError):
    pass


@dataclass
class DashboardData:
    synthetic_dir: Path

    @classmethod
    def from_settings(cls, settings: Optional[dict[str, Any]] = None) -> "DashboardData":
        settings = settings or load_settings()
        return cls(resolve_path(settings["data"]["synthetic_dir"]))

    def _read(self, name: str) -> pd.DataFrame:
        path = self.synthetic_dir / name
        if not path.exists():
            raise MissingDatasetError(
                f"{path} not found. Take the dataset from the data branch:\n"
                "  git checkout origin/feature/sruthi-data-models -- data/synthetic/\n"
                "  git reset -q data/synthetic/"
            )
        return pd.read_csv(path)

    def available(self) -> bool:
        return (self.synthetic_dir / "task_sessions.csv").exists()

    # --- tables --------------------------------------------------------------

    def sessions(self) -> pd.DataFrame:
        return self._read("task_sessions.csv")

    def tasks(self) -> pd.DataFrame:
        return self._read("tasks.csv")

    def safety_events(self, session_id: Optional[str] = None) -> pd.DataFrame:
        events = self._read("safety_events.csv")
        return events if session_id is None else events[events.session_id == session_id]

    def telemetry(self, session_id: Optional[str] = None) -> pd.DataFrame:
        rows = self._read("telemetry.csv")
        return rows if session_id is None else rows[rows.session_id == session_id]

    def repository(self) -> PassportRepository:
        return PassportRepository(
            self.synthetic_dir / "operators.csv", self.synthetic_dir / "machines.csv"
        )

    # --- leakage boundary ----------------------------------------------------

    def _session_row(self, session_id: str) -> pd.Series:
        sessions = self.sessions()
        match = sessions[sessions.session_id == session_id]
        if match.empty:
            raise KeyError(f"No session {session_id!r}")
        return match.iloc[0]

    def pre_task_context(self, session_id: str) -> dict[str, Any]:
        """Session fields a model may see. The four outcome columns are removed."""
        row = self._session_row(session_id).to_dict()
        return {k: v for k, v in row.items() if k not in ACTUAL_COLUMNS}

    def outcome(self, session_id: str) -> dict[str, Any]:
        """What actually happened. For display beside a prediction, never as input."""
        row = self._session_row(session_id).to_dict()
        return {k: row[k] for k in ACTUAL_COLUMNS if k in row}

    def build_session_context(self, session_id: str) -> SessionContext:
        context = self.pre_task_context(session_id)
        repo = self.repository()
        operator = repo.get_operator(context["operator_id"])
        machine = repo.get_machine(context["machine_id"])

        return SessionContext(
            session_id=session_id,
            operator_id=context["operator_id"],
            machine_id=context["machine_id"],
            task_id=context.get("task_id"),
            timestamp=context.get("start_timestamp"),
            operator=OperatorContext(
                operator_id=context["operator_id"],
                experience=operator.years_experience if operator else None,
                machine_skill=operator.machine_skill_level if operator else None,
                task_skill=operator.task_skill_level if operator else None,
                personal_baseline=operator.baseline.to_dict() if operator else None,
                recent_workload=operator.recent_workload_hours if operator else None,
                fatigue_proxy=operator.fatigue_proxy if operator else None,
            ),
            machine=MachineContext(
                machine_id=context["machine_id"],
                machine_model=machine.machine_model if machine else "",
                machine_type=machine.machine_type if machine else "",
                machine_condition=context.get("machine_condition"),
                engine_hours=context.get("engine_hours"),
                load=context.get("engine_load_pct"),
                attachment=context.get("attachment_type"),
            ),
            task=TaskContext(
                task_id=context.get("task_id"),
                task_type=context.get("task_type"),
                task_phase=context.get("task_phase"),
                workload=context.get("workload"),
                target_output=context.get("target_output"),
                priority=context.get("priority"),
            ),
            site=SiteContext(
                soil_material=context.get("soil_material"),
                moisture=context.get("soil_moisture_pct"),
                hardness=context.get("soil_hardness_index"),
                slope=context.get("site_slope_deg"),
                surface=context.get("surface_type"),
                haul_distance=context.get("travel_distance_km"),
            ),
            traffic=TrafficContext(congestion=context.get("congestion_level")),
            weather=WeatherContext(
                rain_mm=context.get("rain_mm"),
                temperature_c=context.get("temperature_c"),
                heat_index_c=context.get("heat_index_c"),
                visibility_m=context.get("visibility_m"),
                dust_level=context.get("dust_level"),
                day_night=context.get("day_night"),
            ),
            temporal=TemporalContext(
                timestamp=context.get("start_timestamp"),
                shift_elapsed_min=context.get("shift_elapsed_min"),
                cumulative_engine_hours=context.get("operator_cumulative_hours"),
                recent_idle_ratio=context.get("recent_idle_ratio"),
                recent_load_ratio=context.get("recent_load_ratio"),
                recent_incidents=context.get("recent_incident_count"),
            ),
            data_quality=DataQuality(
                source=context.get("data_source", "synthetic"),
                timestamp=context.get("start_timestamp"),
                synthetic_flag=bool(context.get("synthetic_flag", True)),
                confidence=float(context.get("data_quality_confidence", 0.85)),
            ),
        )
