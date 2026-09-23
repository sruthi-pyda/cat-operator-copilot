"""Reads the operator and machine tables that back the Operator Passport.

The Passport supplies context to other features. Nothing here scores or ranks an
operator -- baselines are carried through as reference values, not judgements.

CSV conventions shared with scripts/generate_synthetic_data.py:
  - `authorized_machine_types` is multi-value, delimited by `|`
    (`;` and `,` are also accepted when reading).
  - dates are ISO-8601 (`YYYY-MM-DD` or a full timestamp).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from shared.config import load_settings, resolve_path

_LIST_DELIMITERS = ("|", ";", ",")
_TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in _TRUE_VALUES


def _parse_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip().strip("[]")
    if not text:
        return ()
    for delimiter in _LIST_DELIMITERS:
        if delimiter in text:
            parts = text.split(delimiter)
            return tuple(p.strip().strip("'\"") for p in parts if p.strip())
    return (text.strip("'\""),)


def _parse_date(value: Any) -> Optional[date]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


@dataclass(frozen=True)
class OperatorBaseline:
    """Reference values for this operator, used to contextualise observations."""

    idle_ratio: Optional[float] = None
    cycle_time_sec: Optional[float] = None
    fuel_l_per_cycle: Optional[float] = None
    safety_event_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "baseline_idle_ratio": self.idle_ratio,
            "baseline_cycle_time_sec": self.cycle_time_sec,
            "baseline_fuel_l_per_cycle": self.fuel_l_per_cycle,
            "baseline_safety_event_rate": self.safety_event_rate,
        }


@dataclass(frozen=True)
class OperatorRecord:
    operator_id: str
    name: str = ""
    role: str = ""
    years_experience: Optional[float] = None
    machine_skill_level: str = ""
    task_skill_level: str = ""
    authorized_machine_types: tuple[str, ...] = ()
    certification_status: str = ""
    certification_expiry: Optional[date] = None
    baseline: OperatorBaseline = OperatorBaseline()
    recent_workload_hours: Optional[float] = None
    fatigue_proxy: Optional[float] = None
    training_status: str = ""
    face_registered: bool = False
    face_embedding_path: Optional[str] = None
    face_model: Optional[str] = None
    data_source: str = "synthetic"
    synthetic_flag: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OperatorRecord":
        operator_id = str(row.get("operator_id", "")).strip()
        if not operator_id:
            raise ValueError("operators row is missing operator_id")
        return cls(
            operator_id=operator_id,
            name=str(row.get("name", "") or ""),
            role=str(row.get("role", "") or ""),
            years_experience=_parse_float(row.get("years_experience")),
            machine_skill_level=str(row.get("machine_skill_level", "") or ""),
            task_skill_level=str(row.get("task_skill_level", "") or ""),
            authorized_machine_types=_parse_list(row.get("authorized_machine_types")),
            certification_status=str(row.get("certification_status", "") or ""),
            certification_expiry=_parse_date(row.get("certification_expiry")),
            baseline=OperatorBaseline(
                idle_ratio=_parse_float(row.get("baseline_idle_ratio")),
                cycle_time_sec=_parse_float(row.get("baseline_cycle_time_sec")),
                fuel_l_per_cycle=_parse_float(row.get("baseline_fuel_l_per_cycle")),
                safety_event_rate=_parse_float(row.get("baseline_safety_event_rate")),
            ),
            recent_workload_hours=_parse_float(row.get("recent_workload_hours")),
            fatigue_proxy=_parse_float(row.get("fatigue_proxy")),
            training_status=str(row.get("training_status", "") or ""),
            face_registered=_parse_bool(row.get("face_registered")),
            face_embedding_path=(row.get("face_embedding_path") or None),
            face_model=(row.get("face_model") or None),
            data_source=str(row.get("data_source", "synthetic") or "synthetic"),
            synthetic_flag=_parse_bool(row.get("synthetic_flag"), default=True),
        )


@dataclass(frozen=True)
class MachineRecord:
    machine_id: str
    machine_model: str = ""
    machine_type: str = ""
    machine_condition: str = ""
    engine_hours: Optional[float] = None
    attachment_type: str = ""
    fuel_capacity_l: Optional[float] = None
    current_fuel_level_pct: Optional[float] = None
    maintenance_status: str = ""
    maintenance_due_hours: Optional[float] = None
    engine_health_score: Optional[float] = None
    hydraulic_health_score: Optional[float] = None
    data_source: str = "synthetic"
    synthetic_flag: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MachineRecord":
        machine_id = str(row.get("machine_id", "")).strip()
        if not machine_id:
            raise ValueError("machines row is missing machine_id")
        return cls(
            machine_id=machine_id,
            machine_model=str(row.get("machine_model", "") or ""),
            machine_type=str(row.get("machine_type", "") or ""),
            machine_condition=str(row.get("machine_condition", "") or ""),
            engine_hours=_parse_float(row.get("engine_hours")),
            attachment_type=str(row.get("attachment_type", "") or ""),
            fuel_capacity_l=_parse_float(row.get("fuel_capacity_l")),
            current_fuel_level_pct=_parse_float(row.get("current_fuel_level_pct")),
            maintenance_status=str(row.get("maintenance_status", "") or ""),
            maintenance_due_hours=_parse_float(row.get("maintenance_due_hours")),
            engine_health_score=_parse_float(row.get("engine_health_score")),
            hydraulic_health_score=_parse_float(row.get("hydraulic_health_score")),
            data_source=str(row.get("data_source", "synthetic") or "synthetic"),
            synthetic_flag=_parse_bool(row.get("synthetic_flag"), default=True),
        )

    def maintenance_indicators(self) -> dict[str, Any]:
        return {
            "maintenance_status": self.maintenance_status,
            "maintenance_due_hours": self.maintenance_due_hours,
            "engine_health_score": self.engine_health_score,
            "hydraulic_health_score": self.hydraulic_health_score,
        }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required table not found: {path}. "
            "Run scripts/generate_synthetic_data.py first."
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class PassportRepository:
    """In-memory lookup over operators.csv and machines.csv."""

    def __init__(self, operators_path: Path, machines_path: Path) -> None:
        self._operators = {
            record.operator_id: record
            for record in (OperatorRecord.from_row(r) for r in _read_csv(operators_path))
        }
        self._machines = {
            record.machine_id: record
            for record in (MachineRecord.from_row(r) for r in _read_csv(machines_path))
        }

    @classmethod
    def from_settings(cls, settings: Optional[dict[str, Any]] = None) -> "PassportRepository":
        settings = settings or load_settings()
        synthetic_dir = resolve_path(settings["data"]["synthetic_dir"])
        return cls(synthetic_dir / "operators.csv", synthetic_dir / "machines.csv")

    def get_operator(self, operator_id: str) -> Optional[OperatorRecord]:
        return self._operators.get(operator_id)

    def get_machine(self, machine_id: str) -> Optional[MachineRecord]:
        return self._machines.get(machine_id)

    def operators(self) -> list[OperatorRecord]:
        return list(self._operators.values())

    def face_registered_operators(self) -> list[OperatorRecord]:
        return [op for op in self._operators.values() if op.face_registered]
