from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class OperatorContext:
    operator_id: str
    experience: Optional[float] = None
    machine_skill: Optional[str] = None
    task_skill: Optional[str] = None
    certifications: Optional[List[str]] = None
    authorization: Optional[str] = None
    personal_baseline: Optional[Dict[str, Any]] = None
    recent_workload: Optional[float] = None
    fatigue_proxy: Optional[float] = None


@dataclass
class MachineContext:
    machine_id: str
    machine_model: Optional[str] = None
    machine_type: Optional[str] = None
    machine_condition: Optional[str] = None
    engine_hours: Optional[float] = None
    engine_state: Optional[str] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    load: Optional[float] = None
    attachment: Optional[str] = None
    fuel_level: Optional[float] = None
    maintenance_indicators: Optional[Dict[str, Any]] = None


@dataclass
class TaskContext:
    task_id: str
    task_type: Optional[str] = None
    task_phase: Optional[str] = None
    workload: Optional[str] = None
    target_output: Optional[float] = None
    priority: Optional[int] = None
    deadline: Optional[str] = None
    dependencies: Optional[List[str]] = None
    location: Optional[str] = None
    planned_sequence: Optional[int] = None


@dataclass
class SiteContext:
    soil_material: Optional[str] = None
    moisture: Optional[float] = None
    hardness: Optional[float] = None
    slope: Optional[float] = None
    surface: Optional[str] = None
    route: Optional[str] = None
    work_zone_constraints: Optional[str] = None
    haul_distance: Optional[float] = None
    congestion: Optional[str] = None


@dataclass
class TrafficContext:
    worker_proximity: Optional[float] = None
    relative_direction: Optional[str] = None
    closing_speed: Optional[float] = None
    equipment_density: Optional[int] = None
    congestion: Optional[str] = None
    blocked_routes: Optional[bool] = None


@dataclass
class WeatherContext:
    rain_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    heat_index_c: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    visibility_m: Optional[float] = None
    dust_level: Optional[str] = None
    day_night: Optional[str] = None


@dataclass
class TemporalContext:
    timestamp: Optional[str] = None
    shift_elapsed_min: Optional[float] = None
    cumulative_engine_hours: Optional[float] = None
    recent_idle_ratio: Optional[float] = None
    recent_load_ratio: Optional[float] = None
    recent_heat: Optional[float] = None
    recent_incidents: Optional[int] = None
    task_sequence: Optional[int] = None


@dataclass
class DataQuality:
    source: str = "synthetic"
    timestamp: Optional[str] = None
    freshness: Optional[str] = None
    missingness: Optional[float] = None
    synthetic_flag: bool = True
    confidence: float = 0.85
    source_agreement: Optional[float] = None


@dataclass
class SessionContext:
    session_id: str
    operator_id: str
    machine_id: str
    task_id: Optional[str] = None
    timestamp: Optional[str] = None
    operator: Optional[OperatorContext] = None
    machine: Optional[MachineContext] = None
    task: Optional[TaskContext] = None
    site: Optional[SiteContext] = None
    traffic: Optional[TrafficContext] = None
    weather: Optional[WeatherContext] = None
    temporal: Optional[TemporalContext] = None
    data_quality: Optional[DataQuality] = None


@dataclass
class PredictionResult:
    session_id: str
    eta_p10: float
    eta_p50: float
    eta_p90: float
    fuel_p10: float
    fuel_p50: float
    fuel_p90: float
    confidence: float
    factors: List[str]
    predicted_vs_actual: Optional[float]
    synthetic_flag: bool
    data_quality_confidence: float


@dataclass
class SafetyEvent:
    event_id: str
    severity: str
    trigger_reason: str
    required_action: str
    confidence: float
    timestamp: str
    synthetic_flag: bool
    # Optional context (added by Safety Guardian; defaults keep older callers valid)
    recommendation: str = ""
    session_id: Optional[str] = None
    operator_id: Optional[str] = None
    machine_id: Optional[str] = None
    rule_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorResult:
    session_id: str
    attribution: str
    confidence: float
    observed_value: float
    expected_value: float
    operator_residual: float
    context_explained_component: float
    coaching_eligible: bool
    synthetic_flag: bool


@dataclass
class TrainingTrigger:
    trigger_event_id: str
    operator_id: str
    issue_type: str
    attribution_type: str
    confidence: float
    lesson_id: str
    escalation_state: str
    synthetic_flag: bool


@dataclass
class AttentionEvent:
    event_id: str
    event_type: str
    severity: str
    urgency: str
    actionability: str
    confidence: float
    operator_state: str
    decision: str
    reason: str
    timestamp: str
