"""Evidence the Buddy is allowed to answer from.

The Buddy is *grounded*: it may only repeat what a trusted source already says.
It retrieves, it does not compose operating instructions.

Every item carries the provenance the architecture requires -- source,
timestamp, freshness, synthetic_flag, confidence, source_agreement -- so the
operator can always see where an answer came from and how old it is.

`SOURCE_AUTHORITY` ranks sources for conflict resolution. Safety incidents and
approved manual snippets outrank everything, which is what makes the Buddy defer
to the Safety Guardian instead of arguing with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from shared.config import load_settings

SOURCE_SAFETY_INCIDENT = "safety_incident"
SOURCE_MACHINE_MANUAL = "machine_manual"
SOURCE_TELEMETRY = "telemetry"
SOURCE_PASSPORT = "passport"
SOURCE_SESSION = "session"
SOURCE_TASK_PLAN = "task_plan"
SOURCE_TRAINING_STATE = "training_state"
SOURCE_PREDICTION = "prediction"

# Higher wins a conflict. Safety and the approved manual are authoritative;
# a prediction is the weakest source because it is an estimate, not an observation.
SOURCE_AUTHORITY: dict[str, int] = {
    SOURCE_SAFETY_INCIDENT: 100,
    SOURCE_MACHINE_MANUAL: 90,
    SOURCE_TELEMETRY: 70,
    SOURCE_PASSPORT: 60,
    SOURCE_TASK_PLAN: 55,
    SOURCE_TRAINING_STATE: 50,
    SOURCE_SESSION: 50,
    SOURCE_PREDICTION: 30,
}

APPROVED_SOURCES = frozenset(SOURCE_AUTHORITY)

FRESHNESS_FRESH = "fresh"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Evidence:
    source: str
    value: Any
    content: str = ""
    timestamp: Optional[str] = None
    synthetic_flag: bool = True
    confidence: float = 0.0
    source_agreement: Optional[float] = None

    @property
    def authority(self) -> int:
        return SOURCE_AUTHORITY.get(self.source, 0)

    def age_minutes(self, as_of: datetime) -> Optional[float]:
        if not self.timestamp:
            return None
        try:
            recorded = datetime.fromisoformat(str(self.timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        if recorded.tzinfo is not None and as_of.tzinfo is None:
            recorded = recorded.replace(tzinfo=None)
        return (as_of - recorded).total_seconds() / 60.0

    def freshness(self, as_of: datetime, max_age_min: float) -> str:
        age = self.age_minutes(as_of)
        if age is None:
            return FRESHNESS_UNKNOWN
        return FRESHNESS_FRESH if age <= max_age_min else FRESHNESS_STALE

    def to_dict(self, as_of: Optional[datetime] = None, max_age_min: float = 60.0) -> dict[str, Any]:
        as_of = as_of or datetime.now()
        return {
            "source": self.source,
            "value": self.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "freshness": self.freshness(as_of, max_age_min),
            "synthetic_flag": self.synthetic_flag,
            "confidence": self.confidence,
            "source_agreement": self.source_agreement,
        }


def max_evidence_age_min(settings: Optional[dict] = None) -> float:
    settings = settings if settings is not None else load_settings()
    return float(settings.get("buddy", {}).get("max_evidence_age_min", 60))


def filter_approved(items: Iterable[Evidence]) -> tuple[Evidence, ...]:
    """Drop anything that did not come from a trusted source."""
    return tuple(item for item in items if item.source in APPROVED_SOURCES)


def partition_by_freshness(
    items: Iterable[Evidence], as_of: datetime, max_age_min: float
) -> tuple[tuple[Evidence, ...], tuple[Evidence, ...]]:
    """Return (usable, stale). Unknown-age evidence is treated as usable but is
    reported as `unknown` freshness so the operator can see it was undated."""
    usable, stale = [], []
    for item in items:
        if item.freshness(as_of, max_age_min) == FRESHNESS_STALE:
            stale.append(item)
        else:
            usable.append(item)
    return tuple(usable), tuple(stale)
