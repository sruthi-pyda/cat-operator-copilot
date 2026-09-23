"""Compares a prediction against what actually happened.

Shown only after the fact. The actuals are the model's targets, so they appear
here as outcomes and never as inputs — `DashboardData.build_session_context()`
cannot carry them.

The honest question is not "how close was P50" but **did the actual fall inside
the P10–P90 band**. A point estimate can be far out while the interval was
perfectly reasonable, and an interval wide enough to always contain the answer
is not a good interval either. Both are reported.

A prediction call that returns a zeroed field predicted nothing for it:
`predict_task` populates ETA and zeroes fuel, `predict_fuel` does the reverse.
That is treated as absent rather than as a prediction of zero, so the dashboard
never shows "0.0 L" where it means "not predicted".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

METRIC_DURATION = "duration"
METRIC_FUEL = "fuel"


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    unit: str
    actual: float
    predicted_p50: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None

    @property
    def predicted(self) -> bool:
        return self.predicted_p50 is not None

    @property
    def error(self) -> Optional[float]:
        """Actual minus predicted. Positive means the actual overran the estimate."""
        if not self.predicted:
            return None
        return self.actual - self.predicted_p50

    @property
    def within_interval(self) -> Optional[bool]:
        if not self.predicted or self.p10 is None or self.p90 is None:
            return None
        return self.p10 <= self.actual <= self.p90

    def to_dict(self) -> dict[str, Any]:
        dash = "—"
        return {
            "metric": self.metric,
            "predicted P50": f"{self.predicted_p50:.1f} {self.unit}" if self.predicted else dash,
            "P10-P90": f"{self.p10:.1f} - {self.p90:.1f}" if self.predicted else dash,
            "actual": f"{self.actual:.1f} {self.unit}",
            "error": f"{self.error:+.1f} {self.unit}" if self.predicted else dash,
            "actual within range": (
                "yes" if self.within_interval else "no"
            ) if self.predicted else dash,
        }


def _predicted_or_none(value: Any) -> Optional[float]:
    """A zeroed percentile means this call did not predict that quantity."""
    if value is None:
        return None
    number = float(value)
    return None if number == 0.0 else number


def compare_prediction_to_outcome(
    prediction: Any, outcome: dict[str, Any]
) -> tuple[ComparisonRow, ...]:
    """One row per predictable quantity, whether or not it was predicted."""
    eta_p50 = _predicted_or_none(getattr(prediction, "eta_p50", None))
    fuel_p50 = _predicted_or_none(getattr(prediction, "fuel_p50", None))

    return (
        ComparisonRow(
            metric=METRIC_DURATION,
            unit="min",
            actual=float(outcome["actual_task_duration_min"]),
            predicted_p50=eta_p50,
            p10=_predicted_or_none(getattr(prediction, "eta_p10", None)) if eta_p50 else None,
            p90=_predicted_or_none(getattr(prediction, "eta_p90", None)) if eta_p50 else None,
        ),
        ComparisonRow(
            metric=METRIC_FUEL,
            unit="L",
            actual=float(outcome["actual_fuel_used_l"]),
            predicted_p50=fuel_p50,
            p10=_predicted_or_none(getattr(prediction, "fuel_p10", None)) if fuel_p50 else None,
            p90=_predicted_or_none(getattr(prediction, "fuel_p90", None)) if fuel_p50 else None,
        ),
    )
