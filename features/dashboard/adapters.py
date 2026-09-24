"""Ports from the dashboard to the other features.

The dashboard calls the documented API-contract functions and never imports a
teammate's internal model code. Each feature is resolved lazily by name, so the
dashboard runs before Safety, Behavior, Prediction, Optimization and Attention
exist.

When a feature is not integrated yet the adapter returns
`available=False` with a reason, and the UI says so. It never substitutes a
placeholder number: a fabricated ETA on screen is worse than a blank one,
because it looks like a result.

Only ImportError and AttributeError are treated as "not integrated". If a
teammate's function exists and raises, that is a real defect and it propagates
rather than being reported as a missing feature.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from shared.schemas import SessionContext

FEATURE_PREDICTION = "prediction"
FEATURE_SAFETY = "safety"
FEATURE_BEHAVIOR = "behavior"
FEATURE_OPTIMIZATION = "optimization"
FEATURE_ATTENTION = "attention"

REASON_NOT_INTEGRATED = "feature_not_integrated_yet"


@dataclass(frozen=True)
class AdapterResult:
    available: bool
    feature: str
    value: Any = None
    reason: str = ""
    owner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "feature": self.feature,
            "reason": self.reason,
            "owner": self.owner,
        }


# Who to chase when something is missing, shown in the dashboard's status strip.
FEATURE_OWNERS = {
    FEATURE_PREDICTION: "Member 1",
    FEATURE_BEHAVIOR: "Member 1",
    FEATURE_SAFETY: "Member 2",
    FEATURE_OPTIMIZATION: "Member 2",
    FEATURE_ATTENTION: "Member 2",
}


# Where a contract function may live. The package itself is the intended home --
# `from .api import predict_task` in features/prediction/__init__.py makes the
# contract explicit. The extra paths exist because a package __init__ left empty
# is an easy thing to forget, and a silently unintegrated feature shows on the
# dashboard as "not integrated yet" rather than as an error anyone would chase.
#
# `features.prediction.api` is listed under behavior because that is where
# `analyze_behavior` is actually implemented, not a guess.
CONTRACT_MODULES: dict[str, tuple[str, ...]] = {
    FEATURE_PREDICTION: ("features.prediction", "features.prediction.api"),
    FEATURE_BEHAVIOR: (
        "features.behavior",
        "features.behavior.behavior_model",
        "features.prediction.api",
    ),
    FEATURE_SAFETY: ("features.safety", "features.safety.api"),
    FEATURE_OPTIMIZATION: ("features.optimization", "features.optimization.api"),
    FEATURE_ATTENTION: ("features.attention", "features.attention.api"),
}


def resolve(feature: str, function_name: str) -> Optional[Callable]:
    """Return the contract function for a feature, or None if not integrated."""
    for module_path in CONTRACT_MODULES.get(feature, (f"features.{feature}",)):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        function = getattr(module, function_name, None)
        if callable(function):
            return function
    return None


def _call(feature: str, function_name: str, *args, **kwargs) -> AdapterResult:
    function = resolve(feature, function_name)
    owner = FEATURE_OWNERS.get(feature, "")
    if function is None:
        return AdapterResult(
            available=False,
            feature=feature,
            reason=f"{feature}.{function_name} {REASON_NOT_INTEGRATED}",
            owner=owner,
        )
    return AdapterResult(available=True, feature=feature, value=function(*args, **kwargs), owner=owner)


def get_prediction(session_context: SessionContext) -> AdapterResult:
    """Prefer `predict_combined`: it populates ETA *and* fuel.

    `predict_task` returns ETA with the fuel fields zeroed and `predict_fuel`
    does the reverse, so using either alone would put a 0.0 on screen next to a
    real number, which reads as a prediction rather than as an absence.
    """
    for function_name in ("predict_combined", "predict_task"):
        if resolve(FEATURE_PREDICTION, function_name) is not None:
            return _call(FEATURE_PREDICTION, function_name, session_context)
    return _call(FEATURE_PREDICTION, "predict_combined", session_context)


def get_safety(session_context: SessionContext) -> AdapterResult:
    return _call(FEATURE_SAFETY, "evaluate_safety", session_context)


def get_behavior(session_context: SessionContext) -> AdapterResult:
    return _call(FEATURE_BEHAVIOR, "analyze_behavior", session_context)


def get_plan(tasks: Any, session_context: SessionContext) -> AdapterResult:
    return _call(FEATURE_OPTIMIZATION, "generate_plan", tasks, session_context)


def route_event(event: Any) -> AdapterResult:
    return _call(FEATURE_ATTENTION, "route_event", event)


def integration_status() -> dict[str, AdapterResult]:
    """What is wired up right now -- rendered as the dashboard's status strip."""
    expected = {
        FEATURE_PREDICTION: "predict_task",
        FEATURE_BEHAVIOR: "analyze_behavior",
        FEATURE_SAFETY: "evaluate_safety",
        FEATURE_OPTIMIZATION: "generate_plan",
        FEATURE_ATTENTION: "route_event",
    }
    status = {}
    for feature, function_name in expected.items():
        available = resolve(feature, function_name) is not None
        status[feature] = AdapterResult(
            available=available,
            feature=feature,
            reason="" if available else f"{feature}.{function_name} {REASON_NOT_INTEGRATED}",
            owner=FEATURE_OWNERS.get(feature, ""),
        )
    return status
