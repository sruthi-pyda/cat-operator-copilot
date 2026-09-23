"""Structured retrieval for the Grounded Buddy.

A deliberately small router, not a RAG system: each trusted source declares the
words that make it relevant, and contributes an Evidence item carrying text that
already exists somewhere authoritative. Nothing here composes new operating
advice -- that is the point of grounding.

Sources, in the order the architecture lists them: current session, Operator
Passport, telemetry, predictions, safety incidents, training state, task plan,
approved manual snippets.

The manual matters more than its size suggests. `buddy.ask()` discards every
non-safety source once a question is classified safety-critical, so without
manual snippets a safety question can only ever defer. With them, the Buddy can
explain what the approved guidance says while still leaving the decision to the
Safety Guardian.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from features.buddy.evidence import (
    SOURCE_MACHINE_MANUAL,
    SOURCE_PASSPORT,
    SOURCE_PREDICTION,
    SOURCE_SAFETY_INCIDENT,
    SOURCE_SESSION,
    SOURCE_TASK_PLAN,
    SOURCE_TELEMETRY,
    SOURCE_TRAINING_STATE,
    Evidence,
)
from shared.config import PROJECT_ROOT, load_yaml

MANUAL_PATH = PROJECT_ROOT / "features" / "buddy" / "content" / "machine_manual.yaml"

FUEL_WORDS = ("fuel", "diesel", "tank", "refuel", "range")
TASK_WORDS = ("task", "next", "order", "plan", "sequence", "schedule", "job", "change")
ETA_WORDS = ("eta", "how long", "duration", "finish", "remaining", "time left")
TRAINING_WORDS = ("training", "lesson", "quiz", "coach", "course", "module")
OPERATOR_WORDS = (
    "who am i", "certification", "certified", "authorised", "authorized",
    "licence", "license", "my profile", "my skill", "my baseline", "operator",
)
MACHINE_WORDS = (
    "machine", "excavator", "engine", "hours", "attachment", "condition",
    "maintenance", "state", "idle", "speed",
)
SAFETY_WORDS = (
    "safe", "safety", "danger", "hazard", "risk", "worker", "proximity", "swing",
    "emergency", "stop", "injur", "accident", "incident", "collision", "seatbelt",
    "evacuat", "rollover", "blind", "slope", "load", "lift",
)


@dataclass(frozen=True)
class ManualSnippet:
    snippet_id: str
    topic: str
    title: str
    keywords: tuple[str, ...]
    content: str
    synthetic_flag: bool = True


def _mentions(question: str, words: Iterable[str]) -> bool:
    text = (question or "").lower()
    return any(word in text for word in words)


def load_manual(path: Optional[Path] = None) -> tuple[ManualSnippet, ...]:
    document = load_yaml(path or MANUAL_PATH)
    synthetic = bool(document.get("meta", {}).get("synthetic_flag", True))
    return tuple(
        ManualSnippet(
            snippet_id=entry["snippet_id"],
            topic=entry.get("topic", ""),
            title=entry.get("title", ""),
            keywords=tuple(entry.get("keywords", ())),
            content=" ".join(str(entry.get("content", "")).split()),
            synthetic_flag=synthetic,
        )
        for entry in document.get("snippets", [])
    )


def match_snippets(
    question: str, snippets: Optional[Sequence[ManualSnippet]] = None
) -> tuple[ManualSnippet, ...]:
    """Snippets whose keywords, topic or title appear in the question."""
    snippets = snippets if snippets is not None else load_manual()
    text = (question or "").lower()
    if not text:
        return ()
    matched = [
        snippet
        for snippet in snippets
        if any(keyword.lower() in text for keyword in snippet.keywords)
        or snippet.topic.replace("_", " ") in text
        or snippet.title.lower() in text
    ]
    return tuple(matched)


def _value(row: Any, field: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(field)
    return getattr(row, field, None)


def retrieve(
    question: str,
    session_context: Any = None,
    telemetry_row: Any = None,
    safety_events: Sequence[dict[str, Any]] = (),
    task_plan: Any = None,
    prediction: Any = None,
    training_state: Any = None,
    manual: Optional[Sequence[ManualSnippet]] = None,
) -> tuple[Evidence, ...]:
    """Evidence from every approved source the question touches.

    Returned highest-authority first so a caller inspecting the list sees the
    source that would win a conflict at the top.
    """
    timestamp = getattr(session_context, "timestamp", None)
    evidence: list[Evidence] = []

    for snippet in match_snippets(question, manual):
        evidence.append(
            Evidence(
                source=SOURCE_MACHINE_MANUAL,
                value=snippet.snippet_id,
                content=f"{snippet.title}: {snippet.content}",
                confidence=0.95,
                synthetic_flag=snippet.synthetic_flag,
            )
        )

    for event in safety_events:
        evidence.append(
            Evidence(
                source=SOURCE_SAFETY_INCIDENT,
                value=event.get("severity"),
                content=(
                    f"A {event.get('severity')} safety event was recorded: "
                    f"{event.get('trigger_reason')}. Required action: "
                    f"{event.get('required_action')}."
                ),
                timestamp=event.get("timestamp"),
                confidence=float(event.get("confidence", 0.9) or 0.9),
                synthetic_flag=bool(event.get("synthetic_flag", True)),
            )
        )

    if telemetry_row is not None:
        if _mentions(question, FUEL_WORDS):
            level = _value(telemetry_row, "fuel_level_pct")
            if level is not None:
                evidence.append(
                    Evidence(
                        source=SOURCE_TELEMETRY, value=level,
                        content=f"Fuel level is {level} percent.",
                        timestamp=_value(telemetry_row, "timestamp"),
                        confidence=0.9,
                    )
                )
        if _mentions(question, MACHINE_WORDS + SAFETY_WORDS):
            state = _value(telemetry_row, "machine_state")
            if state is not None:
                evidence.append(
                    Evidence(
                        source=SOURCE_TELEMETRY, value=state,
                        content=f"The machine is currently {state}.",
                        timestamp=_value(telemetry_row, "timestamp"),
                        confidence=0.9,
                    )
                )

    if session_context is not None and _mentions(question, OPERATOR_WORDS):
        operator = getattr(session_context, "operator", None)
        if operator is not None:
            evidence.append(
                Evidence(
                    source=SOURCE_PASSPORT, value=session_context.operator_id,
                    content=(
                        f"Operator {session_context.operator_id} is recorded as "
                        f"{operator.machine_skill or 'unrated'} on this machine type, "
                        f"authorization {operator.authorization or 'unknown'}."
                    ),
                    timestamp=timestamp, confidence=0.85,
                )
            )

    if task_plan is not None and _mentions(question, TASK_WORDS):
        evidence.append(
            Evidence(
                source=SOURCE_TASK_PLAN,
                value=getattr(task_plan, "optimized_sequence", None) or str(task_plan),
                content=str(getattr(task_plan, "reason", task_plan)),
                timestamp=timestamp, confidence=0.8,
            )
        )
    elif session_context is not None and _mentions(question, TASK_WORDS):
        task = getattr(session_context, "task", None)
        if task is not None:
            evidence.append(
                Evidence(
                    source=SOURCE_SESSION, value=session_context.task_id,
                    content=(
                        f"The current task is {session_context.task_id} "
                        f"({task.task_type or 'unspecified type'})."
                    ),
                    timestamp=timestamp, confidence=0.75,
                )
            )

    if prediction is not None and _mentions(question, ETA_WORDS + FUEL_WORDS):
        evidence.append(
            Evidence(
                source=SOURCE_PREDICTION, value=getattr(prediction, "eta_p50", None),
                content=(
                    f"Estimated duration is {getattr(prediction, 'eta_p50', '?')} minutes "
                    f"(P10 {getattr(prediction, 'eta_p10', '?')} to "
                    f"P90 {getattr(prediction, 'eta_p90', '?')})."
                ),
                timestamp=timestamp,
                confidence=float(getattr(prediction, "confidence", 0.5) or 0.5),
            )
        )

    if training_state is not None and _mentions(question, TRAINING_WORDS):
        evidence.append(
            Evidence(
                source=SOURCE_TRAINING_STATE, value=training_state,
                content=f"Training state: {training_state}.",
                timestamp=timestamp, confidence=0.8,
            )
        )

    evidence.sort(key=lambda item: item.authority, reverse=True)
    return tuple(evidence)
