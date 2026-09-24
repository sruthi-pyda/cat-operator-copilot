"""Conflict handling for the Grounded Buddy.

When trusted sources disagree, the architecture allows exactly three outcomes,
in this order:

  1. state the conflict,
  2. use the authoritative source if one clearly exists,
  3. otherwise defer.

Silently picking a winner is not permitted, so a conflict is always reported on
the response even when it was resolvable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from features.buddy.evidence import SOURCE_MACHINE_MANUAL, SOURCE_SAFETY_INCIDENT, Evidence

RESOLUTION_NO_CONFLICT = "no_conflict"
RESOLUTION_BY_AUTHORITY = "resolved_by_authority"
RESOLUTION_DEFER = "unresolved_defer"

NUMERIC_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class ConflictResult:
    conflicted: bool
    resolution: str
    winner: Optional[Evidence] = None
    competing_values: tuple[Any, ...] = ()
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflicted": self.conflicted,
            "resolution": self.resolution,
            "winner_source": self.winner.source if self.winner else None,
            "competing_values": list(self.competing_values),
            "sources": list(self.sources),
        }


def _values_agree(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(left), abs(right), 1.0)
        return abs(left - right) <= NUMERIC_RELATIVE_TOLERANCE * scale
    return str(left).strip().lower() == str(right).strip().lower()


# Sources that naturally report a list rather than a single fact. A session has
# many safety events and several manual snippets can match one question; two of
# them differing is not a contradiction. Every other source answers "what is X",
# so two differing values from it genuinely IS a conflict and must still defer.
MULTI_VALUED_SOURCES = frozenset({SOURCE_SAFETY_INCIDENT, SOURCE_MACHINE_MANUAL})


def _collapse_multi_valued(items: Sequence[Evidence]) -> list[Evidence]:
    """Keep one representative from each naturally multi-valued source.

    Without this, a session holding a CRITICAL and a MEDIUM incident looked like
    one source contradicting itself, and the Buddy deferred on questions it could
    answer. Callers order items most-severe-first, so the representative kept is
    the one that matters.
    """
    collapsed: list[Evidence] = []
    seen: set[str] = set()
    for item in items:
        if item.source in MULTI_VALUED_SOURCES:
            if item.source in seen:
                continue
            seen.add(item.source)
        collapsed.append(item)
    return collapsed


def detect_conflict(items: Sequence[Evidence]) -> ConflictResult:
    if not items:
        return ConflictResult(conflicted=False, resolution=RESOLUTION_NO_CONFLICT)

    sources = tuple(item.source for item in items)
    values = tuple(item.value for item in items)
    everything = _collapse_multi_valued(items)

    # The approved manual states guidance; every other source states a fact about
    # this machine or shift. They answer different questions, so they cannot
    # contradict each other -- comparing them produced "sources disagree" between
    # a refuelling procedure and a fuel gauge. The manual can still win on
    # authority; it just never counts as a dissenting voice.
    factual = [item for item in everything if item.source != SOURCE_MACHINE_MANUAL]
    comparable = factual or everything

    reference = comparable[0].value
    conflicted = any(not _values_agree(reference, item.value) for item in comparable[1:])

    if not conflicted:
        winner = max(everything, key=lambda item: item.authority)
        return ConflictResult(
            conflicted=False,
            resolution=RESOLUTION_NO_CONFLICT,
            winner=winner,
            competing_values=values,
            sources=sources,
        )

    ranked = sorted(comparable, key=lambda item: item.authority, reverse=True)
    top_authority = ranked[0].authority
    tied = [item for item in ranked if item.authority == top_authority]

    # A tie at the top means no source outranks the other: defer rather than guess.
    if len(tied) > 1:
        return ConflictResult(
            conflicted=True,
            resolution=RESOLUTION_DEFER,
            competing_values=values,
            sources=sources,
        )

    return ConflictResult(
        conflicted=True,
        resolution=RESOLUTION_BY_AUTHORITY,
        winner=ranked[0],
        competing_values=values,
        sources=sources,
    )
