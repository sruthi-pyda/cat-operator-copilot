"""Peer learning — part of Training, not a separate feature.

Shows how other operators handled a comparable task in comparable conditions.
Two rules make it shareable rather than surveillance:

  1. only examples explicitly flagged `approved_for_peer_learning` are ever
     returned, and
  2. the source is an anonymised id, never a real operator id.

Matching is on context, not on ranking people: task type and machine type must
agree, and `context_similarity_score` orders what is left. An example from an
easy site is not advice for someone working hard ground, so a similarity floor
applies and the score is always shown alongside the technique.

Nothing here says one operator is better than another. It says "this worked in
conditions like yours".
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from shared.config import load_settings, resolve_path

DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_LIMIT = 5

_TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class PeerExample:
    example_id: str
    source_operator_anonymized_id: str
    task_type: str
    machine_type: str
    attachment_type: str
    site_condition: str
    technique_description: str
    observed_metric: str
    context_similarity_score: float
    approved_for_peer_learning: bool
    synthetic_flag: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PeerExample":
        return cls(
            example_id=str(row.get("example_id", "")),
            source_operator_anonymized_id=str(row.get("source_operator_anonymized_id", "")),
            task_type=str(row.get("task_type", "") or ""),
            machine_type=str(row.get("machine_type", "") or ""),
            attachment_type=str(row.get("attachment_type", "") or ""),
            site_condition=str(row.get("site_condition", "") or ""),
            technique_description=str(row.get("technique_description", "") or ""),
            observed_metric=str(row.get("observed_metric", "") or ""),
            context_similarity_score=_parse_float(row.get("context_similarity_score")),
            approved_for_peer_learning=_parse_bool(row.get("approved_for_peer_learning")),
            synthetic_flag=_parse_bool(row.get("synthetic_flag")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source": self.source_operator_anonymized_id,
            "task_type": self.task_type,
            "machine_type": self.machine_type,
            "site_condition": self.site_condition,
            "technique": self.technique_description,
            "observed_metric": self.observed_metric,
            "context_similarity_score": self.context_similarity_score,
            "synthetic_flag": self.synthetic_flag,
        }


def default_peer_path(settings: Optional[dict[str, Any]] = None) -> Path:
    settings = settings or load_settings()
    return resolve_path(settings["data"]["synthetic_dir"]) / "peer_examples.csv"


def load_peer_examples(path: Optional[Path] = None) -> tuple[PeerExample, ...]:
    path = path or default_peer_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate or fetch the synthetic dataset first."
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return tuple(PeerExample.from_row(row) for row in csv.DictReader(handle))


def find_similar(
    examples: Sequence[PeerExample],
    task_type: Optional[str] = None,
    machine_type: Optional[str] = None,
    site_condition: Optional[str] = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    limit: int = DEFAULT_LIMIT,
) -> tuple[PeerExample, ...]:
    """Approved examples matching the context, most similar first.

    An unapproved example is never returned, whatever it matches.
    """
    def matches(example: PeerExample) -> bool:
        if not example.approved_for_peer_learning:
            return False
        if example.context_similarity_score < min_similarity:
            return False
        if task_type and example.task_type.lower() != task_type.lower():
            return False
        if machine_type and example.machine_type.lower() != machine_type.lower():
            return False
        if site_condition and example.site_condition.lower() != site_condition.lower():
            return False
        return True

    selected = sorted(
        (e for e in examples if matches(e)),
        key=lambda e: e.context_similarity_score,
        reverse=True,
    )
    return tuple(selected[:limit]) if limit else tuple(selected)
