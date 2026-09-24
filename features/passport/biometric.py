"""Face registration and login for the Operator Passport.

    register: frames -> embeddings -> local store
    login:    frame  -> embedding  -> nearest registered operator -> threshold

Recognition is delegated to a pretrained model (DeepFace + ArcFace, D004/D013);
nothing here trains anything. The recognizer is injected, so tests run without a
webcam and without importing TensorFlow -- `DeepFaceRecognizer` imports DeepFace
lazily, inside the call.

Privacy (architecture section 17): embeddings and any captured frames stay on the
local disk under a gitignored directory, nothing is uploaded, and only the three
real team members are ever registered. Raw images are never written into
operators.csv -- the CSV holds a path.

Identity is not authorization. A successful match only says *who* this is;
`authorization.check_authorization` separately decides what they may operate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from shared.config import load_settings, resolve_path

EMBEDDING_FILENAME = "embeddings.json"


class NoFaceDetectedError(ValueError):
    pass


class Recognizer(Protocol):
    """Anything that can turn one image into an embedding vector."""

    model_name: str

    def embed(self, image: Any) -> list[float]: ...


@dataclass
class DeepFaceRecognizer:
    """DeepFace wrapper. DeepFace is imported lazily to keep test runs fast."""

    model_name: str = "ArcFace"
    detector_backend: str = "opencv"
    enforce_detection: bool = True

    def embed(self, image: Any) -> list[float]:
        from deepface import DeepFace

        try:
            representations = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
            )
        except ValueError as exc:  # DeepFace raises ValueError when no face is found
            raise NoFaceDetectedError(str(exc)) from exc
        if not representations:
            raise NoFaceDetectedError("No face detected in the supplied frame")
        return list(representations[0]["embedding"])


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimensions differ ({len(left)} vs {len(right)}); "
            "they were probably produced by different models."
        )
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass(frozen=True)
class RegistrationResult:
    operator_id: str
    embeddings_stored: int
    frames_rejected: int
    model: str
    path: str
    registered_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "embeddings_stored": self.embeddings_stored,
            "frames_rejected": self.frames_rejected,
            "face_model": self.model,
            "face_embedding_path": self.path,
            "face_registration_timestamp": self.registered_at,
            "face_registered": self.embeddings_stored > 0,
        }


@dataclass(frozen=True)
class IdentificationResult:
    authenticated: bool
    operator_id: Optional[str]
    confidence: float
    threshold: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "operator_id": self.operator_id,
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class EmbeddingStore:
    """Per-operator embeddings on local disk, one folder each."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def from_settings(cls, settings: Optional[dict[str, Any]] = None) -> "EmbeddingStore":
        settings = settings or load_settings()
        return cls(resolve_path(settings["biometric"]["registration_image_dir"]))

    def path_for(self, operator_id: str) -> Path:
        return self.root / operator_id / EMBEDDING_FILENAME

    def save(self, operator_id: str, embeddings: Sequence[Sequence[float]], model: str) -> Path:
        path = self.path_for(operator_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "operator_id": operator_id,
            "face_model": model,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "embeddings": [list(e) for e in embeddings],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load(self, operator_id: str) -> Optional[dict[str, Any]]:
        path = self.path_for(operator_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_all(self) -> dict[str, dict[str, Any]]:
        if not self.root.exists():
            return {}
        records = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            record = self.load(folder.name)
            if record and record.get("embeddings"):
                records[folder.name] = record
        return records

    def registered_operator_ids(self) -> list[str]:
        return sorted(self.load_all())


def register_operator(
    operator_id: str,
    frames: Iterable[Any],
    recognizer: Recognizer,
    store: EmbeddingStore,
) -> RegistrationResult:
    """Embed each captured frame; frames with no detectable face are skipped.

    Skipping is not defensive padding -- a webcam capture routinely yields a
    frame where the operator blinked or turned away, and losing one frame should
    not fail the whole registration.
    """
    embeddings: list[list[float]] = []
    rejected = 0
    for frame in frames:
        try:
            embeddings.append(recognizer.embed(frame))
        except NoFaceDetectedError:
            rejected += 1

    if not embeddings:
        raise NoFaceDetectedError(
            f"No usable frames for {operator_id}: all {rejected} captures lacked a detectable face."
        )

    path = store.save(operator_id, embeddings, recognizer.model_name)
    return RegistrationResult(
        operator_id=operator_id,
        embeddings_stored=len(embeddings),
        frames_rejected=rejected,
        model=recognizer.model_name,
        path=str(path),
        registered_at=datetime.now().isoformat(timespec="seconds"),
    )


def confidence_threshold(settings: Optional[dict[str, Any]] = None) -> float:
    settings = settings or load_settings()
    return float(settings["biometric"]["face_confidence_threshold"])


def identify(
    frame: Any,
    recognizer: Recognizer,
    store: EmbeddingStore,
    threshold: Optional[float] = None,
    settings: Optional[dict[str, Any]] = None,
) -> IdentificationResult:
    """Match one frame against every registered operator.

    Confidence is the best cosine similarity against any stored embedding for
    that operator. Below `threshold` nothing is identified and no session may be
    created. The threshold is configurable (`biometric.face_confidence_threshold`)
    and must be calibrated against the real registrations -- the shipped default
    is a starting point, not a validated operating point.
    """
    limit = threshold if threshold is not None else confidence_threshold(settings)
    probe = recognizer.embed(frame)

    scores: dict[str, float] = {}
    for operator_id, record in store.load_all().items():
        scores[operator_id] = max(
            cosine_similarity(probe, stored) for stored in record["embeddings"]
        )

    if not scores:
        return IdentificationResult(
            authenticated=False, operator_id=None, confidence=0.0, threshold=limit, scores={}
        )

    best_id = max(scores, key=lambda key: scores[key])
    best_score = scores[best_id]
    authenticated = best_score >= limit
    return IdentificationResult(
        authenticated=authenticated,
        operator_id=best_id if authenticated else None,
        confidence=best_score,
        threshold=limit,
        scores=scores,
    )
