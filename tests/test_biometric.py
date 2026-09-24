"""Tests for face registration and login (Feature 01).

No test may need a webcam or import TensorFlow: the recognizer is injected, so a
deterministic fake stands in for DeepFace. `FakeRecognizer` maps a label to a
fixed vector, which lets each test state the similarity it is exercising.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from features.passport.biometric import (
    EmbeddingStore,
    IdentificationResult,
    NoFaceDetectedError,
    confidence_threshold,
    cosine_similarity,
    identify,
    register_operator,
)

# Deliberately near-orthogonal so an unrelated probe scores far below threshold.
FACE_A = [1.0, 0.0, 0.0, 0.0]
FACE_A_VARIANT = [0.95, 0.10, 0.05, 0.0]   # same person, different frame
FACE_B = [0.0, 1.0, 0.0, 0.0]
STRANGER = [0.0, 0.0, 1.0, 0.0]

SETTINGS = {
    "biometric": {"face_confidence_threshold": 0.70, "registration_image_dir": "data/face_registrations"}
}


@dataclass
class FakeRecognizer:
    """Returns a preset vector per frame label; unknown labels have no face."""

    model_name: str = "ArcFace"
    vectors: dict = field(default_factory=dict)

    def embed(self, image):
        if image not in self.vectors:
            raise NoFaceDetectedError(f"no face in {image!r}")
        return list(self.vectors[image])


@pytest.fixture
def recognizer():
    return FakeRecognizer(vectors={
        "a1": FACE_A, "a2": FACE_A_VARIANT, "b1": FACE_B, "stranger": STRANGER,
    })


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(tmp_path / "face_registrations")


# --- similarity --------------------------------------------------------------

def test_cosine_similarity_endpoints():
    assert cosine_similarity(FACE_A, FACE_A) == pytest.approx(1.0)
    assert cosine_similarity(FACE_A, FACE_B) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_zero_vector_scores_zero_rather_than_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mismatched_dimensions_raise_a_named_cause():
    with pytest.raises(ValueError, match="different models"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# --- registration ------------------------------------------------------------

def test_registration_stores_one_embedding_per_usable_frame(recognizer, store):
    result = register_operator("OP1003", ["a1", "a2"], recognizer, store)
    assert result.embeddings_stored == 2
    assert result.frames_rejected == 0
    assert result.to_dict()["face_registered"] is True
    assert result.to_dict()["face_model"] == "ArcFace"


def test_frames_without_a_detectable_face_are_skipped_not_fatal(recognizer, store):
    result = register_operator("OP1003", ["a1", "blink", "a2"], recognizer, store)
    assert result.embeddings_stored == 2
    assert result.frames_rejected == 1


def test_registration_fails_when_no_frame_has_a_face(recognizer, store):
    with pytest.raises(NoFaceDetectedError, match="No usable frames"):
        register_operator("OP1003", ["blink", "turned_away"], recognizer, store)


def test_no_raw_image_is_written_only_embeddings(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    record = store.load("OP1003")
    assert set(record) == {"operator_id", "face_model", "registered_at", "embeddings"}
    assert record["embeddings"] == [FACE_A]
    written = [p.name for p in (store.root / "OP1003").iterdir()]
    assert written == ["embeddings.json"]


def test_store_round_trip_and_listing(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    register_operator("OP1002", ["b1"], recognizer, store)
    assert store.registered_operator_ids() == ["OP1002", "OP1003"]
    assert store.load("OP9999") is None


def test_empty_store_lists_nothing(store):
    assert store.load_all() == {}
    assert store.registered_operator_ids() == []


# --- identification ----------------------------------------------------------

def test_a_different_frame_of_the_same_person_authenticates(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    result = identify("a2", recognizer, store, settings=SETTINGS)
    assert result.authenticated is True
    assert result.operator_id == "OP1003"
    assert result.confidence > 0.70


def test_the_right_operator_is_chosen_among_several(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    register_operator("OP1002", ["b1"], recognizer, store)
    result = identify("a2", recognizer, store, settings=SETTINGS)
    assert result.operator_id == "OP1003"
    assert result.scores["OP1003"] > result.scores["OP1002"]


def test_an_unregistered_face_is_refused(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    result = identify("stranger", recognizer, store, settings=SETTINGS)
    assert result.authenticated is False
    assert result.operator_id is None


def test_a_refused_match_yields_no_operator_id_to_open_a_session_with(recognizer, store):
    """Architecture: verification failure must not produce an authorized session."""
    register_operator("OP1003", ["a1"], recognizer, store)
    result = identify("stranger", recognizer, store, settings=SETTINGS)
    assert result.operator_id is None
    assert result.to_dict()["operator_id"] is None


def test_identification_against_an_empty_store_is_refused(recognizer, store):
    result = identify("a1", recognizer, store, settings=SETTINGS)
    assert result.authenticated is False
    assert result.operator_id is None
    assert result.confidence == 0.0


def test_threshold_is_configurable_and_changes_the_outcome(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    borderline = FakeRecognizer(vectors={"probe": [0.8, 0.6, 0.0, 0.0]})  # cos = 0.8
    assert identify("probe", borderline, store, threshold=0.70).authenticated is True
    assert identify("probe", borderline, store, threshold=0.90).authenticated is False


def test_threshold_is_read_from_settings():
    assert confidence_threshold(SETTINGS) == 0.70
    assert confidence_threshold() == pytest.approx(0.70)


def test_identification_payload_matches_the_api_contract(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    payload = identify("a2", recognizer, store, settings=SETTINGS).to_dict()
    assert payload["authenticated"] is True
    assert payload["operator_id"] == "OP1003"
    assert isinstance(payload["confidence"], float)


# --- login hand-off to the dashboard -----------------------------------------

def test_no_login_recorded_means_no_active_session(store):
    assert store.active_login() is None


def test_a_successful_login_is_recorded(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    result = identify("a2", recognizer, store, settings=SETTINGS)
    store.record_login(result)

    session = store.active_login()
    assert session is not None
    assert session.operator_id == "OP1003"
    # Stored rounded to 4dp, which is plenty for a display value.
    assert session.confidence == pytest.approx(result.confidence, abs=1e-4)
    assert session.threshold == 0.70
    assert session.age_minutes() < 1


def test_a_refused_login_clears_any_previous_session(recognizer, store):
    """A failed scan must not leave the last person signed in."""
    register_operator("OP1003", ["a1"], recognizer, store)
    store.record_login(identify("a2", recognizer, store, settings=SETTINGS))
    assert store.active_login() is not None

    store.record_login(identify("stranger", recognizer, store, settings=SETTINGS))
    assert store.active_login() is None


def test_a_corrupt_session_file_does_not_raise(store):
    """The dashboard reads this on every rerun; bad JSON must not take it down."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.session_path().write_text("{not valid json", encoding="utf-8")
    assert store.active_login() is None


def test_clear_login_removes_the_session(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    store.record_login(identify("a2", recognizer, store, settings=SETTINGS))
    store.clear_login()
    assert store.active_login() is None


def test_the_session_file_holds_no_biometric_data(recognizer, store):
    """It is a hand-off between processes, not a credential -- no embedding."""
    import json

    register_operator("OP1003", ["a1"], recognizer, store)
    store.record_login(identify("a2", recognizer, store, settings=SETTINGS))
    payload = json.loads(store.session_path().read_text(encoding="utf-8"))
    assert set(payload) == {"operator_id", "confidence", "threshold", "authenticated_at"}


def test_a_frame_with_no_face_propagates_rather_than_silently_failing(recognizer, store):
    register_operator("OP1003", ["a1"], recognizer, store)
    with pytest.raises(NoFaceDetectedError):
        identify("blink", recognizer, store, settings=SETTINGS)
