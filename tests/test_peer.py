"""Tests for peer learning inside the Training Hub (Feature 06).

Peer learning is part of Training, not a separate feature. The two rules that
make it shareable rather than surveillance are the ones most worth pinning:
unapproved examples are never returned, and the source is always anonymised.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from features.training.peer import PeerExample, find_similar, load_peer_examples

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def examples() -> tuple[PeerExample, ...]:
    return load_peer_examples(FIXTURES / "peer_examples.csv")


def test_examples_load(examples):
    assert len(examples) == 7
    assert examples[0].example_id == "PE0001"


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthetic dataset"):
        load_peer_examples(tmp_path / "nope.csv")


# --- the two rules that matter -----------------------------------------------

def test_an_unapproved_example_is_never_returned(examples):
    """PE0004 has the highest similarity in the set and must still be excluded."""
    found = find_similar(examples, task_type="grading", machine_type="excavator")
    assert "PE0004" not in {e.example_id for e in found}


def test_unapproved_is_excluded_even_with_no_other_filters(examples):
    found = find_similar(examples, min_similarity=0.0, limit=0)
    assert all(e.approved_for_peer_learning for e in found)


def test_sources_are_anonymised_never_real_operator_ids(examples):
    for example in examples:
        assert not example.source_operator_anonymized_id.startswith("OP")


# --- context matching --------------------------------------------------------

def test_only_matching_task_and_machine_are_returned(examples):
    found = find_similar(examples, task_type="grading", machine_type="excavator")
    ids = {e.example_id for e in found}
    assert "PE0006" not in ids   # different task type
    assert "PE0007" not in ids   # different machine type


def test_examples_below_the_similarity_floor_are_dropped(examples):
    found = find_similar(examples, task_type="grading", machine_type="excavator")
    assert "PE0005" not in {e.example_id for e in found}


def test_lowering_the_floor_admits_the_weaker_example(examples):
    found = find_similar(
        examples, task_type="grading", machine_type="excavator", min_similarity=0.2
    )
    assert "PE0005" in {e.example_id for e in found}


def test_results_are_ordered_most_similar_first(examples):
    found = find_similar(examples, task_type="grading", machine_type="excavator")
    scores = [e.context_similarity_score for e in found]
    assert scores == sorted(scores, reverse=True)
    assert found[0].example_id == "PE0001"


def test_site_condition_narrows_further(examples):
    wet = find_similar(examples, task_type="grading", machine_type="excavator",
                       site_condition="wet")
    assert {e.example_id for e in wet} == {"PE0001", "PE0003"}


def test_limit_caps_the_results(examples):
    assert len(find_similar(examples, task_type="grading", machine_type="excavator",
                            limit=1)) == 1


def test_no_match_returns_nothing_rather_than_a_loose_guess(examples):
    assert find_similar(examples, task_type="trenching", machine_type="hauler") == ()


def test_payload_exposes_the_similarity_score(examples):
    """The score is shown with the technique so it is never read as universal advice."""
    payload = find_similar(examples, task_type="grading", machine_type="excavator")[0].to_dict()
    assert payload["context_similarity_score"] == 0.92
    assert payload["source"] == "ANON_A1"
    assert payload["synthetic_flag"] is True
    assert "operator_id" not in payload
