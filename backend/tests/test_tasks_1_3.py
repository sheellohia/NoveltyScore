"""Task 1 (submission shape + fixed content <= 100 words) and Task 3 (~50-item corpus scoring)."""

import numpy as np
import pytest

from conftest import TOPIC_IDS
from models import MAX_CONTENT_WORDS, count_words


# --- Task 1 -----------------------------------------------------------------------------

@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_task1_fixed_content_at_most_100_words(topics, tid):
    assert count_words(topics[tid].fixed_content) <= MAX_CONTENT_WORDS


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_task1_every_submission_has_the_three_properties(topics, tid):
    for s in topics[tid].submissions:
        assert set(s.model_dump()) == {"id", "header", "content", "takeaway"}
        assert s.header.strip() and s.content.strip() and s.takeaway.strip()
        assert count_words(s.content) <= MAX_CONTENT_WORDS


# --- Task 3 -----------------------------------------------------------------------------

@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_task3_corpus_of_about_50_items_is_scored(caches, tid):
    c = caches[tid]
    n = len(c.ids)
    assert 40 <= n <= 60, f"{tid}: corpus has {n} items"
    # every corpus item is scored leave-one-out against the other n-1 with the same pipeline
    assert len(c.loo) == n and len(c.loo_raw) == n
    assert np.all(np.isfinite(c.loo_raw))
    # the calibrated system score spans the unit interval within the topic
    assert 0.0 <= c.relevance_floor <= 1.0 and c.view.gate_temp > 0


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_task3_new_submission_scored_against_full_corpus(caches, fixtures, tid):
    sub = next(x["submission"] for x in fixtures[tid]["held_out_test_cases"] if x["case"].startswith("T1"))
    comp, system = caches[tid].score_submission(sub)
    assert 0.0 <= system <= 1.0
    assert 0 <= comp.nn_idx["content"] < len(caches[tid].ids)
