"""Scoring guardrails. No LLM calls: embeddings are local and the judge is never invoked here.

Run (from backend/):  .venv/bin/python -m pytest
"""

import numpy as np
import pytest

from conftest import TOPIC_IDS
from scoring import config
from scoring.pipeline import percentile_rank


def score(cache, sub):
    return cache.score_submission(sub)


def case(fixtures, tid, prefix):
    return next(c["submission"] for c in fixtures[tid]["held_out_test_cases"] if c["case"].startswith(prefix))


# --- adversarial: off-topic must stay rejected ------------------------------------

@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_offtopic_T2_scores_below_0_2(caches, fixtures, tid):
    comp, system = score(caches[tid], case(fixtures, tid, "T2"))
    assert system < 0.2, f"{tid}: T2 system={system:.3f} gate={comp.gate:.3f}"


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_offtopic_T2_is_stopped_by_the_gate(caches, fixtures, tid):
    comp, _ = score(caches[tid], case(fixtures, tid, "T2"))
    assert comp.gate < 0.1, f"{tid}: T2 gate={comp.gate:.3f}"


# --- on-topic content passes the gate ------------------------------------------------

@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_on_topic_corpus_passes_gate(caches, tid):
    gates = np.array([L.gate for L in caches[tid].loo])
    assert (gates >= 0.9).mean() >= 0.9, f"{tid}: only {(gates >= 0.9).mean():.0%} of on-topic items have gate >= 0.9"


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_floor_sits_below_on_topic_p5(caches, tid):
    c = caches[tid]
    assert c.relevance_floor < np.percentile(c.gate_relevance_dist, config.FLOOR_ON_TOPIC_PCT)


def test_ui_four_day_week_submission_passes_gate(caches):
    sub = {
        "header": "Four-day weeks will reshape childcare markets",
        "content": ("If millions of parents are home every Friday, daycare centres lose a fifth of weekday demand. "
                    "Many run on thin margins and fixed staffing, so providers may shift to four-day contracts, raise "
                    "prices, or close. Parents on five-day shifts, often lower-paid service workers, could find care "
                    "harder to get. The policy aimed at wellbeing may quietly widen a class gap in childcare access."),
        "takeaway": "A shorter week for some could mean worse childcare for others.",
    }
    comp, _ = score(caches["four_day_week"], sub)
    assert comp.gate >= 0.95, f"gate={comp.gate:.3f}"


# --- redundancy is penalized once (novelty), not twice (novelty + gate) --------------

def test_generics_pass_gate_but_rank_low_on_novelty(caches, fixtures):
    gates, novelty = [], []
    for tid, c in caches.items():
        nov = c.loo_component_dists["novelty"]
        for sid in fixtures[tid]["generic_ids"]:
            i = c.ids.index(sid)
            gates.append(c.loo[i].gate)
            novelty.append(percentile_rank(nov[i], nov))
    assert np.mean(gates) >= 0.9, f"generics mean gate {np.mean(gates):.3f}: redundancy is being gated"
    assert max(novelty) < 0.25, f"a planted generic has novelty percentile {max(novelty):.2f}"


# --- ordering sanity -------------------------------------------------------------------

@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_novel_T1_beats_duplicate_T3(caches, fixtures, tid):
    _, t1 = score(caches[tid], case(fixtures, tid, "T1"))
    _, t3 = score(caches[tid], case(fixtures, tid, "T3"))
    assert t1 > t3, f"{tid}: T1={t1:.2f} <= T3={t3:.2f}"
