"""Task 4: a HIGH-novelty / LOW-relevance submission must NOT be rewarded.

The held-out T2 case of every topic is off-topic text that no corpus item resembles (so its
novelty is high). The asymmetric relevance gate must still drive its score to ~0.
"""

import pytest
from fastapi.testclient import TestClient

import main
from conftest import TOPIC_IDS
from scoring import judge as judge_mod
from scoring import novelty
from scoring.pipeline import percentile_rank


def t2(fixtures, tid):
    return next(c["submission"] for c in fixtures[tid]["held_out_test_cases"] if c["case"].startswith("T2"))


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_high_novelty_low_relevance_not_rewarded(caches, fixtures, tid):
    cache = caches[tid]
    comp, system = cache.score_submission(t2(fixtures, tid))
    novelty_pct = percentile_rank(comp.blended_novelty, cache.loo_component_dists["novelty"])
    # precondition: it really is high-novelty relative to this topic's corpus ...
    assert novelty_pct >= 0.75, f"{tid}: T2 novelty percentile {novelty_pct:.2f} is not high"
    # ... and it is stopped by relevance, so it is not rewarded
    assert comp.gate < 0.1, f"{tid}: gate {comp.gate:.3f}"
    assert system < 0.05, f"{tid}: system_score {system:.3f} for a high-novelty, low-relevance submission"


@pytest.fixture(scope="module")
def client():
    def no_judge(*_a, **_k):
        raise judge_mod.JudgeUnavailable("disabled in tests", reason="rate-limited")

    mp = pytest.MonkeyPatch()
    mp.setattr(novelty, "judge", no_judge)
    with TestClient(main.app) as c:
        yield c
    mp.undo()


@pytest.mark.parametrize("tid", TOPIC_IDS)
def test_high_novelty_low_relevance_not_rewarded_via_api(client, fixtures, tid):
    j = client.post("/api/score", json={"topic_id": tid, **t2(fixtures, tid)}).json()
    assert j["system_score"] < 0.05
    assert j["flags"]["low_relevance"] is True
