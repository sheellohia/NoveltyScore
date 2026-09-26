"""API contract tests. The Gemini judge is replaced with an unavailable stub (no network)."""

import pytest
from fastapi.testclient import TestClient

import main
from scoring import judge as judge_mod
from scoring import novelty

TOP_LEVEL = {"system_score", "baseline_score", "llm_score", "breakdown", "flags", "rationale", "surfacing", "llm_source"}
BREAKDOWN = {"relevance", "novelty", "recombination_novelty"}


@pytest.fixture(scope="module")
def client():
    def no_judge(*_a, **_k):
        raise judge_mod.JudgeUnavailable(
            "gemini-3.7-flash: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'quota'}}", reason="rate-limited")

    mp = pytest.MonkeyPatch()
    mp.setattr(novelty, "judge", no_judge)
    with TestClient(main.app) as c:
        yield c
    mp.undo()


def body(**over):
    b = {"topic_id": "remote_work", "header": "Remote work reshapes city budgets",
         "content": "When office workers move away, city income-tax revenue leaves with them.",
         "takeaway": "Remote work is a fiscal story."}
    b.update(over)
    return b


def test_topics_list_and_detail_shape(client):
    topics = client.get("/api/topics").json()
    assert len(topics) == 5
    detail = client.get(f"/api/topics/{topics[0]['topic_id']}").json()
    assert set(detail) == {"topic_id", "title", "fixed_content"}


def test_score_contract(client):
    r = client.post("/api/score", json=body())
    assert r.status_code == 200
    j = r.json()
    assert set(j) == TOP_LEVEL
    assert set(j["breakdown"]) == BREAKDOWN
    assert all(0.0 <= j[k] <= 1.0 for k in ("system_score", "baseline_score", "llm_score"))


def test_surfacing_block(client):
    j = client.post("/api/score", json=body()).json()["surfacing"]
    assert 0.0 <= j["net_new_ratio"] <= 1.0
    assert set(j["thresholds"]) == {"novel_max_redundancy", "redundant_min_redundancy"}
    assert j["sentences"] and all(x["label"] in ("novel", "partial", "redundant") for x in j["sentences"])
    assert all(set(x) >= {"text", "label", "redundancy", "closest_id"} for x in j["sentences"])


def test_breakdown_relevance_is_absolute_cosine(client):
    from scoring.corpus_cache import get_cache
    from scoring.embed import embed

    b = body()
    j = client.post("/api/score", json=b).json()
    vec = embed([b["content"]])[0]
    cos = float(get_cache("remote_work").view.fixed_vec @ vec)
    assert j["breakdown"]["relevance"] == pytest.approx(min(max(cos, 0.0), 1.0), abs=1e-3)


def test_content_over_100_words_rejected(client):
    r = client.post("/api/score", json=body(content=" ".join(["w"] * 101)))
    assert r.status_code == 400 and "101 words" in r.json()["detail"]


def test_takeaway_over_120_chars_rejected(client):
    r = client.post("/api/score", json=body(takeaway="x" * 121))
    assert r.status_code == 400 and "120" in r.json()["detail"]


def test_blank_header_rejected(client):
    assert client.post("/api/score", json=body(header="  ")).status_code == 400


def test_unknown_topic_404(client):
    assert client.post("/api/score", json=body(topic_id="nope")).status_code == 404


def test_judge_failure_is_clean_and_labelled(client):
    j = client.post("/api/score", json=body()).json()
    assert j["llm_source"] == "unavailable"
    assert j["llm_score"] == 0.0
    assert "LLM judge: unavailable (rate-limited)." in j["rationale"]
    assert "RESOURCE_EXHAUSTED" not in j["rationale"] and "{" not in j["rationale"]


def test_corpus_item_falls_back_to_cached_reference(client, topics):
    item = topics["remote_work"].submissions[0]
    j = client.post("/api/score", json=body(header=item.header, content=item.content, takeaway=item.takeaway)).json()
    assert j["llm_source"] == "cached"
    assert f"corpus reference for {item.id}" in j["rationale"]
