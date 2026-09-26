"""Novelty scoring entry point.

The API layer only ever calls `score(submission, topic)` and relies on the
ScoreResponse shape.

Status:
  system_score, breakdown, flags  REAL  embeddings + per-topic LOO calibration (corpus_cache.py)
  llm_score                       REAL  Gemini judge (judge.py). llm_source says where it came from:
                                        "live" (this call), "cached" (earlier verdict for this exact input,
                                        or the corpus reference if the text is a corpus item) or
                                        "unavailable" (llm_score is 0.0 and must not be displayed as a score)
  baseline_score                  REAL  TF-IDF nearest-neighbor novelty (baseline.py), uncalibrated
  flags.gaming                    REAL  grader-directed text (flags.py) OR the LLM judge's gaming verdict
                                        when a live/cached verdict exists
"""

import logging

from models import Breakdown, Flags, ScoreResponse, Surfacing
from scoring import baseline, config, surfacing
from scoring.corpus_cache import TopicCache, get_cache, get_sentence_index
from scoring.flags import grader_directed
from scoring.judge import JudgeUnavailable, corpus_reference_score, judge
from scoring.pipeline import Components, percentile_rank

log = logging.getLogger(__name__)

FIELD_LABEL = {"header": "header", "content": "content", "takeaway": "takeaway"}


def llm_judgement(submission: dict, topic: dict) -> tuple[float, str, str, bool | None]:
    """(llm_score, llm_source, rationale line, judge's gaming verdict or None). Never returns a
    score for a different input, and never puts raw API error text in the rationale."""
    try:
        r = judge(submission, topic)
        source = "cached" if r.source == "cache" else "live"
        log.info("llm judge path=%s model=%s score=%.3f", source, r.model, r.score)
        label = f"LLM judge ({r.model}, cached)" if source == "cached" else f"LLM judge ({r.model})"
        return r.score, source, f"{label}: {r.verdict.rationale}", r.verdict.gaming
    except JudgeUnavailable as e:
        ref = corpus_reference_score(submission, topic)
        if ref is not None:
            score, sid = ref
            log.info("llm judge path=cached-corpus-reference item=%s score=%.3f (live call failed: %s)", sid, score, e.reason)
            return score, "cached", f"LLM judge: cached score (corpus reference for {sid}).", None
        log.warning("llm judge path=unavailable reason=%s detail=%s", e.reason, str(e)[:300])
        return 0.0, "unavailable", f"LLM judge: unavailable ({e.reason}).", None


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _item(cache: TopicCache, idx: int) -> str:
    return f'{cache.ids[idx]} "{_short(cache.headers[idx])}"'


def recombination_explain(comp: Components, cache: TopicCache, recomb_pct: float, novelty_pct: float) -> str:
    """The demo line: which fields bridge which corpus regions."""
    r = comp.recombination
    a, b = r.lowest_pair
    na, nb = r.neighbors[a][0], r.neighbors[b][0]
    head = f"Recombination ({recomb_pct:.0%} pct): "
    if comp.gate < 0.5:
        return head + "not credited. The submission is below this topic's relevance floor, so fields landing in different corpus regions reflects being off-topic, not a bridge."
    if recomb_pct < 0.5 or na == nb:
        return head + (
            "conventional framing. Header, content and takeaway point at the same cluster around "
            f"{_item(cache, r.neighbors['content'][0])} (lowest overlap {r.lowest_overlap:.2f}, "
            f"{FIELD_LABEL[a]}/{FIELD_LABEL[b]})."
        )
    bridge = (
        f"your {FIELD_LABEL[a]} sits near {_item(cache, na)} while your {FIELD_LABEL[b]} sits near "
        f"{_item(cache, nb)} (neighbor overlap {r.lowest_overlap:.2f})"
    )
    if novelty_pct < 0.5:
        # Fields diverge, but the substance is already in the corpus: likely a reworded header/takeaway.
        return head + f"mostly surface. {bridge[0].upper() + bridge[1:]}, but the content itself closely matches {_item(cache, comp.nn_idx['content'])}."
    return head + f"{bridge}, bridging corpus regions that rarely co-occur."


def score(submission: dict, topic: dict) -> ScoreResponse:
    """Score a submission's novelty against a topic's corpus.

    submission: {topic_id, header, content, takeaway}
    topic: full topic dict (topic_id, title, fixed_content, submissions)
    """
    cache = get_cache(topic["topic_id"])
    comp, system = cache.score_submission(submission)
    d = cache.loo_component_dists
    pct = {
        "relevance": percentile_rank(comp.relevance, d["relevance"]),
        "novelty": percentile_rank(comp.blended_novelty, d["novelty"]),
        "recombination_novelty": percentile_rank(comp.recombination.score, d["recombination_novelty"]),
    }
    # Relevance is shown as an ABSOLUTE cosine (0.62 -> 62%); novelty axes stay percentiles.
    breakdown = {
        "relevance": min(max(comp.relevance, 0.0), 1.0),
        "novelty": pct["novelty"],
        "recombination_novelty": pct["recombination_novelty"],
    }

    duplicate = comp.nn_sim["content"] >= cache.dedup_threshold
    low_relevance = comp.gate_relevance < cache.relevance_floor

    nn = comp.nn_idx["content"]
    mode = "" if config.COMBINE_MODE == "gate" else f" [{config.COMBINE_MODE} mode]"
    system_line = (
        f"System{mode}: more novel than {system:.0%} of this topic's corpus. "
        f"Closest match {_item(cache, nn)} (similarity {comp.nn_sim['content']:.2f}, "
        f"duplicate threshold {cache.dedup_threshold:.2f}). Relevance {comp.relevance:.2f} to the fixed content "
        f"({pct['relevance']:.0%} percentile within this topic); relevance gate {comp.gate:.2f}."
    )
    recomb_line = recombination_explain(comp, cache, pct["recombination_novelty"], pct["novelty"])

    baseline_score = round(baseline.novelty(cache.tfidf, baseline.submission_text(submission)), 3)
    llm_score, llm_source, judge_line, judge_gaming = llm_judgement(submission, topic)
    gaming = grader_directed(*(submission[f] for f in config.FIELDS)) or bool(judge_gaming)

    si = get_sentence_index(topic["topic_id"])
    surf = surfacing.surface(submission["content"], si, cache.ids)
    labels = [x["label"] for x in surf["sentences"]]
    surf_line = (
        f"New information: {surf['net_new_ratio']:.0%} of the content is net-new "
        f"({labels.count('novel')} novel, {labels.count('partial')} partial, {labels.count('redundant')} already said)."
    )

    log.info(
        "score %s raw=%.3f system=%.2f rel=%.3f gate_rel=%.3f floor=%.3f gate=%.3f nn=%s:%.3f knn=%.3f recomb=%.3f",
        topic["topic_id"], comp.raw, system, comp.relevance, comp.gate_relevance, cache.relevance_floor, comp.gate,
        cache.ids[nn], comp.nn_sim["content"], comp.knn_novelty, comp.recombination.score,
    )

    return ScoreResponse(
        system_score=round(system, 3),
        baseline_score=baseline_score,
        llm_score=llm_score,
        llm_source=llm_source,
        breakdown=Breakdown(**{k: round(v, 3) for k, v in breakdown.items()}),
        flags=Flags(duplicate=duplicate, low_relevance=low_relevance, gaming=gaming),
        rationale=f"{system_line} {recomb_line} {surf_line} {judge_line}",
        surfacing=Surfacing(
            net_new_ratio=surf["net_new_ratio"],
            thresholds={"novel_max_redundancy": round(si.novel_max, 4),
                        "redundant_min_redundancy": round(si.redundant_min, 4)},
            sentences=surf["sentences"],
        ),
    )
