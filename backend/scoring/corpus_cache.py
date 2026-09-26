"""Per-topic corpus cache with leave-one-out (LOO) calibration.

For each topic we embed every corpus item (per field) once, then score each item
against the other N-1 with the exact pipeline a new submission goes through.
That gives the topic's own reference distributions, from which we derive:

  relevance_floor = see config.RELEVANCE_FLOOR_METHOD:
                    "offtopic":   min(off_mean + K*off_std, on_topic_p5 - margin), where "off" is
                                  the other topics' corpus content scored against THIS topic;
                                  the gate should separate on- from off-topic, not rank on-topic items
                    "percentile": RELEVANCE_FLOOR_PCT percentile of on-topic relevance (pre-fix)
  gate_temp       = GATE_TEMP, or (adaptive) chosen so gate(on_topic_p5) == GATE_TARGET_AT_ON_P5
  dedup_threshold = DEDUP_PCT percentile of within-corpus NN content similarity
  system_score    = percentile rank of a submission's raw score in the LOO raw distribution

config.py values are used only when a distribution is degenerate.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from data_loader import load_fixtures, load_topics
from models import Topic
from scoring import baseline, config, surfacing
from scoring.embed import embed
from scoring.pipeline import (
    RAW_SIGNALS,
    Components,
    CorpusView,
    centroid,
    finalize,
    measure,
    novelty_signal,
    percentile_rank,
)

log = logging.getLogger(__name__)


@dataclass
class TopicCache:
    topic_id: str
    ids: list[str]
    headers: list[str]
    view: CorpusView
    relevance_dist: np.ndarray  # (N,) content-vs-fixed cosine per corpus item (absolute relevance)
    gate_relevance_dist: np.ndarray  # (N,) on-topic gate-signal relevance (LOO)
    off_topic_dist: np.ndarray  # (M,) other topics' content, gate signal vs this topic
    floor_detail: dict
    nn_sim_dist: dict[str, np.ndarray]  # field -> (N,) within-corpus NN cosine
    dedup_threshold: float
    loo: list[Components]  # LOO components per corpus item
    loo_raw: np.ndarray  # (N,) the system_score reference distribution
    signal_dists: dict[str, np.ndarray]  # raw LOO novelty signals, for rank-normalizing live scores
    loo_component_dists: dict[str, np.ndarray] = field(default_factory=dict)
    used_fallback: dict[str, bool] = field(default_factory=dict)
    tfidf: baseline.TfidfIndex | None = None
    baseline_loo: np.ndarray | None = None  # (N,) TF-IDF LOO novelty

    @property
    def relevance_floor(self) -> float:
        return self.view.relevance_floor

    def score_submission(self, sub: dict) -> tuple[Components, float]:
        """THE scoring path for any new submission: (components, system_score percentile).

        Used by the API, experiments and tests, so every caller applies the same toggles."""
        texts = {f: sub[f] for f in config.FIELDS}
        fv = dict(zip(config.FIELDS, embed([texts[f] for f in config.FIELDS])))
        net_new = None
        if config.NET_NEW_BLEND != "off":
            net_new = surfacing.surface(sub["content"], get_sentence_index(self.topic_id), self.ids)["net_new_ratio"]
        comp = finalize(measure(fv, self.view, texts=texts, net_new=net_new), self.view, self.signal_dists)
        return comp, percentile_rank(comp.raw, self.loo_raw)

    def label(self, idx: int) -> str:
        return f'{self.ids[idx]} "{self.headers[idx]}"'


def _degenerate(dist: np.ndarray) -> bool:
    return len(dist) < config.MIN_DIST_SIZE or float(np.std(dist)) < config.MIN_DIST_STD


def _off_topic_contents(topic_id: str) -> list[str]:
    return [s.content for tid, t in load_topics().items() if tid != topic_id for s in t.submissions]


def _floor_and_temp(on: np.ndarray, off: np.ndarray) -> tuple[float, float, dict, bool]:
    """Returns (floor, temp, detail, used_fallback)."""
    detail: dict = {"method": config.RELEVANCE_FLOOR_METHOD}
    fallback = False
    if not config.CALIBRATE_THRESHOLDS or _degenerate(on):
        floor, fallback = config.RELEVANCE_FLOOR_FALLBACK, True
    elif config.RELEVANCE_FLOOR_METHOD == "offtopic":
        cap = float(np.percentile(on, config.FLOOR_ON_TOPIC_PCT)) - config.FLOOR_MARGIN
        if _degenerate(off):
            floor, detail["binding"] = cap, "on-topic cap (off-topic reference degenerate)"
        else:
            cand = float(off.mean() + config.FLOOR_OFFTOPIC_K * off.std())
            floor = min(cand, cap)
            detail.update(offtopic_candidate=cand, on_topic_cap=cap,
                          binding="off-topic reference" if cand <= cap else "on-topic cap")
    else:
        floor = float(np.percentile(on, config.RELEVANCE_FLOOR_PCT))
    detail["floor"] = floor

    temp = config.GATE_TEMP
    if config.GATE_TEMP_MODE == "adaptive" and not _degenerate(on):
        gap = float(np.percentile(on, config.FLOOR_ON_TOPIC_PCT)) - floor
        target = config.GATE_TARGET_AT_ON_P5
        if gap > 0:
            temp = gap / math.log(target / (1 - target))
    detail["gate_temp"] = temp
    return floor, temp, detail, fallback


def build_topic_cache(topic: Topic) -> TopicCache:
    subs = topic.submissions
    n = len(subs)
    texts = [getattr(s, f) for f in config.FIELDS for s in subs] + [topic.fixed_content]
    all_vecs = embed(texts)
    vecs = {f: all_vecs[i * n : (i + 1) * n] for i, f in enumerate(config.FIELDS)}
    fixed_vec = all_vecs[-1]

    # Distributions that don't depend on thresholds.
    relevance_dist = vecs["content"] @ fixed_vec
    off_vecs = embed(_off_topic_contents(topic.topic_id)) if len(load_topics()) > 1 else np.zeros((0, fixed_vec.shape[0]))
    if config.RELEVANCE_GATE_SIGNAL == "centroid":
        all_ref = np.ones(n, dtype=bool)
        gate_rel = np.array([centroid(vecs["content"], np.arange(n) != i) @ vecs["content"][i] for i in range(n)])
        off = off_vecs @ centroid(vecs["content"], all_ref) if len(off_vecs) else np.zeros(0)
    else:
        gate_rel = relevance_dist
        off = off_vecs @ fixed_vec if len(off_vecs) else np.zeros(0)

    nn_sim_dist = {}
    for f in config.FIELDS:
        s = vecs[f] @ vecs[f].T
        np.fill_diagonal(s, -np.inf)
        nn_sim_dist[f] = s.max(axis=1) if n > 1 else np.zeros(n)

    used_fallback = {}
    floor, temp, floor_detail, used_fallback["relevance_floor"] = _floor_and_temp(gate_rel, off)
    if not config.CALIBRATE_THRESHOLDS or _degenerate(nn_sim_dist["content"]):
        dedup, used_fallback["dedup_threshold"] = config.DEDUP_THRESHOLD_FALLBACK, True
    else:
        dedup, used_fallback["dedup_threshold"] = float(np.percentile(nn_sim_dist["content"], config.DEDUP_PCT)), False

    doc_tokens = [set(baseline.tokens(baseline.submission_text(s.model_dump()))) for s in subs]
    token_df: dict[str, float] = {}
    for toks in doc_tokens:
        for t in toks:
            token_df[t] = token_df.get(t, 0.0) + 1.0 / n
    view = CorpusView(vecs=vecs, fixed_vec=fixed_vec, relevance_floor=floor, gate_temp=temp,
                      gate_relevance_dist=gate_rel, token_df=token_df)

    # Leave-one-out: same pipeline, item i measured against the other N-1 ...
    ids = [s.id for s in subs]
    if config.NET_NEW_BLEND != "off":
        si = get_sentence_index(topic.topic_id)
        net_new = [surfacing.surface(s.content, si, ids, exclude_owner=i)["net_new_ratio"] for i, s in enumerate(subs)]
    else:
        net_new = [None] * n
    loo = [measure({f: vecs[f][i] for f in config.FIELDS}, view, exclude=frozenset({i}),
                   texts={f: getattr(subs[i], f) for f in config.FIELDS}, net_new=net_new[i]) for i in range(n)]
    # ... then blended/combined exactly as a live submission would be.
    signal_dists = {k: np.array([get(c) for c in loo]) for k, get in RAW_SIGNALS.items()}
    signal_dists["knn"] = np.array([novelty_signal(c, signal_dists) for c in loo])
    for c in loo:
        finalize(c, view, signal_dists)
    loo_raw = np.array([c.raw for c in loo])

    texts = [baseline.submission_text(s.model_dump()) for s in subs]
    tfidf = baseline.build_index(texts)

    return TopicCache(
        topic_id=topic.topic_id,
        ids=[s.id for s in subs],
        headers=[s.header for s in subs],
        view=view,
        relevance_dist=relevance_dist,
        gate_relevance_dist=gate_rel,
        off_topic_dist=off,
        floor_detail=floor_detail,
        nn_sim_dist=nn_sim_dist,
        dedup_threshold=dedup,
        loo=loo,
        loo_raw=loo_raw,
        signal_dists=signal_dists,
        loo_component_dists={
            "relevance": relevance_dist,
            "novelty": np.array([c.blended_novelty for c in loo]),
            "recombination_novelty": np.array([c.recombination.score for c in loo]),
        },
        used_fallback=used_fallback,
        tfidf=tfidf,
        baseline_loo=baseline.loo_novelty(tfidf, texts),
    )


_caches: dict[str, TopicCache] = {}


def get_cache(topic_id: str) -> TopicCache:
    if topic_id not in _caches:
        topic = load_topics()[topic_id]
        _caches[topic_id] = build_topic_cache(topic)
        _validate(_caches[topic_id])
    return _caches[topic_id]


_sentence_indexes: dict[str, surfacing.SentenceIndex] = {}


def get_sentence_index(topic_id: str) -> surfacing.SentenceIndex:
    """Corpus sentence index for redundancy-aware surfacing (built once per topic)."""
    if topic_id not in _sentence_indexes:
        topic = load_topics()[topic_id]
        _sentence_indexes[topic_id] = surfacing.build_index([s.content for s in topic.submissions])
    return _sentence_indexes[topic_id]


def build_all() -> None:
    for tid in load_topics():
        si = get_sentence_index(tid)
        log.info("surfacing %-18s %d corpus sentences, novel<=%.3f redundant>=%.3f (%s)",
                 tid, len(si.texts), si.novel_max, si.redundant_min, config.SURF_EMBED_MODEL)
        c = get_cache(tid)
        g = np.array([L.gate for L in c.loo])
        log.info(
            "cache %-18s N=%d mode=%s signal=%s floor=%.3f (%s%s) temp=%.4f on-topic gate mean=%.3f min=%.3f "
            "off-topic mean=%.3f dedup=%.3f%s",
            tid, len(c.ids), config.COMBINE_MODE, config.RELEVANCE_GATE_SIGNAL, c.relevance_floor,
            c.floor_detail.get("method"), f", bound by {c.floor_detail['binding']}" if "binding" in c.floor_detail else "",
            c.view.gate_temp, g.mean(), g.min(), float(c.off_topic_dist.mean()) if len(c.off_topic_dist) else float("nan"),
            c.dedup_threshold, " (fallback)" if c.used_fallback["dedup_threshold"] else "",
        )


def _validate(cache: TopicCache) -> None:
    """Log where planted near-duplicate / generic ids land in the LOO distribution."""
    fx = load_fixtures().get(cache.topic_id)
    if not fx:
        log.info("validate %-18s no fixtures", cache.topic_id)
        return
    idx = {sid: i for i, sid in enumerate(cache.ids)}
    groups = {
        "near_duplicate": [d["id"] for d in fx.get("near_duplicate_ids", [])],
        "generic": list(fx.get("generic_ids", [])),
    }
    for name, sids in groups.items():
        ranks = {sid: percentile_rank(cache.loo_raw[idx[sid]], cache.loo_raw) for sid in sids if sid in idx}
        missing = [sid for sid in sids if sid not in idx]
        bad = {sid: r for sid, r in ranks.items() if r >= config.VALIDATION_QUANTILE}
        status = "PASS" if not bad and not missing else "WARN"
        detail = " ".join(f"{sid}={r:.2f}" for sid, r in ranks.items())
        extra = f" above-q{config.VALIDATION_QUANTILE:.2f}={sorted(bad)}" if bad else ""
        extra += f" missing={missing}" if missing else ""
        log.info("validate %-18s %-14s %s  pct: %s%s", cache.topic_id, name, status, detail, extra)

    # Ceiling note: each planted near-dup's *original* is equally duplicated in LOO, so ~15
    # items compete for 12.5 bottom-quartile slots. AUC (planted vs ordinary items, originals
    # excluded) is the threshold-free view: 1.0 = every planted item below every ordinary one.
    planted = [idx[s] for s in groups["near_duplicate"] + groups["generic"] if s in idx]
    originals = {idx[d["duplicates"]] for d in fx.get("near_duplicate_ids", []) if d.get("duplicates") in idx}
    ordinary = [i for i in range(len(cache.ids)) if i not in planted and i not in originals]
    if planted and ordinary:
        r = cache.loo_raw
        auc = float(np.mean([(r[p] < r[o]) + 0.5 * (r[p] == r[o]) for p in planted for o in ordinary]))
        log.info("validate %-18s %-14s AUC=%.3f (planted vs %d ordinary items)", cache.topic_id, "separation", auc, len(ordinary))
