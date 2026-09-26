"""Core novelty math, shared by live scoring and leave-one-out calibration.

Two steps, both used identically for a new submission and for LOO corpus items:

  measure()   raw, uncalibrated signals (similarities, kNN novelty, recombination).
              A new submission is measured against all N corpus items; corpus item i
              is measured against the other N-1 via exclude={i}.
  finalize()  blend + combine with relevance. When NORMALIZE_COMPONENTS is on, each
              novelty signal is first mapped to its percentile rank within the topic's
              LOO distribution for that signal, so blend weights act on one scale.
"""

import math
from dataclasses import dataclass

import numpy as np

from scoring import config
from scoring.recombination import RecombinationResult, recombination_novelty


@dataclass
class CorpusView:
    """What the pipeline needs to know about a topic's corpus."""

    vecs: dict[str, np.ndarray]  # field -> (N, d) unit vectors
    fixed_vec: np.ndarray  # (d,) fixed_content embedding
    relevance_floor: float  # on the gate-signal scale (see config.RELEVANCE_GATE_SIGNAL)
    gate_temp: float = 0.03
    gate_relevance_dist: np.ndarray | None = None  # corpus gate-signal relevance (harmonic mode percentile)
    token_df: dict[str, float] | None = None  # token -> fraction of corpus submissions containing it


@dataclass
class Components:
    relevance: float  # cos(content, fixed_content): the absolute relevance shown to users
    gate_relevance: float  # the signal the gate thresholds (fixed or corpus-centroid cosine)
    nn_sim: dict[str, float]  # field -> max cosine to any reference item
    nn_idx: dict[str, int]  # field -> index of that nearest item
    knn_novelty: float  # field-aggregated kNN novelty
    recombination: RecombinationResult
    field_weights: dict[str, float] | None = None  # weights used for knn_novelty
    net_new: float | None = None  # surfacing.net_new_ratio (LOO for corpus items)
    # set by finalize()
    blended_novelty: float = 0.0
    gate: float = 1.0
    raw: float = 0.0  # combined score; calibrated to system_score by percentile rank


# Raw signals whose LOO distributions are kept for rank-normalizing (key -> getter).
RAW_SIGNALS = {
    "knn_field": lambda c: c.knn_novelty,
    "net_new": lambda c: c.net_new if c.net_new is not None else 0.0,
    "recombination": lambda c: c.recombination.score,
}
# Signals rank-fused into blended novelty.
NOVELTY_SIGNALS = ("knn", "recombination")


def novelty_signal(comp: "Components", raw_dists: dict[str, np.ndarray] | None) -> float:
    """The "knn" novelty signal: field kNN novelty, optionally blended with net-new information."""
    mode, g = config.NET_NEW_BLEND, config.NET_NEW_GAMMA
    if mode == "off" or comp.net_new is None:
        return comp.knn_novelty
    if mode == "rank" and raw_dists:
        return (g * percentile_rank(comp.knn_novelty, raw_dists["knn_field"])
                + (1 - g) * percentile_rank(comp.net_new, raw_dists["net_new"]))
    return g * comp.knn_novelty + (1 - g) * comp.net_new


def _topk_mean(sims: np.ndarray, k: int) -> float:
    k = min(k, len(sims))
    return float(np.mean(np.partition(sims, -k)[-k:])) if k > 0 else 0.0


def _knn_novelty(sims: dict[str, np.ndarray], mask: np.ndarray, weights: dict[str, float]) -> float:
    """Field-weighted 1 - mean top-k cosine over reference items in `mask`."""
    if not mask.any():
        return 1.0
    return sum(w * (1.0 - _topk_mean(sims[f][mask], config.KNN_K)) for f, w in weights.items())


def field_weights(texts: dict[str, str] | None, token_df: dict[str, float] | None) -> dict[str, float]:
    base = config.KNN_FIELD_WEIGHTS
    if config.FIELD_AGGREGATION != "info_weighted" or texts is None or token_df is None:
        return dict(base)
    from scoring.baseline import tokens  # stopword-filtered tokenizer

    info = {}
    for f, w in base.items():
        toks = tokens(texts.get(f, ""))
        informative = sum(1 for t in toks if token_df.get(t, 0.0) < config.GENERIC_DF_FRAC)
        info[f] = w * math.log1p(informative)
    total = sum(info.values())
    return {f: v / total for f, v in info.items()} if total > 0 else dict(base)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0)))


def centroid(vecs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    c = vecs[mask].mean(axis=0)
    return c / max(float(np.linalg.norm(c)), 1e-12)


def measure(
    field_vecs: dict[str, np.ndarray],
    corpus: CorpusView,
    exclude: frozenset[int] = frozenset(),
    texts: dict[str, str] | None = None,
    net_new: float | None = None,
) -> Components:
    n = len(corpus.vecs["content"])
    ref = np.ones(n, dtype=bool)
    ref[list(exclude)] = False

    sims = {f: corpus.vecs[f] @ field_vecs[f] for f in config.FIELDS}
    masked = {f: np.where(ref, sims[f], -np.inf) for f in config.FIELDS}
    nn_idx = {f: int(np.argmax(masked[f])) for f in config.FIELDS}
    nn_sim = {f: float(masked[f][nn_idx[f]]) for f in config.FIELDS}

    relevance = float(corpus.fixed_vec @ field_vecs["content"])
    if config.RELEVANCE_GATE_SIGNAL == "centroid":
        gate_relevance = float(centroid(corpus.vecs["content"], ref) @ field_vecs["content"])
    else:
        gate_relevance = relevance

    weights = field_weights(texts, corpus.token_df)
    return Components(
        relevance=relevance,
        gate_relevance=gate_relevance,
        nn_sim=nn_sim,
        nn_idx=nn_idx,
        knn_novelty=_knn_novelty(sims, ref, weights),
        recombination=recombination_novelty(field_vecs, corpus.vecs, config.RECOMB_K, exclude),
        field_weights=weights,
        net_new=net_new,
    )


def percentile_rank(value: float, dist: np.ndarray) -> float:
    """Fraction of `dist` below `value` (ties count half). 0-1."""
    if len(dist) == 0:
        return 0.5
    return float((np.sum(dist < value) + 0.5 * np.sum(dist == value)) / len(dist))


def finalize(
    comp: Components,
    corpus: CorpusView,
    signal_dists: dict[str, np.ndarray] | None,
    mode: str | None = None,
) -> Components:
    """Blend novelty signals and combine with relevance. Mutates and returns `comp`.

    signal_dists: LOO distributions of RAW_SIGNALS plus "knn" (the novelty_signal values).
    """
    mode = mode or config.COMBINE_MODE
    w = config.BLEND_WEIGHTS
    vals = {"knn": novelty_signal(comp, signal_dists), "recombination": comp.recombination.score}
    if config.NORMALIZE_COMPONENTS and signal_dists:
        vals = {k: percentile_rank(v, signal_dists[k]) for k, v in vals.items()}
    blended = sum(w.get(k, 0.0) * vals[k] for k in NOVELTY_SIGNALS)

    gate = sigmoid((comp.gate_relevance - corpus.relevance_floor) / corpus.gate_temp)
    if mode == "harmonic":
        # Symmetric: relevance and novelty traded off equally, both on a 0-1 percentile scale.
        rel = (percentile_rank(comp.gate_relevance, corpus.gate_relevance_dist)
               if corpus.gate_relevance_dist is not None else max(comp.gate_relevance, 0.0))
        raw = 2 * blended * rel / (blended + rel) if blended + rel > 0 else 0.0
    elif mode == "none":
        raw = blended
    elif mode == "mmr":
        lam = config.MMR_LAMBDA
        raw = lam * comp.relevance - (1 - lam) * comp.nn_sim["content"]
    else:  # "gate" — Minimal Criteria Novelty Search: clear the relevance floor, then novelty decides
        raw = blended * gate

    comp.blended_novelty, comp.gate, comp.raw = blended, gate, raw
    return comp
