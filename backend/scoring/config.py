"""Paths and tunables for the scoring pipeline.

Paths resolve from this file's location, so they work regardless of the
process CWD (uvicorn from backend/, scripts from repo root, tests, etc.).
"""

import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scoring/config.py -> repo root

DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
FIXTURES_PATH = DATA_DIR / "fixtures" / "test_fixtures.json"

RESULTS_DIR = REPO_ROOT / "results"
JUDGE_CACHE_PATH = RESULTS_DIR / "judge_cache.json"

# --- Reproducibility ---------------------------------------------------------
SEED = 1234


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG the pipeline could touch. Scoring itself is deterministic
    (ONNX inference + fixed arithmetic); this pins anything incidental."""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:  # torch is not a dependency (fastembed uses ONNX), but seed it if present
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

# --- Embeddings (local, via fastembed/ONNX) ---------------------------------
# Chosen by ablation on data/fixtures (backend/scripts/eval_fixtures.py): best planted-item
# AUC of bge-small / bge-base / nomic-v1.5 / arctic-embed-m. See experiments/ for the
# judge-correlation sweep.
EMBED_MODEL = "snowflake/snowflake-arctic-embed-m"
# Models fastembed doesn't ship, registered from their official ONNX export on HF.
CUSTOM_EMBED_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": {"dim": 768, "model_file": "onnx/model.onnx"},
}
EMBED_CACHE_DIR = REPO_ROOT / "backend" / ".cache" / "fastembed"

FIELDS = ("header", "content", "takeaway")

# --- kNN novelty -------------------------------------------------------------
# k=1 (nearest-neighbor novelty): averaging top-5 diluted the one close neighbor that
# marks a near-duplicate (planted-dup AUC 0.85 -> 0.88 at k=1 in ablation).
KNN_K = 1
# Per-field weight when averaging field-level kNN novelty (sums to 1).
KNN_FIELD_WEIGHTS = {"header": 0.2, "content": 0.6, "takeaway": 0.2}
# "fixed_mean"    : use KNN_FIELD_WEIGHTS as-is
# "info_weighted" : w_f ∝ KNN_FIELD_WEIGHTS[f] * log1p(#non-generic tokens in field f), renormalized,
#                   so a short or generic field cannot dilute a substantive one. A token is
#                   generic if it is a stopword or appears in >= GENERIC_DF_FRAC of the topic's
#                   corpus submissions.
# Decided by the pre-registered ablation (results/experiments.json -> aggregation):
# info_weighted KEPT (higher LOO Spearman vs judge, same planted pass-rate, LOTO-stable).
FIELD_AGGREGATION = "info_weighted"
GENERIC_DF_FRAC = 0.2

# Blend sentence-level new information (surfacing.net_new_ratio) into the kNN novelty signal:
# "off"  : kNN field novelty only
# "raw"  : NET_NEW_GAMMA * field_novelty + (1 - NET_NEW_GAMMA) * net_new_ratio
# "rank" : the same mix of their within-topic LOO percentiles (scale-free variant)
# Both blends REVERTED by the same ablation (lower Spearman; "raw" also lost pass-rate).
NET_NEW_BLEND = "off"
NET_NEW_GAMMA = 0.6

# --- Recombination (Uzzi-style cross-field atypicality) ----------------------
RECOMB_K = 5

# --- Blend -------------------------------------------------------------------
# blended_novelty = knn * w_knn + recombination * w_recomb
# (the original 0.6 / 0.25 split, renormalized to sum to 1)
BLEND_WEIGHTS = {"knn": 0.6 / 0.85, "recombination": 0.25 / 0.85}
# Map each signal to its percentile within the topic's LOO distribution before
# blending, so the weights act on one common 0-1 scale.
NORMALIZE_COMPONENTS = True

# --- Combine novelty with relevance ------------------------------------------
# "gate"     : blended * sigmoid((relevance - relevance_floor) / GATE_TEMP)  [default]
#              (asymmetric gate, Minimal Criteria Novelty Search)
# "harmonic" : harmonic mean of blended novelty and relevance percentile (symmetric)
# "mmr"      : MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * nearest-neighbor similarity
# "none"     : blended novelty only, relevance ignored (ablation)
# Override with NOVELTY_COMBINE_MODE=harmonic|mmr|none for ablations.
COMBINE_MODE = os.getenv("NOVELTY_COMBINE_MODE", "gate").strip().lower()
# Absolute gate temperature, used when GATE_TEMP_MODE == "fixed".
GATE_TEMP = 0.03
# "adaptive": per topic, temp = (on_topic_p5 - floor) / logit(GATE_TARGET_AT_ON_P5), so an
# item at the on-topic 5th percentile gets gate == GATE_TARGET_AT_ON_P5 regardless of the
# embedding's cosine scale. "fixed": use GATE_TEMP.
GATE_TEMP_MODE = "adaptive"
GATE_TARGET_AT_ON_P5 = 0.95

# --- Relevance gate signal and floor -------------------------------------------
# Signal the gate thresholds (breakdown.relevance always shows cos(content, fixed_content)):
#   "fixed"    : cos(content, fixed_content)
#   "centroid" : cos(content, centroid of the topic's corpus content), LOO for corpus items
# "centroid" chosen by experiments/diagnose_gate.py: cos-to-fixed-content cannot separate an
# off-topic text that shares vocabulary with the fixed content (ai_tutors T2 "octopus
# intelligence" scored 0.584, above that topic's lowest on-topic item, 0.525); cos-to-corpus-
# centroid put every held-out T2 below every on-topic item.
RELEVANCE_GATE_SIGNAL = "centroid"
# Floor method:
#   "offtopic"   : min(off_mean + FLOOR_OFFTOPIC_K * off_std, on_topic_p5 - FLOOR_MARGIN), where
#                  "off" = the other topics' corpus content scored against this topic
#   "percentile" : RELEVANCE_FLOOR_PCT percentile of on-topic relevance (pre-fix behaviour)
RELEVANCE_FLOOR_METHOD = "offtopic"
FLOOR_OFFTOPIC_K = 2.0
FLOOR_MARGIN = 0.02
FLOOR_ON_TOPIC_PCT = 5
MMR_LAMBDA = 0.5

# --- Per-topic calibration (derived from leave-one-out corpus distributions) --
# When False, thresholds come from the *_FALLBACK constants for every topic (ablation).
CALIBRATE_THRESHOLDS = True
RELEVANCE_FLOOR_PCT = 10  # relevance_floor = this percentile of corpus relevance
DEDUP_PCT = 95  # dedup_threshold = this percentile of within-corpus NN similarity
MIN_DIST_SIZE = 5  # fewer points than this => distribution is degenerate
MIN_DIST_STD = 1e-6

# Fallbacks, used only when a topic's distribution is degenerate.
# Scaled for arctic-embed-m cosine similarities.
RELEVANCE_FLOOR_FALLBACK = 0.60
DEDUP_THRESHOLD_FALLBACK = 0.90

# --- Redundancy-aware surfacing (scoring/surfacing.py) -------------------------
# Percentiles of the corpus's own LOO sentence-redundancy distribution (max cosine of each
# corpus sentence to any sentence of another submission).
# Sentence redundancy is paraphrase detection, so it uses a paraphrase-trained model rather
# than the scoring embedding; see results/experiments.json -> surfacing_model_sweep.
SURF_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SURF_NOVEL_PCT = 50  # redundancy <= p50 -> "novel"
SURF_REDUNDANT_PCT = 75  # redundancy >= p75 -> "redundant"
SURF_NOVEL_MAX_FALLBACK = 0.75  # used only if the distribution is degenerate
SURF_REDUNDANT_MIN_FALLBACK = 0.90

# Startup validation: planted near-duplicate / generic ids should rank below this.
VALIDATION_QUANTILE = 0.25
