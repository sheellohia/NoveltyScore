"""Redundancy-aware surfacing: which sentences of a submission are new information.

Sentence-level version of the TREC Novelty Track task (Harman 2002; Soboroff & Harman
2003): for each sentence, is it already said somewhere in the existing submissions?

    redundancy(sentence)            = max cosine to any sentence of any corpus submission
    novelty_contribution(sentence)  = 1 - redundancy

Labels are self-calibrated per topic from the corpus's own leave-one-out distribution:
each corpus sentence's redundancy against the sentences of the OTHER submissions.

    redundant  redundancy >= SURF_REDUNDANT_PCT percentile of that distribution
    novel      redundancy <= SURF_NOVEL_PCT percentile
    partial    in between

net_new_ratio = word-weighted share of the content that is novel or partial.

This is explanatory and reported alongside the score; it does not change system_score.
"""

import re
from dataclasses import dataclass

import numpy as np

from scoring import config
from scoring.embed import embed

# Split after . ! ? (optionally followed by a closing quote/bracket) when whitespace follows.
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
MIN_WORDS = 3  # fragments shorter than this are merged into the previous sentence


def segment(text: str) -> list[str]:
    """Split text into sentences. Regex-based, no external dependencies."""
    parts = [p.strip() for p in _SPLIT.split(text.strip()) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(p.split()) < MIN_WORDS:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


@dataclass
class SentenceIndex:
    texts: list[str]  # corpus sentences
    owner: np.ndarray  # (S,) index of the corpus submission each sentence came from
    vecs: np.ndarray  # (S, d) unit vectors
    loo_redundancy: np.ndarray  # (S,) each sentence's max cosine to OTHER submissions' sentences
    novel_max: float  # redundancy at or below this -> "novel"
    redundant_min: float  # redundancy at or above this -> "redundant"


def build_index(contents: list[str]) -> SentenceIndex:
    texts, owner = [], []
    for i, c in enumerate(contents):
        for s in segment(c):
            texts.append(s)
            owner.append(i)
    owner_arr = np.array(owner, dtype=int)
    vecs = embed(texts, config.SURF_EMBED_MODEL) if texts else np.zeros((0, 0), dtype=np.float32)
    if len(texts) > 1:
        sims = vecs @ vecs.T
        sims[owner_arr[:, None] == owner_arr[None, :]] = -np.inf  # never match your own submission
        loo = sims.max(axis=1)
        loo = np.where(np.isfinite(loo), loo, 0.0)
    else:
        loo = np.zeros(len(texts))
    if len(loo) >= config.MIN_DIST_SIZE and float(np.std(loo)) > config.MIN_DIST_STD:
        novel_max = float(np.percentile(loo, config.SURF_NOVEL_PCT))
        redundant_min = float(np.percentile(loo, config.SURF_REDUNDANT_PCT))
    else:
        novel_max, redundant_min = config.SURF_NOVEL_MAX_FALLBACK, config.SURF_REDUNDANT_MIN_FALLBACK
    return SentenceIndex(texts, owner_arr, vecs, loo, novel_max, redundant_min)


def label_for(redundancy: float, index: SentenceIndex) -> str:
    if redundancy >= index.redundant_min:
        return "redundant"
    if redundancy <= index.novel_max:
        return "novel"
    return "partial"


def _snippet(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def surface(content: str, index: SentenceIndex, corpus_ids: list[str], exclude_owner: int | None = None) -> dict:
    """Per-sentence labels for `content` against the corpus sentence index.

    exclude_owner: corpus submission index to ignore (for scoring a corpus item leave-one-out).
    """
    sentences = segment(content)
    if not sentences:
        return {"net_new_ratio": 0.0, "sentences": []}
    vecs = embed(sentences, config.SURF_EMBED_MODEL)
    sims = vecs @ index.vecs.T if len(index.texts) else np.zeros((len(sentences), 0))
    if exclude_owner is not None and sims.shape[1]:
        sims[:, index.owner == exclude_owner] = -np.inf

    out, words_new, words_total = [], 0, 0
    for s, row in zip(sentences, sims):
        if row.size and np.isfinite(row).any():
            j = int(np.argmax(row))
            red = float(row[j])
            closest_id, closest = corpus_ids[index.owner[j]], _snippet(index.texts[j])
        else:
            red, closest_id, closest = 0.0, None, None
        lab = label_for(red, index)
        n = len(s.split())
        words_total += n
        words_new += n if lab in ("novel", "partial") else 0
        out.append({"text": s, "label": lab, "redundancy": round(red, 4),
                    "novelty_contribution": round(1.0 - red, 4),
                    "closest_id": closest_id, "closest_snippet": closest})
    return {"net_new_ratio": round(words_new / words_total, 4) if words_total else 0.0, "sentences": out}
