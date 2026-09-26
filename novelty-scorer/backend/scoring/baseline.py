"""Lexical baseline: TF-IDF nearest-neighbor novelty.

The deliberately naive reference the system must beat: no embeddings, no
relevance gate, no calibration.

    baseline_novelty = 1 - max cosine(TF-IDF(submission), TF-IDF(corpus item))

The IDF is fit on the topic corpus only, so a new submission never changes it.
"""

import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an and are as at be but by for from has have i if in into is it its of on or so that the their "
    "them they this to was we were will with you your our not no can more than".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def submission_text(sub: dict) -> str:
    return " ".join(sub[f] for f in ("header", "content", "takeaway"))


@dataclass
class TfidfIndex:
    vocab: dict[str, int]
    idf: np.ndarray
    matrix: np.ndarray  # (N, V) L2-normalized TF-IDF rows

    def transform(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float64)
        for t in tokens(text):
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def build_index(texts: list[str]) -> TfidfIndex:
    docs = [tokens(t) for t in texts]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    df = np.zeros(len(vocab))
    for d in docs:
        for t in set(d):
            df[vocab[t]] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1.0  # smoothed IDF
    index = TfidfIndex(vocab=vocab, idf=idf, matrix=np.zeros((len(docs), len(vocab))))
    index.matrix = np.stack([index.transform(t) for t in texts]) if texts else index.matrix
    return index


def novelty(index: TfidfIndex, text: str, exclude: int | None = None) -> float:
    sims = index.matrix @ index.transform(text)
    if exclude is not None:
        sims[exclude] = -np.inf
    return float(np.clip(1.0 - np.max(sims), 0.0, 1.0)) if len(sims) else 1.0


def loo_novelty(index: TfidfIndex, texts: list[str]) -> np.ndarray:
    """Each corpus item's baseline novelty against the other N-1."""
    return np.array([novelty(index, t, exclude=i) for i, t in enumerate(texts)])
