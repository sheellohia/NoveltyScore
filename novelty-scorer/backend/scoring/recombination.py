"""Atypical-recombination novelty (after Uzzi et al., 2013, "Atypical Combinations
and Scientific Impact").

Idea: embed each field of a submission separately and ask which corpus items each
field lands near. A conventional submission sits in one cluster on every axis:
its header, content and takeaway all point at the same neighbors. A submission
that bridges regions of the corpus that don't usually co-occur (e.g. a header
from one region of the corpus, a takeaway from another) has field neighborhoods that
barely overlap.

    recombination_novelty = 1 - mean(pairwise Jaccard of per-field top-k neighbor sets)
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from scoring.config import FIELDS, RECOMB_K


def neighbor_set(
    field_vec: np.ndarray,
    corpus_field_vecs: np.ndarray,
    k: int = RECOMB_K,
    exclude: frozenset[int] = frozenset(),
) -> set[int]:
    """Indices of the top-k corpus items by cosine (unit vectors), skipping `exclude`."""
    sims = corpus_field_vecs @ field_vec
    if exclude:
        sims = sims.copy()
        sims[list(exclude)] = -np.inf
    k = min(k, len(sims) - len(exclude))
    if k <= 0:
        return set()
    top = np.argpartition(-sims, k - 1)[:k]
    return {int(i) for i in top}


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


@dataclass
class RecombinationResult:
    score: float  # 0-1, higher = fields bridge less-related corpus regions
    pair_overlaps: dict[tuple[str, str], float]
    lowest_pair: tuple[str, str]
    lowest_overlap: float
    neighbors: dict[str, list[int]]  # field -> neighbor indices, nearest first


def recombination_novelty(
    field_vecs: dict[str, np.ndarray],
    corpus_vecs: dict[str, np.ndarray],
    k: int = RECOMB_K,
    exclude: frozenset[int] = frozenset(),
) -> RecombinationResult:
    sets = {f: neighbor_set(field_vecs[f], corpus_vecs[f], k, exclude) for f in FIELDS}
    overlaps = {(a, b): _jaccard(sets[a], sets[b]) for a, b in combinations(FIELDS, 2)}
    lowest_pair = min(overlaps, key=overlaps.get)

    ordered: dict[str, list[int]] = {}
    for f in FIELDS:
        sims = corpus_vecs[f] @ field_vecs[f]
        ordered[f] = sorted(sets[f], key=lambda i: -sims[i])

    return RecombinationResult(
        score=float(1.0 - np.mean(list(overlaps.values()))),
        pair_overlaps=overlaps,
        lowest_pair=lowest_pair,
        lowest_overlap=overlaps[lowest_pair],
        neighbors=ordered,
    )
