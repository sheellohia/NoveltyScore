"""Evaluate the system scorer against data/fixtures/test_fixtures.json.

Usage (from backend/):  .venv/bin/python scripts/eval_fixtures.py [--mode gate|harmonic|mmr]

Reports, per topic and overall:
  AUC          planted near-dup + generic items vs ordinary corpus items (originals excluded)
  planted<q25  planted items whose LOO system score is in the bottom quartile
  held-out     system_score for T1..T4 and how many of the 6 expected orderings hold
               (high > low_medium > low > very_low)
No LLM calls; embeddings are local.
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import load_fixtures, load_topics  # noqa: E402
from scoring import config  # noqa: E402
from scoring.corpus_cache import build_topic_cache  # noqa: E402
from scoring.pipeline import percentile_rank  # noqa: E402

ORDER = {"high": 3, "low_medium": 2, "low": 1, "very_low": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "harmonic", "mmr"], default=config.COMBINE_MODE)
    args = ap.parse_args()
    config.COMBINE_MODE = args.mode
    logging.disable(logging.WARNING)

    print(f"model={config.EMBED_MODEL} mode={args.mode} k={config.KNN_K} "
          f"normalize={config.NORMALIZE_COMPONENTS} gate={config.RELEVANCE_GATE_SIGNAL}/{config.RELEVANCE_FLOOR_METHOD}/"
          f"{config.GATE_TEMP_MODE}\n")
    print(f"{'topic':18s} {'AUC':>5s} {'q25':>5s}  {'T1':>5s} {'T4':>5s} {'T3':>5s} {'T2':>5s}  order")
    tot_auc, tot_q, tot_n, tot_ok, tot_pairs = [], 0, 0, 0, 0
    fixtures = load_fixtures()
    for tid, topic in load_topics().items():
        fx = fixtures.get(tid)
        if not fx:
            continue
        c = build_topic_cache(topic)
        idx = {s: i for i, s in enumerate(c.ids)}
        planted = [idx[d["id"]] for d in fx["near_duplicate_ids"]] + [idx[g] for g in fx["generic_ids"]]
        originals = {idx[d["duplicates"]] for d in fx["near_duplicate_ids"]}
        ordinary = [i for i in range(len(c.ids)) if i not in planted and i not in originals]
        r = c.loo_raw
        auc = float(np.mean([(r[p] < r[o]) + 0.5 * (r[p] == r[o]) for p in planted for o in ordinary]))
        q = sum(percentile_rank(r[p], r) < config.VALIDATION_QUANTILE for p in planted)

        sc = {}
        for case in fx["held_out_test_cases"]:
            s = case["submission"]
            sc[case["expect"]] = c.score_submission(s)[1]
        pairs = list(itertools.combinations(sc, 2))
        ok = sum((ORDER[a] - ORDER[b]) * (sc[a] - sc[b]) > 0 for a, b in pairs)

        print(f"{tid:18s} {auc:5.3f} {q:2d}/{len(planted):<2d}  "
              + " ".join(f"{sc.get(k, float('nan')):5.2f}" for k in ("high", "low_medium", "low", "very_low"))
              + f"  {ok}/{len(pairs)}")
        tot_auc.append(auc); tot_q += q; tot_n += len(planted); tot_ok += ok; tot_pairs += len(pairs)
    print(f"\n{'OVERALL':18s} {np.mean(tot_auc):5.3f} {tot_q}/{tot_n}  held-out order {tot_ok}/{tot_pairs}")


if __name__ == "__main__":
    main()
