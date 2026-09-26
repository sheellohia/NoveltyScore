"""Print and save sample surfacing output for held-out cases (no LLM calls).

Usage (from repo root):
    backend/.venv/bin/python experiments/surfacing_samples.py [--topic remote_work] [--cases T3 T1]
Writes results/surfacing_samples.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from data_loader import load_fixtures, load_topics  # noqa: E402
from scoring import config, surfacing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="remote_work")
    ap.add_argument("--cases", nargs="+", default=["T3", "T1"])
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    t = load_topics()[args.topic]
    ids = [s.id for s in t.submissions]
    idx = surfacing.build_index([s.content for s in t.submissions])
    out = {"topic": args.topic, "thresholds": {"novel_max_redundancy": idx.novel_max,
                                               "redundant_min_redundancy": idx.redundant_min}, "cases": {}}
    print(f"topic {args.topic}: novel if redundancy <= {idx.novel_max:.3f}, redundant if >= {idx.redundant_min:.3f} "
          f"({config.SURF_EMBED_MODEL})")
    for c in load_fixtures()[args.topic]["held_out_test_cases"]:
        key = c["case"].split("_")[0]
        if key not in args.cases:
            continue
        r = surfacing.surface(c["submission"]["content"], idx, ids)
        out["cases"][c["case"]] = r
        print(f"\n== {c['case']} (expect {c['expect']})  net_new_ratio = {r['net_new_ratio']:.2f}")
        for x in r["sentences"]:
            tag = f"≈ {x['closest_id']}" if x["label"] == "redundant" else f"(closest {x['closest_id']})"
            print(f"  [{x['label']:9s}] red={x['redundancy']:.3f} {tag:18s} {x['text'][:95]}")
            if x["label"] == "redundant":
                print(f"  {'':11s} already said: \"{x['closest_snippet']}\"")
    path = config.RESULTS_DIR / "surfacing_samples.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
