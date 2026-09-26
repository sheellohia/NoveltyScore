"""Score every corpus item with the judge once and cache it in results/judge_cache.json.

Usage (from repo root):
    backend/.venv/bin/python experiments/precache_judge.py [--runs 3] [--retries 3] [--wait 90]

Uses the listwise leave-one-out protocol (one call per topic per run; see
backend/scoring/judge.py::judge_corpus). Reads the cache first and only calls the API for
topics/runs that are missing under the CURRENT prompts. Safe to re-run; prints what it did.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from data_loader import load_topics  # noqa: E402
from scoring import judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--wait", type=int, default=90, help="seconds between retries")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    summary = {"cached": 0, "fetched": 0, "failed": 0}
    for run in range(args.runs):
        for tid, t in load_topics().items():
            topic = t.model_dump()
            if judge.cached_corpus(topic, run) is not None:
                summary["cached"] += 1
                print(f"cached   {tid:18s} run{run}")
                continue
            for attempt in range(args.retries):
                try:
                    judge.judge_corpus(topic, run=run)
                    summary["fetched"] += 1
                    print(f"fetched  {tid:18s} run{run}")
                    break
                except judge.JudgeUnavailable as e:
                    print(f"retry    {tid:18s} run{run} ({e.reason}), attempt {attempt + 1}/{args.retries}")
                    if attempt + 1 < args.retries:
                        time.sleep(args.wait)
            else:
                summary["failed"] += 1
                print(f"FAILED   {tid:18s} run{run}")
    print(f"\n{summary}  (a legacy set from older prompts, if present, is still used by "
          f"run_experiments.py --cache-only)")


if __name__ == "__main__":
    main()
