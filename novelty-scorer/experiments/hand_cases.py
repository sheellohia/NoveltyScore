"""Eyeball table for hand-written test cases: system score before/after an aggregation change,
next to the judge. For inspection only; KEEP/REVERT decisions come from run_experiments.py.

Usage (from repo root):
    backend/.venv/bin/python experiments/hand_cases.py [--call-judge]

Reads data/fixtures/hand_cases.json:
    [{"name": "...", "topic_id": "...", "header": "...", "content": "...", "takeaway": "..."}, ...]
The judge column uses a cached live verdict when one exists; with --call-judge it calls the API
(result is cached). Otherwise it prints "not measured". Writes results/hand_cases.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "experiments"))

from data_loader import load_topics  # noqa: E402
from run_experiments import BASE_AGG, override  # noqa: E402
from scoring import config, judge  # noqa: E402
from scoring.corpus_cache import build_topic_cache  # noqa: E402

CASES_PATH = config.DATA_DIR / "fixtures" / "hand_cases.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-judge", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    if not CASES_PATH.exists():
        print(f"{CASES_PATH} not found. Add the hand-written cases there (see this file's docstring); "
              "nothing was measured.")
        return
    cases = json.loads(CASES_PATH.read_text())
    topics = load_topics()
    rows = []
    for c in cases:
        topic = topics[c["topic_id"]]
        with override(**BASE_AGG):
            before = build_topic_cache(topic).score_submission(c)[1]
        after = build_topic_cache(topic).score_submission(c)[1]
        jscore, jsrc = None, "not measured"
        hit = judge._cache().get(judge.live_key(c, topic.model_dump()))
        if hit:
            jscore, jsrc = hit["score"], "cached"
        elif args.call_judge:
            try:
                jscore, jsrc = judge.judge(c, topic.model_dump()).score, "live"
            except judge.JudgeUnavailable as e:
                jsrc = f"unavailable ({e.reason})"
        rows.append({"name": c["name"], "topic_id": c["topic_id"], "system_before": before, "system_after": after,
                     "judge": jscore, "judge_source": jsrc,
                     "gap_before": None if jscore is None else abs(before - jscore),
                     "gap_after": None if jscore is None else abs(after - jscore)})
    fmt = lambda x: "  n/a" if x is None else f"{x:5.2f}"
    print(f"{'case':36s} {'before':>6s} {'after':>6s} {'judge':>6s}  judge source   |gap| before -> after")
    for r in rows:
        print(f"{r['name'][:36]:36s} {fmt(r['system_before'])}  {fmt(r['system_after'])}  {fmt(r['judge'])}  "
              f"{r['judge_source']:14s} {fmt(r['gap_before'])} -> {fmt(r['gap_after'])}")
    (config.RESULTS_DIR / "hand_cases.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
