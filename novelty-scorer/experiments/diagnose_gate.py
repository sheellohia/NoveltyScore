"""Relevance-gate diagnosis: where the floor sits vs on-topic and off-topic relevance.

Usage (from repo root):
    backend/.venv/bin/python experiments/diagnose_gate.py --label before|after

Writes results/gate_diagnosis_{label}.json and prints a readable summary. Uses the
pipeline exactly as currently configured, so the same script measures before and after.
No LLM calls.

Per topic:
  on-topic relevance   cos(content, this fixed_content) for this topic's corpus
  off-topic relevance  cos(content, this fixed_content) for the OTHER topics' corpora
  corpus gate          soft_gate for every corpus item (LOO)
  sanity items         lowest-relevance item, median-relevance item, first planted generic
  held-out cases       T1..T4: relevance, gate, final system score
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from data_loader import load_fixtures, load_topics  # noqa: E402
from scoring import config  # noqa: E402
from scoring.corpus_cache import build_topic_cache  # noqa: E402
from scoring.embed import embed  # noqa: E402
from scoring.pipeline import percentile_rank, sigmoid  # noqa: E402

# The four-day-week submission used in the earlier UI test.
UI_TEST = {
    "topic_id": "four_day_week",
    "header": "Four-day weeks will reshape childcare markets",
    "content": ("If millions of parents are home every Friday, daycare centres lose a fifth of weekday demand. "
                "Many run on thin margins and fixed staffing, so providers may shift to four-day contracts, raise "
                "prices, or close. Parents on five-day shifts, often lower-paid service workers, could find care "
                "harder to get. The policy aimed at wellbeing may quietly widen a class gap in childcare access."),
    "takeaway": "A shorter week for some could mean worse childcare for others.",
}


def score_sub(cache, sub):
    comp, system = cache.score_submission(sub)
    return {"relevance": comp.relevance, "gate_relevance": comp.gate_relevance, "floor": cache.relevance_floor, "gate": comp.gate,
            "novelty_pct": percentile_rank(comp.blended_novelty, cache.loo_component_dists["novelty"]),
            "system": system}


def q(x, p):
    return float(np.percentile(x, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    config.set_seeds()
    topics, fixtures = load_topics(), load_fixtures()
    out = {"label": args.label, "gate_signal": config.RELEVANCE_GATE_SIGNAL, "floor_method": config.RELEVANCE_FLOOR_METHOD,
           "gate_temp_mode": config.GATE_TEMP_MODE, "topics": {}}

    for tid, topic in topics.items():
        c = build_topic_cache(topic)
        fx = fixtures[tid]
        on = c.relevance_dist  # absolute relevance (fixed content)
        off = embed([s.content for t2, tp in topics.items() if t2 != tid for s in tp.submissions]) @ c.view.fixed_vec
        gon, goff = c.gate_relevance_dist, c.off_topic_dist  # the gate's own signal
        gates = np.array([L.gate for L in c.loo])
        sys_pct = np.array([percentile_rank(x, c.loo_raw) for x in c.loo_raw])
        nov = c.loo_component_dists["novelty"]
        nov_pct = np.array([percentile_rank(x, nov) for x in nov])
        ids = c.ids

        def item(i, why):
            return {"id": ids[i], "why": why, "relevance": float(on[i]), "gate_relevance": float(gon[i]), "floor": c.relevance_floor,
                    "gate": float(gates[i]), "novelty_pct": float(nov_pct[i]), "system": float(sys_pct[i])}

        order = np.argsort(on)
        generics = [ids.index(g) for g in fx["generic_ids"]]
        out["topics"][tid] = {
            "floor": c.relevance_floor,
            "floor_detail": c.floor_detail,
            "gate_signal": config.RELEVANCE_GATE_SIGNAL,
            "gate_temp": c.view.gate_temp,
            "gate_on_topic": {"p5": q(gon, 5), "min": float(gon.min())},
            "gate_off_topic": {"mean": float(goff.mean()), "std": float(goff.std())} if len(goff) else None,
            "on_topic": {"min": float(on.min()), "p5": q(on, 5), "p10": q(on, 10), "median": q(on, 50), "max": float(on.max())},
            "off_topic": {"n": len(off), "mean": float(off.mean()), "std": float(off.std()), "p95": q(off, 95), "max": float(off.max())},
            "corpus_gate": {"mean": float(gates.mean()), "min": float(gates.min()),
                            "frac_below_0.5": float((gates < 0.5).mean()), "frac_below_0.9": float((gates < 0.9).mean())},
            "off_topic_gate_mean": float(np.mean([sigmoid((r - c.relevance_floor) / c.view.gate_temp) for r in goff])) if len(goff) else None,
            "sanity": [item(int(order[0]), "lowest relevance"), item(int(order[len(order) // 2]), "median relevance"),
                       item(generics[0], "planted generic")],
            "generics": [item(i, "planted generic") for i in generics],
            "cases": {cse["case"].split("_")[0]: score_sub(c, cse["submission"]) for cse in fx["held_out_test_cases"]},
        }
        if tid == UI_TEST["topic_id"]:
            out["ui_test_submission"] = score_sub(c, UI_TEST)

    path = config.RESULTS_DIR / f"gate_diagnosis_{args.label}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"== gate diagnosis [{args.label}]  signal={out['gate_signal']} floor={out['floor_method']} temp_mode={out['gate_temp_mode']}")
    for tid, d in out["topics"].items():
        on, off, g = d["on_topic"], d["off_topic"], d["corpus_gate"]
        print(f"\n{tid}: [{d['gate_signal']}] floor {d['floor']:.3f} temp {d['gate_temp']:.4f} ({d['floor_detail'].get('binding', d['floor_detail']['method'])}) | abs on-topic p5 {on['p5']:.3f} med {on['median']:.3f} min {on['min']:.3f} "
              f"| off-topic mean {off['mean']:.3f} sd {off['std']:.3f} p95 {off['p95']:.3f} max {off['max']:.3f}")
        print(f"   corpus gate mean {g['mean']:.3f} min {g['min']:.3f}  <0.5: {g['frac_below_0.5']:.0%}  <0.9: {g['frac_below_0.9']:.0%}"
              f"  | other-topic items' mean gate {d['off_topic_gate_mean']:.3f}")
        for s in d["sanity"]:
            print(f"   sanity {s['id']:4s} ({s['why']:16s}) rel {s['relevance']:.3f} gate_sig {s['gate_relevance']:.3f} floor {s['floor']:.3f} gate {s['gate']:.3f} "
                  f"novelty_pct {s['novelty_pct']:.2f} final {s['system']:.2f}")
        cs = d["cases"]
        print("   cases " + "  ".join(f"{k}: sig {v['gate_relevance']:.3f} gate {v['gate']:.2f} final {v['system']:.2f}" for k, v in sorted(cs.items())))
    u = out.get("ui_test_submission")
    if u:
        print(f"\nUI four-day-week submission: rel {u['relevance']:.3f} gate_sig {u['gate_relevance']:.3f} floor {u['floor']:.3f} gate {u['gate']:.3f} final {u['system']:.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
