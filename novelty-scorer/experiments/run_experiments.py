"""Experiment sweep: score configurations against the cached Gemini judge.

Usage (from repo root):
    backend/.venv/bin/python experiments/run_experiments.py

Writes:
    results/experiments.json     everything (per-config, per-topic, LOTO, determinism, judge variance)
    results/summary.csv          one flat row per config
    results/criteria_report.json C1-C6: target, actual, verdict

Judge scores come ONLY from results/judge_cache.json (listwise LOO protocol, see
backend/scoring/judge.py::judge_corpus). Missing judge runs are fetched once and
cached; if the API is unavailable, metrics that need them are recorded as null
("not measured"), never estimated.
"""

import copy
import csv
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from data_loader import load_fixtures, load_topics  # noqa: E402
from scoring import baseline, config, surfacing  # noqa: E402
from scoring import embed as embed_mod  # noqa: E402
from scoring import judge as judge_mod  # noqa: E402
from scoring.corpus_cache import build_topic_cache  # noqa: E402
from scoring.pipeline import percentile_rank  # noqa: E402

log = logging.getLogger("experiments")

JUDGE_RUNS = 3  # run 0 = reference oracle; runs 1-2 (shuffled order) only for variance
DEPLOYED_EMBED = "snowflake/snowflake-arctic-embed-m"
EMBEDDINGS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}
FULL_WEIGHTS = {"knn": 0.6 / 0.85, "recombination": 0.25 / 0.85}

# ---------------------------------------------------------------------------
# Success criteria (explicit, checkable). PARTIAL bands are defined up front,
# not chosen after seeing results.
# ---------------------------------------------------------------------------
CRITERIA = {
    "C1": {"name": "Adversarial rejection", "target": "max T2 across topics < 0.2",
           "pass": lambda a: a["t2_max"] < 0.2, "partial": lambda a: a["t2_mean"] < 0.2},
    "C2": {"name": "Ordering correctness", "target": "T1>T4>T3>T2 holds for >= 4/5 topics",
           "pass": lambda a: a["order_topics"] >= 4, "partial": lambda a: a["order_topics"] >= 3},
    "C3": {"name": "Redundancy detection", "target": ">= 80% of planted dupes+generics in bottom quartile",
           "pass": lambda a: a["redundancy"] >= 0.80, "partial": lambda a: a["redundancy"] >= 0.60},
    "C4": {"name": "Judge fidelity", "target": "system LOO Spearman > baseline Spearman",
           "pass": lambda a: a["system_spearman"] > a["baseline_spearman"], "partial": lambda a: False},
    "C5": {"name": "Generalization", "target": "mean per-fold |in-sample Spearman - LOTO held-out Spearman| < 0.15",
           "pass": lambda a: a["gap"] < 0.15, "partial": lambda a: a["gap"] < 0.25},
    "C6": {"name": "Determinism", "target": "system score variance across 3 identical runs == 0 "
                                             "(judge run-to-run variance reported for contrast)",
           "pass": lambda a: a["system_max_var"] == 0.0, "partial": lambda a: False},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for v in np.unique(x):  # average ranks for ties
            m = x == v
            r[m] = r[m].mean()
        return r

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


class override:
    """Temporarily set scoring.config attributes."""

    def __init__(self, **kw):
        self.kw, self.saved = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = copy.deepcopy(getattr(config, k))
            setattr(config, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(config, k, v)


# ---------------------------------------------------------------------------
# judge data (cache only; fetch missing runs once)
# ---------------------------------------------------------------------------
def load_judge(topics, cache_only: bool = False) -> dict:
    """{run: {topic_id: {item_id: score}}}; a topic/run is absent if unavailable."""
    out: dict[int, dict] = {r: {} for r in range(JUDGE_RUNS)}
    models: dict[str, str] = {}
    versions: dict[str, str] = {}
    for run in range(JUDGE_RUNS):
        # One prompt version per run across ALL topics: use current verdicts only if every topic has them.
        all_current = all(judge_mod.cached_corpus(t.model_dump(), run) is not None for t in topics.values())
        for tid, t in topics.items():
            res, version = judge_mod.cached_corpus_with_fallback(t.model_dump(), run, allow_current=all_current)
            if version != "current" and not cache_only and not all_current:
                try:
                    res, version = judge_mod.judge_corpus(t.model_dump(), run=run), "current"
                except judge_mod.JudgeUnavailable as e:
                    log.warning("judge run %d for %s: API unavailable (%s); using %s", run, tid, str(e)[:80], version)
            if res is None:
                log.warning("judge run %d for %s not available", run, tid)
                continue
            versions[f"{tid}/run{run}"] = version
            out[run][tid] = {sid: r["score"] for sid, r in res.items()}
            models[f"{tid}/run{run}"] = next(iter(res.values()))["model"]
    return {"scores": out, "models": models, "prompt_versions": versions}


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------
def evaluate(topics, fixtures, judge0, cfg: dict) -> dict:
    """cfg: {"kind": "baseline"|"system", "calibrated": bool, "overrides": {...}}"""
    per_topic = {}
    with override(**cfg.get("overrides", {})):
        for tid, topic in topics.items():
            cache = build_topic_cache(topic)
            fx = fixtures.get(tid, {})
            ids = cache.ids

            if cfg["kind"] == "baseline":
                dist = cache.baseline_loo
                loo_scores = dist.copy()

                def score_case(sub, c=cache):
                    return baseline.novelty(c.tfidf, baseline.submission_text(sub))
            else:
                dist = cache.loo_raw
                loo_scores = (np.array([percentile_rank(x, dist) for x in dist]) if cfg["calibrated"] else dist.copy())

                def score_case(sub, c=cache, d=dist):
                    comp, pctl = c.score_submission(sub)
                    diag[sub["header"]] = {"relevance": comp.relevance, "relevance_floor": c.relevance_floor,
                                           "gate": comp.gate, "blended_novelty": comp.blended_novelty}
                    return pctl if cfg["calibrated"] else comp.raw

            gate_cov = float(np.mean([L.gate >= 0.9 for L in cache.loo])) if cfg["kind"] == "system" else None

            # 1. judge correlation over corpus items (LOO)
            j = judge0.get(tid)
            rho = spearman(loo_scores, [j[s] for s in ids]) if j else None

            # 4. redundancy: planted items in bottom quartile of this config's LOO distribution
            idx = {s: i for i, s in enumerate(ids)}
            planted = [d["id"] for d in fx.get("near_duplicate_ids", [])] + list(fx.get("generic_ids", []))
            ranks = {s: percentile_rank(loo_scores[idx[s]], loo_scores) for s in planted if s in idx}

            # diagnostic: does the embedding find each planted copy's original, and is that
            # similarity distinctive within the topic? (system configs only)
            nn_hits, nn_pcts = [], []
            if cfg["kind"] == "system":
                nn_dist = cache.nn_sim_dist["content"]
                for d in fx.get("near_duplicate_ids", []):
                    if d["id"] in idx and d.get("duplicates") in idx:
                        L = cache.loo[idx[d["id"]]]
                        nn_hits.append(ids[L.nn_idx["content"]] == d["duplicates"])
                        nn_pcts.append(percentile_rank(L.nn_sim["content"], nn_dist))

            # 3/5. held-out cases
            diag: dict[str, dict] = {}
            cases = {c["case"].split("_")[0]: float(score_case(c["submission"])) for c in fx.get("held_out_test_cases", [])}
            case_diag = {c["case"].split("_")[0]: diag[c["submission"]["header"]]
                         for c in fx.get("held_out_test_cases", []) if c["submission"]["header"] in diag}
            t = cases
            ordered = all(k in t for k in ("T1", "T2", "T3", "T4")) and t["T1"] > t["T4"] > t["T3"] > t["T2"]
            per_topic[tid] = {
                "ids": ids,
                "spearman": rho,
                "on_topic_gate_ge_0.9": gate_cov,
                "relevance": [round(float(x), 6) for x in cache.relevance_dist] if cfg["kind"] == "system" else None,
                "gates": [round(float(L.gate), 6) for L in cache.loo] if cfg["kind"] == "system" else None,
                "floor": cache.relevance_floor if cfg["kind"] == "system" else None,
                "floor_detail": cache.floor_detail if cfg["kind"] == "system" else None,
                "loo_scores": [round(float(x), 6) for x in loo_scores],
                "planted_ranks": ranks,
                "planted_in_bottom_q": sum(r < config.VALIDATION_QUANTILE for r in ranks.values()),
                "planted_n": len(ranks),
                "dup_nn_hits": [bool(h) for h in nn_hits],
                "dup_nn_sim_pct": nn_pcts,
                "cases": t,
                "case_diagnostics": case_diag,
                "ordered": bool(ordered),
                "case_pass": bool(ordered and t["T2"] < 0.2 and t["T1"] > 0.6),
            }

    pooled_sys, pooled_j = [], []
    for tid, pt in per_topic.items():
        if tid in judge0:
            pooled_sys += pt["loo_scores"]
            pooled_j += [judge0[tid][s] for s in pt["ids"]]
    n_planted = sum(pt["planted_n"] for pt in per_topic.values())
    t2 = [pt["cases"]["T2"] for pt in per_topic.values() if "T2" in pt["cases"]]
    return {
        "spearman_mean": mean_or_none([pt["spearman"] for pt in per_topic.values()]),
        "spearman_pooled": spearman(pooled_sys, pooled_j) if pooled_j else None,
        "planted_case_pass_rate": sum(pt["case_pass"] for pt in per_topic.values()) / len(per_topic),
        "order_topics": sum(pt["ordered"] for pt in per_topic.values()),
        "redundancy_bottom_q": (sum(pt["planted_in_bottom_q"] for pt in per_topic.values()) / n_planted) if n_planted else None,
        "t2_mean": float(np.mean(t2)) if t2 else None,
        "t2_max": float(np.max(t2)) if t2 else None,
        "t1_mean": mean_or_none([pt["cases"].get("T1") for pt in per_topic.values()]),
        "on_topic_gate_coverage": mean_or_none([pt["on_topic_gate_ge_0.9"] for pt in per_topic.values()]),
        "dup_nn_hit_rate": mean_or_none([h for pt in per_topic.values() for h in pt["dup_nn_hits"]]),
        "dup_nn_sim_pct_mean": mean_or_none([x for pt in per_topic.values() for x in pt["dup_nn_sim_pct"]]),
        "per_topic": per_topic,
    }


# ---------------------------------------------------------------------------
# configuration sets
# ---------------------------------------------------------------------------
PREFIX_GATE = {"RELEVANCE_GATE_SIGNAL": "fixed", "RELEVANCE_FLOOR_METHOD": "percentile",
               "GATE_TEMP_MODE": "fixed", "GATE_TEMP": 0.03}
FIXED_GATE = {"RELEVANCE_GATE_SIGNAL": "centroid", "RELEVANCE_FLOOR_METHOD": "offtopic", "GATE_TEMP_MODE": "adaptive"}


DEPLOYED_AGG = {"FIELD_AGGREGATION": config.FIELD_AGGREGATION, "NET_NEW_BLEND": config.NET_NEW_BLEND,
                "NET_NEW_GAMMA": config.NET_NEW_GAMMA}
BASE_AGG = {"FIELD_AGGREGATION": "fixed_mean", "NET_NEW_BLEND": "off", "NET_NEW_GAMMA": config.NET_NEW_GAMMA}


def system_overrides(embed, combine, gate, weights=None, calibrated=True, gate_knobs=None, agg=None) -> dict:
    return {
        **(DEPLOYED_AGG if agg is None else agg),
        "EMBED_MODEL": embed,
        "NORMALIZE_COMPONENTS": combine == "rankfusion",
        "COMBINE_MODE": {"asymmetric": "gate", "harmonic": "harmonic", "none": "none"}[gate],
        "BLEND_WEIGHTS": weights or FULL_WEIGHTS,
        "CALIBRATE_THRESHOLDS": calibrated,
        **(FIXED_GATE if gate_knobs is None else gate_knobs),
    }


def matrix_configs() -> list[dict]:
    rows = []
    for (ename, emodel), combine, gate in itertools.product(EMBEDDINGS.items(), ["linear", "rankfusion"], ["harmonic", "asymmetric"]):
        rows.append({
            "name": f"{ename} | {combine} | {gate}", "group": "matrix", "kind": "system", "calibrated": True,
            "embedding": ename, "combine": combine, "gate": gate,
            "overrides": system_overrides(emodel, combine, gate),
        })
    # Deployed embedding, same 2x2 (beyond the requested 8): gives the deployed choices a direct comparison.
    for combine, gate in itertools.product(["linear", "rankfusion"], ["harmonic", "asymmetric"]):
        deployed = combine == "rankfusion" and gate == "asymmetric"
        rows.append({
            "name": f"arctic-embed-m | {combine} | {gate}" + (" (deployed)" if deployed else ""),
            "group": "reference", "kind": "system", "calibrated": True, "embedding": "snowflake-arctic-embed-m",
            "combine": combine, "gate": gate, "overrides": system_overrides(DEPLOYED_EMBED, combine, gate),
        })
    return rows


def ladder_configs() -> list[dict]:
    knn_only = {"knn": 1.0, "recombination": 0.0}
    base = {"group": "ladder", "embedding": "snowflake-arctic-embed-m"}
    pre = PREFIX_GATE
    rungs = [
        {**base, "name": "L0 baseline (TF-IDF)", "kind": "baseline", "calibrated": False, "embedding": "tf-idf",
         "combine": "-", "gate": "none", "overrides": {}},
        {**base, "name": "L1 +embeddings (kNN, ungated, raw)", "kind": "system", "calibrated": False,
         "combine": "linear", "gate": "none",
         "overrides": system_overrides(DEPLOYED_EMBED, "linear", "none", knn_only, calibrated=False, gate_knobs=pre, agg=BASE_AGG)},
        {**base, "name": "L2 +gate (asymmetric, fixed floor)", "kind": "system", "calibrated": False,
         "combine": "linear", "gate": "asymmetric",
         "overrides": system_overrides(DEPLOYED_EMBED, "linear", "asymmetric", knn_only, calibrated=False, gate_knobs=pre, agg=BASE_AGG)},
        {**base, "name": "L3 +calibration (LOO percentiles, p10 floor)", "kind": "system", "calibrated": True,
         "combine": "rankfusion", "gate": "asymmetric",
         "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", knn_only, gate_knobs=pre, agg=BASE_AGG)},
        {**base, "name": "L4 +recombination (pre-fix gate)", "kind": "system", "calibrated": True,
         "combine": "rankfusion", "gate": "asymmetric",
         "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", FULL_WEIGHTS, gate_knobs=pre, agg=BASE_AGG)},
        {**base, "name": "L5 +on/off-topic gate", "kind": "system", "calibrated": True,
         "combine": "rankfusion", "gate": "asymmetric",
         "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", FULL_WEIGHTS, agg=BASE_AGG)},
    ]
    if DEPLOYED_AGG != BASE_AGG:
        kept = ", ".join(f"{k}={v}" for k, v in DEPLOYED_AGG.items() if BASE_AGG.get(k) != v)
        rungs.append({**base, "name": f"L6 +aggregation ({kept})", "kind": "system", "calibrated": True,
                      "combine": "rankfusion", "gate": "asymmetric",
                      "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", FULL_WEIGHTS)})
    rungs[-1]["name"] += " (full system)"
    return rungs


AGG_VARIANTS = {
    "A info-weighted fields": {"FIELD_AGGREGATION": "info_weighted", "NET_NEW_BLEND": "off"},
    "B-raw net-new blend (raw)": {"FIELD_AGGREGATION": "fixed_mean", "NET_NEW_BLEND": "raw"},
    "B-rank net-new blend (rank)": {"FIELD_AGGREGATION": "fixed_mean", "NET_NEW_BLEND": "rank"},
    "A+B-raw": {"FIELD_AGGREGATION": "info_weighted", "NET_NEW_BLEND": "raw"},
    "A+B-rank": {"FIELD_AGGREGATION": "info_weighted", "NET_NEW_BLEND": "rank"},
}


def aggregation_configs() -> list[dict]:
    """Pre-registered aggregation toggles, each vs the base (L5: fixed_mean, no net-new blend)."""
    rows = [{"name": "AGG base (fixed_mean, no net-new)", "group": "aggregation", "kind": "system", "calibrated": True,
             "embedding": "snowflake-arctic-embed-m", "combine": "rankfusion", "gate": "asymmetric",
             "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", FULL_WEIGHTS, agg=BASE_AGG)}]
    for name, knobs in AGG_VARIANTS.items():
        rows.append({"name": f"AGG {name}", "group": "aggregation", "kind": "system", "calibrated": True,
                     "embedding": "snowflake-arctic-embed-m", "combine": "rankfusion", "gate": "asymmetric",
                     "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", FULL_WEIGHTS,
                                                   agg={**BASE_AGG, **knobs})})
    return rows


def _spearman_vs_run(metrics, judge_run) -> float | None:
    vals = [spearman(pt["loo_scores"], [judge_run[tid][s] for s in pt["ids"]])
            for tid, pt in metrics["per_topic"].items() if tid in judge_run]
    return mean_or_none(vals)


def aggregation_decisions(results, topic_ids, judge_scores) -> dict:
    """KEEP a toggle only if it raises mean LOO Spearman vs judge (run 0) AND keeps the planted-case
    pass-rate. Robustness (reported, not used for the decision): per-topic wins and Spearman vs the
    other judge runs (shuffled item order)."""
    agg = [r for r in results if r["group"] == "aggregation"]
    base = agg[0]["metrics"]
    other_runs = {r: sc for r, sc in judge_scores.items() if r != 0 and len(sc) == len(topic_ids)}
    rows = []
    for r in agg[1:]:
        m = r["metrics"]
        topic_wins = sum((m["per_topic"][t]["spearman"] or 0) > (base["per_topic"][t]["spearman"] or 0) for t in topic_ids)
        robust = {f"run{k}": {"base": _spearman_vs_run(base, sc), "variant": _spearman_vs_run(m, sc)} for k, sc in other_runs.items()}
        better = m["spearman_mean"] is not None and base["spearman_mean"] is not None and m["spearman_mean"] > base["spearman_mean"]
        keeps = m["planted_case_pass_rate"] >= base["planted_case_pass_rate"]
        rows.append({"name": r["name"], "spearman": m["spearman_mean"], "delta_spearman":
                     None if m["spearman_mean"] is None else m["spearman_mean"] - base["spearman_mean"],
                     "pass_rate": m["planted_case_pass_rate"], "delta_pass_rate": m["planted_case_pass_rate"] - base["planted_case_pass_rate"],
                     "t2_max": m["t2_max"], "improves_spearman": bool(better), "keeps_pass_rate": bool(keeps),
                     "topic_wins": topic_wins, "robustness_other_judge_runs": robust,
                     "decision": "KEEP" if better and keeps else "REVERT"})
    return {"base": {"name": agg[0]["name"], "spearman": base["spearman_mean"], "pass_rate": base["planted_case_pass_rate"],
                     "t2_max": base["t2_max"]},
            "rule": "KEEP iff mean LOO Spearman vs judge > base AND planted-case pass-rate >= base",
            "variants": rows, "loto": loto(agg, topic_ids)}


def gatefix_configs() -> list[dict]:
    """Gate-fix variants compared during the fix (NOT selection candidates). V0 = L4, V4 = L5."""
    variants = {
        "V1 brief literal: fixed-content signal, off-topic floor, temp 0.05":
            {"RELEVANCE_GATE_SIGNAL": "fixed", "RELEVANCE_FLOOR_METHOD": "offtopic", "GATE_TEMP_MODE": "fixed", "GATE_TEMP": 0.05},
        "V2 fixed-content signal, off-topic floor, adaptive temp":
            {"RELEVANCE_GATE_SIGNAL": "fixed", "RELEVANCE_FLOOR_METHOD": "offtopic", "GATE_TEMP_MODE": "adaptive"},
        "V3 centroid signal, off-topic floor, temp 0.05":
            {"RELEVANCE_GATE_SIGNAL": "centroid", "RELEVANCE_FLOOR_METHOD": "offtopic", "GATE_TEMP_MODE": "fixed", "GATE_TEMP": 0.05},
    }
    rows = [{"name": name, "group": "gatefix", "kind": "system", "calibrated": True, "embedding": "snowflake-arctic-embed-m",
             "combine": "rankfusion", "gate": "asymmetric",
             "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric", gate_knobs=knobs, agg=BASE_AGG)}
            for name, knobs in variants.items()]
    rows.append({"name": "V5 no gate (novelty only)", "group": "gatefix", "kind": "system", "calibrated": True,
                 "embedding": "snowflake-arctic-embed-m", "combine": "rankfusion", "gate": "none",
                 "overrides": system_overrides(DEPLOYED_EMBED, "rankfusion", "none", agg=BASE_AGG)})
    return rows


def surfacing_eval(topics, fixtures) -> dict:
    """Redundancy-aware surfacing on the fixtures (deployed config; explanatory feature)."""
    per, dup_all, ord_all, hits, tot = {}, [], [], 0, 0
    for tid, t in topics.items():
        ids = [s.id for s in t.submissions]
        idx = surfacing.build_index([s.content for s in t.submissions])
        fx = fixtures.get(tid, {})
        nd = {d["id"]: d["duplicates"] for d in fx.get("near_duplicate_ids", [])}
        dup, ordn = [], []
        for i, sid in enumerate(ids):
            r = surfacing.surface(t.submissions[i].content, idx, ids, exclude_owner=i)
            if sid in nd:
                dup.append(r["net_new_ratio"])
                red = [x for x in r["sentences"] if x["label"] == "redundant"]
                tot += len(red)
                hits += sum(x["closest_id"] == nd[sid] for x in red)
            elif sid not in nd.values() and sid not in fx.get("generic_ids", []):
                ordn.append(r["net_new_ratio"])
        cases = {}
        for c in fx.get("held_out_test_cases", []):
            r = surfacing.surface(c["submission"]["content"], idx, ids)
            cases[c["case"].split("_")[0]] = {"net_new_ratio": r["net_new_ratio"],
                                              "labels": [x["label"] for x in r["sentences"]]}
        per[tid] = {"novel_max": idx.novel_max, "redundant_min": idx.redundant_min, "n_corpus_sentences": len(idx.texts),
                    "planted_dup_net_new_mean": mean_or_none(dup), "ordinary_net_new_mean": mean_or_none(ordn), "cases": cases}
        dup_all += dup
        ord_all += ordn
    auc = mean_or_none([(d < o) + 0.5 * (d == o) for d in dup_all for o in ord_all])
    return {
        "embed_model": config.SURF_EMBED_MODEL, "novel_pct": config.SURF_NOVEL_PCT, "redundant_pct": config.SURF_REDUNDANT_PCT,
        "per_topic": per,
        "planted_dup_net_new_mean": mean_or_none(dup_all), "ordinary_net_new_mean": mean_or_none(ord_all),
        "auc_dup_below_ordinary": auc,
        "redundant_sentences_pointing_to_original": hits, "redundant_sentences_in_planted_dups": tot,
        "case_net_new_mean": {k: mean_or_none([p["cases"][k]["net_new_ratio"] for p in per.values() if k in p["cases"]])
                              for k in ("T1", "T2", "T3", "T4")},
    }


def surfacing_model_sweep(topics, fixtures) -> list[dict]:
    """Sentence-embedding choice for surfacing, at the deployed thresholds."""
    rows = []
    for name, model in {"snowflake-arctic-embed-m": DEPLOYED_EMBED, **EMBEDDINGS}.items():
        with override(SURF_EMBED_MODEL=model):
            r = surfacing_eval(topics, fixtures)
        rows.append({"model": name, "auc_dup_below_ordinary": r["auc_dup_below_ordinary"],
                     "planted_dup_net_new_mean": r["planted_dup_net_new_mean"],
                     "ordinary_net_new_mean": r["ordinary_net_new_mean"],
                     "redundant_to_original": r["redundant_sentences_pointing_to_original"],
                     "redundant_in_planted": r["redundant_sentences_in_planted_dups"],
                     "T1_net_new": r["case_net_new_mean"]["T1"], "T3_net_new": r["case_net_new_mean"]["T3"]})
    return rows


def gate_fix_analysis(results, judge0) -> dict:
    """Why judge correlation moved with the gate fix: does the judge itself penalize
    weakly-relevant on-topic items (which the pre-fix gate also penalized)?"""
    by = {r["name"][:2]: r for r in results if r["group"] == "ladder"}
    pre, post = by.get("L4"), by.get("L5")
    if not pre or not post:
        return {}
    per = {}
    for tid, pt in pre["metrics"]["per_topic"].items():
        if tid not in judge0 or pt.get("gates") is None:
            continue
        j = np.array([judge0[tid][s] for s in pt["ids"]])
        g = np.array(pt["gates"])
        per[tid] = {
            "spearman_relevance_vs_judge": spearman(pt["relevance"], j),
            "judge_mean_prefix_gated": float(j[g < 0.5].mean()) if (g < 0.5).any() else None,
            "judge_mean_prefix_passed": float(j[g >= 0.5].mean()) if (g >= 0.5).any() else None,
            "n_prefix_gated": int((g < 0.5).sum()),
            "prefix_gate_ge_0.9": float((g >= 0.9).mean()),
            "postfix_gate_ge_0.9": float((np.array(post["metrics"]["per_topic"][tid]["gates"]) >= 0.9).mean()),
        }
    return {
        "per_topic": per,
        "mean_spearman_relevance_vs_judge": mean_or_none([v["spearman_relevance_vs_judge"] for v in per.values()]),
        "judge_mean_prefix_gated": mean_or_none([v["judge_mean_prefix_gated"] for v in per.values()]),
        "judge_mean_prefix_passed": mean_or_none([v["judge_mean_prefix_passed"] for v in per.values()]),
        "spearman_pre": pre["metrics"]["spearman_mean"],
        "spearman_post": post["metrics"]["spearman_mean"],
    }


def probe_configs() -> list[dict]:
    """Embedding-only probe: ungated, uncalibrated kNN novelty per model (NOT selection candidates).
    Separates embedding quality from gate hyperparameters tuned on the deployed model."""
    knn_only = {"knn": 1.0, "recombination": 0.0}
    models = {**EMBEDDINGS, "snowflake-arctic-embed-m": DEPLOYED_EMBED}
    return [{"name": f"P {ename} kNN only", "group": "probe", "kind": "system", "calibrated": False,
             "embedding": ename, "combine": "linear", "gate": "none",
             "overrides": system_overrides(emodel, "linear", "none", knn_only, calibrated=False)}
            for ename, emodel in models.items()]


def sensitivity_configs() -> list[dict]:
    """Gate sensitivity of the full system (NOT selection candidates)."""
    rows = []
    for margin, target in itertools.product([0.01, 0.02, 0.04], [0.90, 0.95, 0.99]):
        ov = system_overrides(DEPLOYED_EMBED, "rankfusion", "asymmetric") | {"FLOOR_MARGIN": margin, "GATE_TARGET_AT_ON_P5": target}
        rows.append({"name": f"S margin={margin} target={target}", "group": "sensitivity", "kind": "system", "calibrated": True,
                     "embedding": "snowflake-arctic-embed-m", "combine": "rankfusion", "gate": "asymmetric",
                     "floor_margin": margin, "gate_target": target, "overrides": ov})
    return rows


# ---------------------------------------------------------------------------
# LOTO selection
# ---------------------------------------------------------------------------
def loto(candidates: list[dict], topic_ids: list[str]) -> dict:
    folds = []
    for held in topic_ids:
        train = [t for t in topic_ids if t != held]

        def train_score(c):
            v = mean_or_none([c["metrics"]["per_topic"][t]["spearman"] for t in train])
            return -np.inf if v is None else v

        best = max(candidates, key=train_score)
        folds.append({
            "held_out": held,
            "selected": best["name"],
            "in_sample_spearman": train_score(best) if np.isfinite(train_score(best)) else None,
            "held_out_spearman": best["metrics"]["per_topic"][held]["spearman"],
        })
    for f in folds:
        f["abs_gap"] = (abs(f["in_sample_spearman"] - f["held_out_spearman"])
                        if f["in_sample_spearman"] is not None and f["held_out_spearman"] is not None else None)
    ins = mean_or_none([f["in_sample_spearman"] for f in folds])
    out = mean_or_none([f["held_out_spearman"] for f in folds])
    return {
        "folds": folds,
        "in_sample_spearman_mean": ins,
        "held_out_spearman_mean": out,
        # When every fold selects the same config, both means average the same 5 per-topic
        # values, so gap_of_means is ~0 by construction. mean_abs_fold_gap is the informative
        # (stricter) number and is what C5 uses.
        "gap_of_means": abs(ins - out) if ins is not None and out is not None else None,
        "mean_abs_fold_gap": mean_or_none([f["abs_gap"] for f in folds]),
        "max_abs_fold_gap": max((f["abs_gap"] for f in folds if f["abs_gap"] is not None), default=None),
        "distinct_configs_selected": len({f["selected"] for f in folds}),
    }


# ---------------------------------------------------------------------------
# determinism + judge variance
# ---------------------------------------------------------------------------
def determinism(topics, fixtures, judge0, runs=3) -> dict:
    cfg = ladder_configs()[-1]
    vectors = []
    for _ in range(runs):
        embed_mod.clear_memo()
        config.set_seeds()
        m = evaluate(topics, fixtures, judge0, cfg)
        vec = []
        for tid in sorted(m["per_topic"]):
            pt = m["per_topic"][tid]
            vec += pt["loo_scores"] + [pt["cases"][k] for k in sorted(pt["cases"])]
        vectors.append(np.array(vec))
    arr = np.stack(vectors)
    # Variance from deviations against run 0: exactly 0.0 when runs are bit-identical
    # (np.var on identical floats can leave ~1e-32 of rounding from the mean).
    dev = arr - arr[0]
    var = (dev**2).mean(axis=0) - dev.mean(axis=0) ** 2
    return {"runs": runs, "n_scores": int(arr.shape[1]),
            "bit_identical": bool(all(np.array_equal(arr[0], a) for a in arr[1:])),
            "max_abs_diff": float(np.abs(dev).max()),
            "system_max_var": float(var.max()), "system_mean_var": float(var.mean())}


def judge_variance(judge_scores: dict, topics) -> dict:
    runs = [r for r in range(JUDGE_RUNS) if len(judge_scores[r]) == len(topics)]
    if len(runs) < 2:
        return {"measured": False, "runs_available": runs}
    per_item_var, pair_rho = [], []
    for tid, t in topics.items():
        ids = [s.id for s in t.submissions]
        mat = np.array([[judge_scores[r][tid][s] for s in ids] for r in runs])
        per_item_var += list(mat.var(axis=0))
        for a, b in itertools.combinations(range(len(runs)), 2):
            pair_rho.append(spearman(mat[a], mat[b]))
    v = np.array(per_item_var)
    return {
        "measured": True, "runs_available": runs,
        "mean_var": float(v.mean()), "max_var": float(v.max()),
        "mean_std": float(np.sqrt(v).mean()),
        "frac_items_changed": float((v > 0).mean()),
        "run_pair_spearman_mean": mean_or_none(pair_rho),
    }


def judge_redundancy(judge0: dict, fixtures) -> float | None:
    hits = n = 0
    for tid, scores in judge0.items():
        fx = fixtures.get(tid, {})
        vals = np.array(list(scores.values()))
        for s in [d["id"] for d in fx.get("near_duplicate_ids", [])] + list(fx.get("generic_ids", [])):
            if s in scores:
                n += 1
                hits += percentile_rank(scores[s], vals) < config.VALIDATION_QUANTILE
    return hits / n if n else None


# ---------------------------------------------------------------------------
def verdict(cid: str, actual: dict) -> str:
    if any(v is None for v in actual.values() if not isinstance(v, (dict, str))):
        return "NOT MEASURED"
    c = CRITERIA[cid]
    return "PASS" if c["pass"](actual) else "PARTIAL" if c["partial"](actual) else "MISS"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-only", action="store_true", help="never call the judge API; use cached runs only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "scoring.embed", "huggingface_hub", "fastembed"):
        logging.getLogger(name).setLevel(logging.WARNING)
    config.set_seeds()
    t0 = time.time()
    topics, fixtures = load_topics(), load_fixtures()
    topic_ids = list(topics)

    log.info("judge: loading %d runs x %d topics from cache (fetching only if missing)", JUDGE_RUNS, len(topics))
    judge = load_judge(topics, cache_only=args.cache_only)
    judge0 = judge["scores"][0]

    results = []
    for cfg in ladder_configs() + aggregation_configs() + gatefix_configs() + matrix_configs() + probe_configs() + sensitivity_configs():
        s = time.time()
        cfg["metrics"] = evaluate(topics, fixtures, judge0, cfg)
        m = cfg["metrics"]
        log.info("%-52s rho=%s pass=%.1f red=%.2f T2max=%.2f (%.0fs)", cfg["name"],
                 f"{m['spearman_mean']:.3f}" if m["spearman_mean"] is not None else "n/a",
                 m["planted_case_pass_rate"], m["redundancy_bottom_q"] or 0, m["t2_max"] or 0, time.time() - s)
        results.append(cfg)

    base_m = results[0]["metrics"]
    for r in results:
        r["metrics"]["baseline_spearman_mean"] = base_m["spearman_mean"]
        r["metrics"]["baseline_spearman_pooled"] = base_m["spearman_pooled"]

    candidates = [r for r in results if r["group"] in ("matrix", "reference")]
    loto_res = loto(candidates, topic_ids)
    winner = max(candidates, key=lambda c: c["metrics"]["spearman_mean"] if c["metrics"]["spearman_mean"] is not None else -np.inf)
    full = next(r for r in results if r["group"] == "ladder" and r["name"].endswith("(full system)"))

    log.info("determinism: 3 fresh runs of the full system")
    det = determinism(topics, fixtures, judge0)
    jvar = judge_variance(judge["scores"], topics)

    fm = full["metrics"]
    actuals = {
        "C1": {"t2_max": fm["t2_max"], "t2_mean": fm["t2_mean"]},
        "C2": {"order_topics": fm["order_topics"], "of": len(topic_ids)},
        "C3": {"redundancy": fm["redundancy_bottom_q"]},
        "C4": {"system_spearman": fm["spearman_mean"], "baseline_spearman": base_m["spearman_mean"],
               "system_spearman_pooled": fm["spearman_pooled"], "baseline_spearman_pooled": base_m["spearman_pooled"]},
        "C5": {"gap": loto_res["mean_abs_fold_gap"], "gap_of_means": loto_res["gap_of_means"],
               "in_sample": loto_res["in_sample_spearman_mean"], "held_out": loto_res["held_out_spearman_mean"]},
        "C6": {"system_max_var": det["system_max_var"], "bit_identical": det["bit_identical"],
               "judge_mean_var": jvar.get("mean_var"), "judge_max_var": jvar.get("max_var"),
               "judge_measured": "yes" if jvar["measured"] else "not measured"},
    }
    criteria = {}
    for cid, c in CRITERIA.items():
        a = actuals[cid]
        core = {k: v for k, v in a.items() if not k.startswith("judge_") and k not in ("of", "bit_identical")}
        criteria[cid] = {"name": c["name"], "target": c["target"], "actual": a, "verdict": verdict(cid, core),
                         "evaluated_on": full["name"] if cid != "C5" else "LOTO over matrix+reference candidates"}

    RES = config.RESULTS_DIR
    RES.mkdir(exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "topics": {tid: len(t.submissions) for tid, t in topics.items()},
        "judge_protocol": "listwise LOO: one Gemini call per topic judges every corpus item vs the other N-1 "
                          "(backend/scoring/prompts/judge_corpus_user.md); run 0 = file order (reference), "
                          "runs 1-2 = seeded shuffles (variance only)",
        "judge_models": judge["models"],
        "judge_prompt_versions": judge["prompt_versions"],
        "judge_runs_complete": {r: len(v) for r, v in judge["scores"].items()},
        "judge_redundancy_bottom_q": judge_redundancy(judge0, fixtures),
        "runtime_s": round(time.time() - t0, 1),
        "config_snapshot": {k: getattr(config, k) for k in (
            "EMBED_MODEL", "KNN_K", "KNN_FIELD_WEIGHTS", "RECOMB_K", "BLEND_WEIGHTS", "NORMALIZE_COMPONENTS",
            "COMBINE_MODE", "RELEVANCE_GATE_SIGNAL", "RELEVANCE_FLOOR_METHOD", "FLOOR_OFFTOPIC_K", "FLOOR_MARGIN",
            "FLOOR_ON_TOPIC_PCT", "GATE_TEMP_MODE", "GATE_TARGET_AT_ON_P5", "DEDUP_PCT", "VALIDATION_QUANTILE", "SEED",
            "SURF_EMBED_MODEL", "SURF_NOVEL_PCT", "SURF_REDUNDANT_PCT", "FIELD_AGGREGATION", "GENERIC_DF_FRAC",
            "NET_NEW_BLEND", "NET_NEW_GAMMA")},
        "judge_default_model": judge_mod.DEFAULT_MODEL,
        "spearman_note": "spearman_mean = mean of per-topic Spearman (50 items each); "
                         "spearman_pooled = one Spearman over all corpus items",
    }
    strip = lambda r: {k: v for k, v in r.items() if k != "overrides"} | {"overrides": {k: v for k, v in r["overrides"].items()}}
    (RES / "experiments.json").write_text(json.dumps({
        "meta": meta, "configs": [strip(r) for r in results], "loto": loto_res,
        "winner": {"name": winner["name"], "spearman_mean": winner["metrics"]["spearman_mean"]},
        "full_system": full["name"], "determinism": det, "judge_variance": jvar,
        "gate_fix_analysis": gate_fix_analysis(results, judge0),
        "aggregation": aggregation_decisions(results, topic_ids, judge["scores"]),
        "surfacing": surfacing_eval(topics, fixtures),
        "surfacing_model_sweep": surfacing_model_sweep(topics, fixtures),
    }, indent=1, default=float))

    cols = ["name", "group", "embedding", "combine", "gate", "calibrated", "floor_margin", "gate_target", "spearman_mean", "spearman_pooled",
            "baseline_spearman_mean", "planted_case_pass_rate", "order_topics", "redundancy_bottom_q",
            "t1_mean", "t2_mean", "t2_max", "on_topic_gate_coverage", "dup_nn_hit_rate", "dup_nn_sim_pct_mean"]
    with (RES / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            row = {**r, **r["metrics"]}
            w.writerow(["" if row.get(c) is None else (round(row[c], 4) if isinstance(row[c], float) else row[c]) for c in cols])

    (RES / "criteria_report.json").write_text(json.dumps(criteria, indent=1, default=float))

    fmt = lambda x: "not measured" if x is None else f"{x:.3f}"
    print("\n" + "=" * 72)
    print(f"Winning config (max mean per-topic Spearman over all 5 topics): {winner['name']}")
    print(f"  its LOO Spearman vs judge : {fmt(winner['metrics']['spearman_mean'])}")
    print(f"Full system                 : {full['name']}")
    print(f"  LOO Spearman vs judge     : {fmt(fm['spearman_mean'])}")
    print(f"  Baseline Spearman         : {fmt(base_m['spearman_mean'])}")
    print(f"  Planted-case pass-rate    : {fm['planted_case_pass_rate']:.0%}")
    print(f"LOTO held-out / in-sample   : {fmt(loto_res['held_out_spearman_mean'])} / {fmt(loto_res['in_sample_spearman_mean'])}"
          f"  (mean per-fold gap {fmt(loto_res['mean_abs_fold_gap'])})")
    for cid, c in criteria.items():
        print(f"  {cid} {c['name']:24s} {c['verdict']}")
    print(f"results -> {RES}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
