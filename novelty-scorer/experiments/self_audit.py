"""Submission-readiness self-audit. Every row is checked against real artifacts.

Usage (from repo root, after `make report` and with the backend running on :8000):
    backend/.venv/bin/python experiments/self_audit.py [--fresh-evidence "text"]

Row 1 (fresh checkout) can't be re-run cheaply inside this script; pass the evidence from a
fresh `git clone` -> `make setup` -> `make backend` / `make frontend` -> `make test` run.
"""

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
MAIN_PATH = [REPO / "backend" / p for p in ("main.py", "models.py", "data_loader.py")] + \
    sorted((REPO / "backend" / "scoring").glob("*.py")) + [REPO / "experiments" / "generate_doc.py",
                                                           REPO / "experiments" / "run_experiments.py"] + \
    sorted((REPO / "frontend" / "src").rglob("*.ts*"))
STUB = re.compile(r"\b(STUB|TODO|FIXME|XXX)\b|not implemented|stub scoring", re.I)


def junit() -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    path = RES / "junit.xml"
    if not path.exists():
        return out
    for tc in ET.parse(path).getroot().iter("testcase"):
        out.setdefault(tc.get("name", "").split("[")[0], []).append(not any(c.tag in ("failure", "error") for c in tc))
    return out


def passed(tests, *names) -> tuple[bool, str]:
    parts, ok = [], True
    for n in names:
        r = tests.get(n)
        ok &= bool(r) and all(r)
        parts.append(f"{n} {sum(r)}/{len(r)}" if r else f"{n} MISSING")
    return ok, "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-evidence", default="")
    args = ap.parse_args()
    T = junit()
    rows = []

    def row(name, ok, evidence):
        rows.append((name, "PASS" if ok else "FAIL", evidence))

    # 1 fresh startup
    row("fresh startup works", bool(args.fresh_evidence), args.fresh_evidence or "no fresh-checkout evidence given")

    # 2 no stubs / raw errors
    hits = []
    for p in MAIN_PATH:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if STUB.search(line) and "placeholder=" not in line:
                hits.append(f"{p.relative_to(REPO)}:{i}")
    try:
        body = json.dumps({"topic_id": "remote_work", "header": "Audit probe",
                           "content": "Remote work shifts which cities collect income tax, not just where people sit.",
                           "takeaway": "Tax bases move with workers."}).encode()
        j = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://localhost:8000/api/score", body, {"Content-Type": "application/json"}), timeout=200))
        raw = re.search(r"Traceback|RESOURCE_EXHAUSTED|\{'error'|\[STUB\]", j["rationale"])
        api_ok, api_ev = not raw, f"live API llm_source={j['llm_source']}, rationale clean={not raw}"
    except Exception as e:  # server not running
        api_ok, api_ev = False, f"API check failed: {e.__class__.__name__}"
    row("no stubs/raw errors on main path", not hits and api_ok,
        (f"stub markers: {hits}" if hits else "0 stub/TODO markers in main-path code") + f"; {api_ev}")

    # 3-6 tasks
    ok, ev = passed(T, "test_task1_fixed_content_at_most_100_words", "test_task1_every_submission_has_the_three_properties",
                    "test_content_over_100_words_rejected")
    row("Task1 three properties + fixed<=100w", ok, ev)
    sol = (REPO / "SOLUTION.md").read_text(encoding="utf-8") if (REPO / "SOLUTION.md").exists() else ""
    ok, ev = passed(T, "test_on_topic_corpus_passes_gate", "test_offtopic_T2_is_stopped_by_the_gate")
    doc_gate = "### Relevance gate" in sol and "### Bug fixes & calibration" in sol
    row("Task2 novelty+relevance gate documented", ok and doc_gate, ev + f"; SOLUTION gate sections={doc_gate}")
    exp = json.loads((RES / "experiments.json").read_text()) if (RES / "experiments.json").exists() else {}
    sizes = list(exp.get("meta", {}).get("topics", {}).values())
    ok, ev = passed(T, "test_task3_corpus_of_about_50_items_is_scored", "test_task3_new_submission_scored_against_full_corpus")
    row("Task3 ~50-item corpus scoring works", ok and bool(sizes) and all(40 <= n <= 60 for n in sizes), ev + f"; sizes={sizes}")
    ok, ev = passed(T, "test_high_novelty_low_relevance_not_rewarded", "test_high_novelty_low_relevance_not_rewarded_via_api")
    row("Task4 high-nov/low-rel NOT rewarded test PASSES", ok, ev)

    # 7 golden dataset
    dr = (REPO / "data" / "README.md").read_text(encoding="utf-8") if (REPO / "data" / "README.md").exists() else ""
    need = ["golden dataset", "data/corpus", "test_fixtures.json", "T1_novel_relevant", "T2_novel_irrelevant", "near-duplicate"]
    miss = [n for n in need if n.lower() not in dr.lower()]
    row("golden dataset named + documented", not miss, "data/README.md covers all" if not miss else f"missing: {miss}")

    # 8 tests + reports
    mk = (REPO / "Makefile").read_text() if (REPO / "Makefile").exists() else ""
    reports = [RES / "test_report.html", RES / "junit.xml", RES / "coverage" / "index.html"]
    failures = sum(1 for r in T.values() for x in r if not x)
    total = sum(len(r) for r in T.values())
    row("tests run with one command + reports generated", "\ntest:" in mk and all(p.exists() for p in reports) and total > 0
        and failures == 0, f"`make test`; {total} tests, {failures} failing; reports: "
        + ", ".join(p.name for p in reports if p.exists()))

    # 9 SOLUTION
    heads = ["## Tasks 1–4", "## 2. Engineering design", "## 3. Design rationale",
             "## 4. Success criteria & level of achievement", "Where the system falls short"]
    miss = [h for h in heads if h not in sol]
    crit = json.loads((RES / "criteria_report.json").read_text()) if (RES / "criteria_report.json").exists() else {}
    row("SOLUTION.md maps to Tasks 1-4 + criteria/achievement", not miss and len(crit) == 6,
        (f"missing: {miss}" if miss else "all sections present") + f"; criteria={ {k: v['verdict'] for k, v in crit.items()} }")

    # 10 agents.md
    am = (REPO / "agents.md").read_text(encoding="utf-8") if (REPO / "agents.md").exists() else ""
    secs = ["Tools", "Collaboration methodology", "prompt patterns", "decided vs. what the agent produced", "Honest notes"]
    miss = [x for x in secs if x.lower() not in am.lower()]
    row("agents.md present + substantive", bool(am) and not miss and len(am.split()) > 800,
        f"{len(am.split())} words; " + ("all 5 sections" if not miss else f"missing: {miss}"))

    # 11 README
    rd = (REPO / "README.md").read_text(encoding="utf-8")
    need = ["## Quickstart", "make setup", "make backend", "make frontend", "make test", "SOLUTION.md", "agents.md", "data/README.md"]
    miss = [n for n in need if n not in rd]
    row("README present + quickstart works", not miss and bool(args.fresh_evidence),
        ("quickstart + links present" if not miss else f"missing: {miss}") + "; quickstart exercised by the fresh checkout (row 1)")

    # 12 determinism + judge cache + secrets
    ok, ev = passed(T, "test_scoring_is_deterministic_across_three_fresh_runs", "test_no_secret_keys_in_repository_files",
                    "test_corpus_item_falls_back_to_cached_reference", "test_judge_failure_is_clean_and_labelled")
    cache = json.loads((RES / "judge_cache.json").read_text()) if (RES / "judge_cache.json").exists() else {}
    n_corpus = sum(1 for k in cache if k.startswith("corpus-"))
    c6 = crit.get("C6", {}).get("verdict")
    row("deterministic + judge-cache fallback + no committed secrets", ok and n_corpus >= 250 and c6 == "PASS",
        ev + f"; cached corpus verdicts={n_corpus}; C6={c6}")

    w = max(len(r[0]) for r in rows)
    print(f"\n{'check':{w}s}  result  evidence")
    print("-" * (w + 90))
    for name, res, evd in rows:
        print(f"[{'x' if res == 'PASS' else ' '}] {name:{w}s}  {res:6s}  {evd}")
    n_fail = sum(r[1] == "FAIL" for r in rows)
    print(f"\n{len(rows) - n_fail}/{len(rows)} PASS")
    (RES / "self_audit.json").write_text(json.dumps([{"check": a, "result": b, "evidence": c} for a, b, c in rows], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
