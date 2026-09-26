# Novelty Scorer

Scores how **novel** a structured user submission (header, content, takeaway) is compared with a
topic's fixed reference text and the corpus of earlier submissions. Relevance comes first: text
that is novel but off-topic is not rewarded. The system also shows, sentence by sentence, what is
new and what has already been said.

**Why.** On platforms with lots of user-generated content, the valuable contributions are the ones
that add something new. Plain similarity search rewards off-topic noise and misses reworded
repeats. This project combines calibrated embedding novelty, an asymmetric relevance gate, and
sentence-level redundancy surfacing. It is measured against a cached LLM judge and a TF-IDF
baseline. The method is domain-agnostic.

| Document | What's in it |
|---|---|
| [`SOLUTION.md`](SOLUTION.md) | Task mapping, design, rationale, success criteria and measured results (generated from `results/`) |
| [`agents.md`](agents.md) | How AI coding agents were used and directed |
| [`docs/agent-collaboration-trace.md`](docs/agent-collaboration-trace.md) | Evidence appendix: paraphrased specs, correction log, keep/revert decisions |
| [`data/README.md`](data/README.md) | The golden dataset: corpus, fixtures, planted cases |

## Quickstart

Prerequisites: Python 3.11+ and Node 18+.

```bash
make setup                 # venv + backend deps (incl. test deps) + frontend deps; creates backend/.env
make backend               # API on http://localhost:8000 (first start downloads ~0.9 GB of local models)
make frontend              # UI on http://localhost:5173  (or: make frontend PORT=5180)
```

- `backend/.env` holds `GEMINI_API_KEY` for the LLM judge. **Without a key, the app still works.**
  Corpus judge scores come from `results/judge_cache.json`, and new submissions show
  `llm_source: "unavailable"`, with the UI displaying "n/a".
- Health check: `curl http://localhost:8000/api/health` → `{"status":"ok","topics":5}`.
  API docs: http://localhost:8000/docs.

Without `make`:

```bash
cd backend && python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/uvicorn main:app --reload --reload-exclude ".venv/*" --reload-exclude ".cache/*" --port 8000
cd frontend && npm install && npm run dev
```

## Run all tests

```bash
make test                  # all tests, no LLM calls
make report                # + results/test_report.html, results/junit.xml, results/coverage/index.html
```

Submission self-audit (all checks against real artifacts; see the script for row 1):
`backend/.venv/bin/python experiments/self_audit.py --fresh-evidence "<fresh checkout result>"`.

Both `make` commands run `pytest` in `backend/`: the guardrails, the Task 4 test
(`test_high_novelty_low_relevance_not_rewarded`), surfacing, determinism, the API contract, and a
secret scan.

## API

| Method | Path                     | Returns                                              |
| ------ | ------------------------ | ---------------------------------------------------- |
| GET    | `/api/topics`            | `[{topic_id, title}]`                                |
| GET    | `/api/topics/{topic_id}` | `{topic_id, title, fixed_content}` (no submissions) |
| POST   | `/api/score`             | score object (below)                                 |
| GET    | `/api/health`            | `{status, topics}`                                   |

### Hit `/api/score`

```bash
curl -s -X POST http://localhost:8000/api/score \
  -H 'Content-Type: application/json' \
  -d '{
    "topic_id": "remote_work",
    "header": "Offices become clubhouses",
    "content": "Downtown offices will not vanish; they will turn into clubhouses for bonding and onboarding.",
    "takeaway": "The office survives as a social venue."
  }'
```

Response:

```json
{
  "system_score": 0.62,
  "baseline_score": 0.48,
  "llm_score": 0.71,
  "breakdown": { "relevance": 0.71, "novelty": 0.55, "recombination_novelty": 0.49 },
  "flags": { "duplicate": false, "low_relevance": false, "gaming": false },
  "rationale": "System: ... New information: ... LLM judge: ...",
  "surfacing": {
    "net_new_ratio": 0.75,
    "thresholds": { "novel_max_redundancy": 0.527, "redundant_min_redundancy": 0.635 },
    "sentences": [
      { "text": "...", "label": "novel", "redundancy": 0.34, "novelty_contribution": 0.66,
        "closest_id": "s23", "closest_snippet": "..." },
      { "text": "...", "label": "redundant", "redundancy": 0.79, "novelty_contribution": 0.21,
        "closest_id": "s11", "closest_snippet": "..." }
    ]
  }
}
```

All scores are 0.0–1.0, and system scoring is deterministic (same input → same scores).
- LLM judge verdicts are cached on disk in `results/judge_cache.json` per exact input, so repeats
  are instant. A live judge call takes ~5–30 s.
- `llm_source` says whether the judge score is `live`, `cached` or `unavailable`.

**Validation (all 400 with `{"detail": "..."}`):**
- `content` over 100 words (whitespace-delimited count)
- `takeaway` over 120 characters
- `header`, `takeaway`, `content` missing or blank

Unknown `topic_id` → 404.

## Data

Data lives at the repo root, separate from code, and is the single source of truth:

```
data/
  corpus/{topic_id}.json          one file per topic (50 submissions each)
  fixtures/test_fixtures.json     per-topic test fixtures, keyed by topic_id
```

Paths are defined once in `backend/scoring/config.py` (`CORPUS_DIR`, `FIXTURES_PATH`) and
resolve from the file location, so any CWD works. The loader globs `data/corpus/*.json` and
keys topics by the `topic_id` **inside** each file, so adding, removing or renaming a corpus
file needs no code change (restart the backend). Tabs are ordered by filename.

Corpus schema:

```json
{
  "topic_id": "string_slug",
  "title": "Human readable topic title",
  "fixed_content": "The fixed piece of content, <= 100 words.",
  "submissions": [
    { "id": "s1", "header": "one-liner", "content": "paragraph <= 100 words",
      "takeaway": "one-liner" }
  ]
}
```

Fixtures schema: `{ "<topic_id>": { "near_duplicate_ids": [{"id", "duplicates"}],
"generic_ids": [...], "held_out_test_cases": [{"case", "expect", "submission"}] } }`.

## How scoring works

All scoring code lives in `backend/scoring/`; tunables are in `config.py`.

| File | Role |
| --- | --- |
| `embed.py` | Local embeddings via fastembed (`snowflake/snowflake-arctic-embed-m`, ~430 MB, downloaded to `backend/.cache/` on first start). No API calls. |
| `pipeline.py` | The one scoring path, used for live submissions **and** leave-one-out corpus items: `measure()` then `finalize()`. |
| `recombination.py` | Uzzi-style atypical-combination signal: per-field top-k neighbor sets, `1 - mean(pairwise Jaccard)`. |
| `corpus_cache.py` | Per-topic cache: embeddings, LOO distributions, derived thresholds, startup fixture validation. |
| `surfacing.py` | Redundancy-aware surfacing: sentence labels (novel / partial / redundant), closest submission, `net_new_ratio`. |
| `novelty.py` | `score()`: assembles the API response and rationale. |
| `judge.py` + `prompts/` | Gemini LLM judge (benchmark). |

**Per submission:**

1. Embed header, content and takeaway separately.
2. Signals vs the topic corpus:
   - relevance = cos(content, fixed_content)
   - kNN novelty = field-weighted `1 - nearest-neighbor cosine`
   - recombination novelty
3. Each novelty signal is mapped to its percentile within the topic's LOO distribution, then
   blended: `0.706 knn + 0.294 recombination` (`BLEND_WEIGHTS`).
4. Asymmetric gate (Minimal Criteria Novelty Search):
   `raw = blended * sigmoid((gate_relevance - relevance_floor) / gate_temp)`, where
   `gate_relevance` = cos(content, topic-corpus centroid) (LOO for corpus items).
   `NOVELTY_COMBINE_MODE=harmonic|mmr` switches the combine for ablations.
5. `system_score` = percentile rank of `raw` within the topic's LOO raw distribution.
   0.8 means more novel than 80% of that topic's own corpus.

**Calibration (per topic, at startup):**
- Every corpus item is scored against the other 49 with the same pipeline.
- `relevance_floor` = min(off-topic mean + 2σ, on-topic p5 − 0.02). "Off-topic" means the other
  topics' corpora scored against this topic. In practice the on-topic cap binds (see SOLUTION.md).
- `gate_temp` is chosen per topic so an item at the on-topic p5 gets gate 0.95. On-topic text
  passes about fully, and off-topic text is stopped.
- `dedup_threshold` = 95th percentile of within-corpus nearest-neighbor content similarity.
- `config.py` values are used only as fallbacks when a distribution is degenerate.

**Response:**
- `breakdown.relevance` is the **absolute** cos(content, fixed content), 0–1. `novelty` and
  `recombination_novelty` are within-topic percentiles, labelled as such in the UI.
- `flags.duplicate` / `flags.low_relevance` come from the derived thresholds.
- `flags.gaming` is set by text addressed to the grader (for example "ignore previous
  instructions" or "rate this 100", see `scoring/flags.py`), or by the LLM judge's gaming verdict
  when one is available.
- `rationale` = system line + recombination explanation (which fields bridge which corpus
  items) + new-information summary + LLM judge line.

**Field aggregation (ablation-decided, see SOLUTION.md):**
- `FIELD_AGGREGATION = "info_weighted"` (kept): each field's weight grows with its number of
  non-generic tokens, so a short or generic header can't dilute the content.
- `NET_NEW_BLEND = "off"` (reverted; the `raw` and `rank` options remain for ablation).

**Judge availability:** the response's `llm_source` is `live`, `cached` or `unavailable`. When it
is `unavailable`, `llm_score` is 0.0 and the UI shows "n/a". API errors appear only as a short
line in the rationale.

Pre-cache all corpus judge scores (API only on cache miss):
`backend/.venv/bin/python experiments/precache_judge.py`

**Redundancy-aware surfacing (`surfacing` key; explanatory only, doesn't change `system_score`):**
- Content is split into sentences. Each sentence's redundancy is its max cosine to any sentence of
  any existing submission, using a paraphrase-trained model (`SURF_EMBED_MODEL`).
- A sentence is `redundant` at or above the corpus's own p75 of leave-one-out sentence
  redundancy, `novel` at or below the p50, and `partial` in between.
- `net_new_ratio` is the word-weighted share of novel and partial sentences. Redundant sentences
  carry the `closest_id` of the submission that already said it.
- Surfacing judges newness only. Relevance is judged per submission, so off-topic text reads as
  "new" and the UI marks it as low relevance.
- Samples: `backend/.venv/bin/python experiments/surfacing_samples.py --topic remote_work --cases T3 T1`

**Evaluate against fixtures (no LLM calls):**

```bash
cd backend && .venv/bin/python scripts/eval_fixtures.py            # or --mode harmonic|mmr
```

## Reproduce the experiments and the writeup

One command sweeps every configuration, then the doc is regenerated from the recorded results
(`SOLUTION.md` never contains a number that isn't in `results/`):

```bash
backend/.venv/bin/python experiments/run_experiments.py && backend/.venv/bin/python experiments/generate_doc.py
```

- `results/judge_cache.json`: every Gemini verdict, keyed
  `{protocol}:{topic_id}:{submission_hash}:{context_hash}`. Cached verdicts are never
  re-requested, and a fresh clone with this file makes no API calls. Add `--cache-only` to
  guarantee that.
- `results/experiments.json`, `results/summary.csv`, `results/criteria_report.json`: the
  sweep's metrics, the leave-one-topic-out selection, determinism, judge variance and the C1–C6
  verdicts.
- Reproducibility: global `SEED` in `backend/scoring/config.py`. Embeddings are local ONNX and
  bit-identical across runs.
- The corpus judge uses one call per topic (listwise leave-one-out). There are 3 runs per topic
  (the 2nd and 3rd shuffle the item order) to measure the judge's run-to-run variance.

## Tests

See [Run all tests](#run-all-tests). Gate diagnosis before/after a change:
`backend/.venv/bin/python experiments/diagnose_gate.py --label <name>`.
