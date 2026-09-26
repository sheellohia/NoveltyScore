# Golden dataset

`data/corpus/*.json` plus `data/fixtures/test_fixtures.json` are the project's **golden dataset**.
It is the single source of truth for the app, the tests (`backend/tests/`) and every experiment
(`experiments/run_experiments.py`). Code reads it only through `backend/scoring/config.py`
(`CORPUS_DIR`, `FIXTURES_PATH`).

## Contents

| topic_id | corpus items | fixed content (words) | content length (words, min / median / max) |
|---|---|---|---|
| ai_tutors | 50 | 73 | 41 / 64 / 72 |
| ev_green | 50 | 69 | 67 / 80 / 84 |
| four_day_week | 50 | 68 | 53 / 61 / 66 |
| remote_work | 50 | 79 | 70 / 76 / 82 |
| teen_social_media | 50 | 74 | 62 / 84 / 90 |

In total there are 5 topics, 250 corpus submissions, and 20 held-out test cases.

## Schema

`data/corpus/{topic_id}.json`, one file per topic. The filename matches the `topic_id` inside the file,
and the loader keys topics by that internal id:

```json
{
  "topic_id": "remote_work",
  "title": "Is Remote Work Here to Stay?",
  "fixed_content": "The fixed reference text for the topic (<= 100 words).",
  "submissions": [
    { "id": "s1", "header": "one-line headline", "content": "paragraph (<= 100 words)", "takeaway": "one line" }
  ]
}
```

`data/fixtures/test_fixtures.json`, keyed by `topic_id`:

```json
{
  "remote_work": {
    "near_duplicate_ids": [ { "id": "s41", "duplicates": "s1" } ],
    "generic_ids": ["s46", "s47", "s48", "s49", "s50"],
    "held_out_test_cases": [
      { "case": "T1_novel_relevant", "expect": "high",
        "submission": { "header": "...", "content": "...", "takeaway": "..." } }
    ]
  }
}
```

## Planted cases

Each topic has the same planted structure.

| Items | What they are | What a good scorer does |
|---|---|---|
| `s1`–`s40` | ordinary submissions | the reference corpus |
| `s41`–`s45` | **near-duplicates**, each a rewrite of the listed original (`duplicates`) | score low; their nearest neighbour is the original |
| `s46`–`s50` | **generic** items: on-topic but content-free | score low on novelty, but still pass the relevance gate |

Held-out cases are not in the corpus and are scored as new submissions:

| Case | Meaning | Expected |
|---|---|---|
| `T1_novel_relevant` | a new idea, on topic | high |
| `T2_novel_irrelevant` | off-topic text unlike anything in the corpus | very low (the relevance gate must stop it) |
| `T3_near_duplicate` | a rewrite of an existing submission | low |
| `T4_generic_relevant` | on-topic but generic | low to medium |

## Provenance

- **Source.** The corpus and fixtures were LLM-generated in a separate content-generation workstream,
  according to the developer. The generation model and prompts are not recorded in this repository.
- **Checks on arrival.** When the files were added, they were checked for:
  - valid JSON with exactly the schema keys;
  - 50 submissions per topic, with unique ids;
  - content of 100 words or less, and fixed content of 100 words or less;
  - every fixture id existing in its corpus.
- **Later edits.** The five per-topic fixture files were merged into one. The unused `category` fields
  were removed later; a check confirmed nothing else in the files changed.

## Judge reference scores

- **Where they live.** LLM-judge scores for every corpus item are cached in
  `results/judge_cache.json`: 3 runs per topic, with runs 2 and 3 shuffling the item order.
- **Filling the cache.** `experiments/precache_judge.py` creates or updates them.
- **How they are used.** They measure the system's agreement with an LLM judge. They are a noisy
  benchmark, not ground truth.
