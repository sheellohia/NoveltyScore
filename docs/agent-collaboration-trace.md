# Agent collaboration trace

This appendix to [`agents.md`](../agents.md) is concrete evidence of how the coding agent (Claude
Code) was directed on this project. It has two kinds of content:

- **Summarized requirements.** The architectural requirements supplied to the agent. These are
  **paraphrased** from the written specs, not quoted.
- **Summarized interaction traces.** What happened when those specs met the data. Numbers are taken
  from files in [`results/`](../results/), named in each section.

Nothing here is a verbatim log.

---

## 1. Representative spec-prompts (paraphrased)

### 1.1 Fixed data schema + API contract (project start)
> *Paraphrased.* "Build the skeleton, UI and **stubbed** scoring only, and keep it modular so real
> scoring drops in later. The data contract for `topic_*.json` is `{topic_id, title, fixed_content,
> …, submissions[{id, header, content, takeaway, …}]}`. Match it exactly, because a parallel
> content-generation workstream uses it. `POST /api/score` returns `{system_score, baseline_score,
> llm_score, breakdown{relevance, novelty, …}, flags{duplicate, low_relevance, gaming}, rationale}`.
> Content over 100 words → 400. `scoring/novelty.py` exposes `score(submission, topic)` marked
> `# TODO: real pipeline`. Keep the contract stable; it is the seam between phases. Run locally;
> don't push."

**Why directed this way:** fixing the schema and the response shape first let the UI, the data
workstream and the scoring research move independently. Every later phase *extended* the contract
(for example with `surfacing` and `llm_source`) instead of breaking it.

### 1.2 Pluggable scoring pipeline with self-validation
> *Paraphrased.*
> - "In `corpus_cache.py`, score every corpus item leave-one-out against the other N−1 using the
>   **same** pipeline as a new submission, and store the distributions.
> - Derive `relevance_floor` and `dedup_threshold` from percentiles; keep config values only as
>   fallbacks for degenerate distributions. `system_score` = percentile rank in the LOO distribution.
> - Add Uzzi-style recombination in a new `recombination.py`, and expose blend weights and the
>   combine mode in config.
> - At startup, log where the planted near-duplicates and generic items land, with PASS/WARN."

**Why directed this way:** each research idea became one module behind the existing `score()`
seam. The spec also demanded a built-in self-check (the startup PASS/WARN), so a feature couldn't
look finished without evidence.

### 1.3 Bug fix with a hard guardrail
> *Paraphrased.*
> - "The relevance gate fires on on-topic submissions because the floor is a percentile of an
>   all-relevant corpus. **Confirm with logging before and after.**
> - Re-derive the floor to separate on-topic from off-topic content, and widen the gate temperature.
>   Show absolute relevance in the UI, not a percentile.
> - **Guardrail:** the off-topic case T2 must *still* score < 0.2 in every topic.
> - Show that on-topic but redundant items are penalized once (by novelty), not twice.
> - Regenerate the docs from results."

**Why directed this way:** the diagnosis was stated as a hypothesis to *confirm*, and the guardrail
was non-negotiable. That is what caught that the literal fix would have broken off-topic rejection
(§2, row 1b).

### 1.4 Keep/revert experiment spec
> *Paraphrased.* "Add information-adaptive field weighting and a net-new blend, each **toggleable**.
> Keep a change only if it improves LOO Spearman vs the judge **and** keeps the planted-case
> pass-rate; revert anything that doesn't. Do **not** tune to individual hand-written inputs; the
> three hand cases are for eyeballing only."

**Why directed this way:** the decision rule was written down before any numbers existed. Section 3
shows it being applied.

---

## 2. Iteration and correction log

"Caught by" means who first noticed the problem. **Human** = the developer. **Agent check** = a
verification step the agent ran: tests, startup validation, ablation, or reviewing generated output.

| # | Issue | Caught by | How it was diagnosed | Fix |
|---|---|---|---|---|
| 1a | **Relevance gate miscalibration, iteration 1:** the gate acted as a relevance *ranking* | Agent check | The startup PASS/WARN check reported WARN on most topics. The breakdown showed that a gate temperature of 0.05 against a relevance spread of about ±0.05 made the gate grade on-topic items, and that averaging 5 neighbours diluted duplicates. | Nearest-neighbour novelty (k=1), gate temperature 0.03, and rank-normalized signals, all chosen by ablation. |
| 1b | **Relevance gate miscalibration, iteration 2:** the floor was a percentile of an all-relevant corpus, so it partly rejected genuinely on-topic content | Human | See the paragraph below this table. | Corpus-centroid relevance with a per-topic adaptive temperature. On-topic items with gate < 0.9 fell to 2–6%, T2 fell to 0.00, and the example submission's gate went from 0.918 to 1.000. |
| 2 | **Silent embedding-model override:** config changes to the embedding model had no effect | Agent check | Four different embedding models gave *identical* metrics, which is impossible. The cause was `from scoring.config import EMBED_MODEL`, which copies the value at import time. | Read the model from config at call time and re-run the ablation. The winner changed to snowflake-arctic-embed-m. |
| 3 | **A percentile was labelled "Relevance"** in the UI | Human | A UI screenshot showed "Relevance 14%" beside rationale text reading "relevance 0.62". The bar was a within-topic percentile. | `breakdown.relevance` is now the absolute cosine (0.62 → 62%). Novelty bars are labelled "(percentile vs corpus)". |
| 4 | **Fabricated stub scores:** placeholder numbers that looked real | Human | Early on the developer asked whether the scores came "from an actual run or random numbers". Later, the judge fallback still showed a seeded pseudo-random `llm_score` when rate-limited; the developer's spec required never showing a fake value. | Stubs were first clearly labelled, then removed. `llm_source` ∈ {live, cached, unavailable}; when unavailable, `llm_score` is 0.0, the UI shows "n/a", and the rationale has one clean line with no raw error JSON. |
| 5 | **False determinism failure** (C6 MISS) | Agent check | C6 said MISS, but the scores were bit-identical. `np.var` on three identical floats left about 1.2e-32 of rounding. | Variance now computed from deviations against run 0, plus a bit-identical check. C6 passes (`bit_identical: true`, variance 0.0). |
| 6 | **Generated doc contradicted its own tables** | Agent check (reviewing the generated `SOLUTION.md`) | The fixed prose called the harmonic gate and linear blend "rejected", but the tables showed the asymmetric gate and rank fusion had the lower max T2 in 0 of 6 compared pairs. | Directional sentences and win counts are now computed from `summary.csv`. The trade-off (better judge correlation, weaker off-topic rejection) is stated explicitly. |
| 7 | Leave-one-topic-out "gap = 0.000" | Agent check | When one config wins every fold, both means average the same per-topic values, so the gap is zero by construction. | Report the per-fold in-sample vs held-out gap and use it for C5. |
| 8 | Judge-cache write race | Agent check | Code review of concurrent writers: the API server, experiment runs and the background judge job. | Exclusive file lock with re-read-and-merge on every write. |

**Row 1b in detail.**
- **How it was caught.** The developer spotted in the UI that on-topic submissions sat right at the
  floor, and wrote the fix spec in §1.3.
- **What the before/after logging showed.** It confirmed the core diagnosis: 34–58% of on-topic corpus
  items had a gate below 0.9. It also corrected one detail: the example submission's gate was 0.918,
  not about 0.5.
- **The literal fix failed its own guardrail.** Relevance measured against the fixed content can't
  separate a vocabulary-sharing off-topic text, so max T2 rose to 0.58.
- **How the fix was chosen.** Four variants were compared, and only corpus-centroid relevance with an
  adaptive temperature met both goals.
- **The cost.** Judge Spearman fell from 0.562 to 0.459. The cause was measured: the judge itself marks
  weakly relevant on-topic items down (within-topic Spearman with relevance 0.263; items the old gate
  suppressed averaged judge 0.21 vs 0.53 for the rest).
- **Decision.** The trade-off was reported and accepted, not hidden.

Sources:
- row 1b: `results/gate_diagnosis_before.json`, `results/gate_diagnosis_after.json`,
  `results/experiments.json` → `gate_fix_analysis`, and `summary.csv` (group `gatefix`);
- row 5: `results/experiments.json` → `determinism`;
- rows 2 and 6: `summary.csv` (groups `probe`, `matrix`).

---

## 3. KEEP/REVERT methodology

- **The rule was fixed in code before the run.** It lives in `aggregation_decisions()` in
  `experiments/run_experiments.py` and is recorded in the results: *KEEP iff mean LOO Spearman vs
  judge > base AND planted-case pass-rate ≥ base.*
- **All variants were registered before any were scored.** That includes the scale-free `rank` form
  of the net-new blend, added because raw nearest-neighbour novelty (about 0.1–0.3) and `net_new_ratio`
  (0–1) are on different scales.

Base (fixed field weights, no net-new blend): Spearman 0.459, pass-rate 0.6.

| Variant | Spearman | Δ Spearman | Pass-rate | Δ pass-rate | Topics improved | Decision |
|---|---|---|---|---|---|---|
| A: info-weighted fields | 0.477 | +0.018 | 0.6 | +0.0 | 4/5 | **KEEP** |
| B-raw: net-new blend (raw values) | 0.304 | −0.154 | 0.4 | −0.2 | 1/5 | **REVERT** |
| B-rank: net-new blend (percentiles) | 0.441 | −0.017 | 0.6 | +0.0 | 2/5 | **REVERT** |
| A + B-raw | 0.315 | −0.144 | 0.4 | −0.2 | 1/5 | **REVERT** |
| A + B-rank | 0.456 | −0.003 | 0.6 | +0.0 | 3/5 | **REVERT** |

- **Robustness.** Reported, but not used for the decision.
  - A also beat the base against judge runs 1 and 2 (shuffled item order).
  - Leave-one-topic-out selection over these candidates picked A in all 5 folds, with a mean per-fold
    gap of 0.026.
- **The reverted blend is still in the code, switched off.** It sits behind
  `NET_NEW_BLEND = "off"`, with `raw` and `rank` available for re-testing. The deployed default is
  `FIELD_AGGREGATION = "info_weighted"`.
- **Hand-written inputs played no role.** The three named hand-test cases weren't in the repository,
  and their text was **not** reconstructed. `experiments/hand_cases.py` reads them from
  `data/fixtures/hand_cases.json` if supplied, and otherwise reports "not measured".

Source: `results/experiments.json` → `aggregation`, and `results/summary.csv` (group `aggregation`).

---

## 4. Attribution: human / agent / proposed-and-accepted

| Developer decided | Agent proposed from evidence; developer accepted | Agent produced |
|---|---|---|
| Problem framing, data contract, domain-agnostic design, local-only workflow | Embedding model (arctic-embed-m), chosen by ablation | Backend and frontend code, API validation, UI components |
| Research direction: asymmetric (MCNS) gate, leave-one-out calibration, recombination, sentence surfacing | k=1 nearest-neighbour novelty; rank fusion of signals | Scoring modules and config toggles |
| Success criteria C1–C6; keep/revert rules; leave-one-topic-out as the overfitting guard | Corpus-centroid gate signal and adaptive temperature | Experiment harness, diagnosis scripts, self-audit |
| Gemini as judge; takeaway limit; removing category; absolute relevance in the UI | Paraphrase-trained model for sentence surfacing | Test suite (81 tests) |
| Accepting the judge-correlation trade-off of the gate fix | Listwise one-call-per-topic judge protocol (to fit free-tier quota); cache locking; judge-version consistency | `SOLUTION.md` generated from result files |

Full narrative: [`agents.md`](../agents.md).
