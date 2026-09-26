# AI coding agent disclosure

**Coding agents were used to build this project.** This document explains how they were directed, what
the developer decided, what the agents produced, and where the agents had to be corrected. It is
written from the project's actual history: the specs handed to the coding agent, the repository, and
the recorded results. Nothing below is a verbatim log. Prompts are paraphrased and marked as such.

**Evidence appendix:** [`docs/agent-collaboration-trace.md`](docs/agent-collaboration-trace.md). It
has paraphrased spec-prompts, the iteration and correction log, the keep/revert decision table, and
the attribution split.

The goal was directed, verified use of AI tools, not bulk code generation. That meant:
- research before code;
- written, bounded specs with fixed contracts;
- verification against recorded metrics before accepting any change;
- explicit guardrails against tuning the system to a handful of examples.

---

## 1. Tools and division of labour

| Tool | Role in this project |
|---|---|
| **Reasoning / research assistant** (used by the developer, outside the repo) | Literature research, design discussion, and drafting the written specs pasted into the coding agent. Per the developer. This work is not visible inside the repository. |
| **Claude Code** (coding agent, in the repo) | Implementation, test scaffolding, experiment harness, diagnosis with logging and ablations, and generating `SOLUTION.md` from result files. |
| Gemini Flash models | Not a coding agent. It is a **component of the system**: the LLM judge used as a benchmark. |
| fastembed / ONNX embedding models | Not agents. Local models used by the scoring pipeline. |

The division of labour: the developer framed problems, set contracts and success criteria, and decided
what to keep. The coding agent implemented, measured, and reported back, including when the measured
results contradicted the spec.

---

## 2. Collaboration methodology

### Research first
Per the developer, novelty-detection literature was surveyed before design:
- the TREC Novelty Track (sentence-level new information);
- MMR and diversity-based ranking;
- k-NN novelty from novelty search;
- Minimal Criteria Novelty Search;
- Uzzi et al.'s atypical recombination;
- work on the originality-vs-quality trade-off in LLM output.

The specs carried that research into the build. The asymmetric relevance gate (MCNS), the Uzzi-style
cross-field recombination signal, and the TREC-style sentence surfacing all arrived as named,
research-grounded requirements. The originality-vs-quality line of work informed framing only. Nothing
in the code implements it.

### Spec-driven prompting
Each build step was one written, bounded spec with a fixed contract and an explicit verification step.
The project went through these phases, in order:

1. **Skeleton + contract.** API and data schema fixed first. Scoring stubbed behind `score()`. Run
   locally, don't push.
2. **LLM judge.** Gemini as the benchmark judge, with the prompt kept in separate files.
3. **Data reorganization.** A single source of truth for the data. Inspect the files before assuming
   their names.
4. **Scoring research features.** Leave-one-out calibration and recombination.
5. **Experiment harness.** A write-up generated *from* the results files, with no invented numbers.
6. **Bug fix with guardrails.** The relevance-gate calibration fix.
7. **Surfacing.** Redundancy-aware sentence surfacing, with a domain-agnostic wording pass.
8. **Aggregation.** Aggregation changes under an explicit keep/revert rule.

Three representative spec *kinds*, paraphrased:

> **Contract-first skeleton (phase 1).** "Build the skeleton, UI and **stubbed** scoring only. The
> response shape is `{system_score, baseline_score, llm_score, breakdown{…}, flags{…}, rationale}`.
> Keep it stable; it's the seam to the scoring phase. `scoring/novelty.py` exposes
> `score(submission, topic)` marked `# TODO: real pipeline`. Validate: content over 100 words → 400.
> Deliver a running app I can click through."

> **Research feature with self-validation (phase 4).** "Replace hardcoded thresholds with per-topic
> leave-one-out distributions. Score each corpus item against the other N−1 with the *same* pipeline.
> Derive `relevance_floor` and `dedup_threshold` from percentiles; keep config values only as
> fallbacks. Add Uzzi-style recombination in its own module. At startup, log where the planted
> near-duplicates fall and print PASS/WARN."

> **Bug fix with guardrails (phase 6).** "The relevance gate fires on on-topic submissions because the
> floor is a percentile of an all-relevant corpus. Confirm with logging **before and after**. The
> guardrail must pass: the off-topic case must still score < 0.2 in every topic. Print before/after
> for sanity items. Regenerate the docs from results."

### Verification over trust
No change was accepted on the agent's say-so. Each one was checked against recorded metrics:
- a golden fixture set: planted near-duplicates, generic items, and 4 held-out cases per topic;
- judge correlation, measured as leave-one-out Spearman against cached judge scores;
- the C1–C6 criteria, defined in code before the run;
- a pytest suite of guardrails and API-contract tests.

The main example is **the relevance-floor calibration bug**, found and fixed over two iterations:

1. **The developer found it** by noticing that, in the UI, on-topic submissions showed relevance
   sitting at the floor. They wrote a diagnosis and a fix spec with a hard guardrail.
2. **The agent measured before changing anything.** Before/after logging confirmed the core diagnosis:
   34–58% of genuinely on-topic items were partially gated. The logging also showed one detail was
   wrong: the example submission's gate was 0.918, not ~0.5.
3. **The fix as written failed the guardrail.** Relevance measured against the 100-word fixed content
   couldn't separate a vocabulary-sharing off-topic text: ai_tutors' T2 would have jumped to 0.58.
4. **The gate variants were compared and the data picked the fix.** Four variants were tested. The
   one adopted measures relevance against the corpus centroid, with a per-topic adaptive temperature.
   It was the only variant that met both goals.
5. **The cost was reported and explained.** Judge correlation fell (0.562 → 0.459). Analysis traced
   this to the judge itself marking weakly relevant on-topic items down. The trade-off went into
   `SOLUTION.md`; it wasn't hidden.

### Guardrails against overfitting
- **Keep/revert rules are written in code before the run.** For example: *KEEP iff LOO Spearman
  improves AND planted-case pass-rate holds*. The aggregation round kept information-weighted fields
  and reverted both net-new blends by this rule.
- **Selection uses leave-one-topic-out**, reporting the per-fold in-sample vs held-out gap. The agent
  pointed out that the gap between the two *means* is zero by construction when one config wins
  every fold.
- **Hand-written examples were for eyeballing only.** When the three named hand-test cases weren't
  in the repository, their text was **not** reconstructed. The harness reads them from a file and
  otherwise reports "not measured".
- **In-sample results are labelled.** Where fixtures were used both to choose settings and to report
  results, the doc says so in its limitations.

---

## 3. Representative prompt patterns

The effective specs shared one structure (annotated, paraphrased):

```
GOAL        one bounded change                 ← e.g. "fix the relevance floor"
CONTRACT    what must not change               ← "/api/score top-level keys stay; extend, don't break"
DIAGNOSIS   hypothesis to CONFIRM, not assume  ← "confirm with logging BEFORE and AFTER"
CHANGE      toggleable, config-exposed         ← "expose AGGREGATION mode: fixed_mean | info_weighted"
GUARDRAILS  must-pass checks                   ← "T2 < 0.2 in every topic"
DECISION    rule fixed in advance              ← "KEEP only if Spearman ↑ AND pass-rate holds; else revert"
EVIDENCE    regenerate outputs from results    ← "no invented numbers; missing = 'not measured'"
```

Two habits made this work in practice:
- **Small interrupts mid-task.** Examples: "add a 120-character limit on Takeaway", "remove category
  from everywhere". Each came with a reason, often a screenshot: a judge rationale criticising the
  category label rather than the idea.
- **Questions that audit the output.** Early on: "are these values coming from an actual run or
  random numbers?" That question led to the stubbed scores being clearly labelled, and later to the
  explicit `llm_source` field.

---

## 4. What the developer decided vs. what the agent produced

**The developer decided:**
- **Problem and constraints:** the problem framing and the data contract; domain-agnostic design; the
  local-only workflow.
- **Design direction:** the research synthesis behind the design; an asymmetric (MCNS) relevance gate;
  leave-one-out calibration; recombination; sentence-level surfacing.
- **Measurement rules:** the success criteria C1–C6; the keep/revert rules; leave-one-topic-out as the
  overfitting guard.
- **Product decisions:** Gemini as the judge; the takeaway limit; removing category; absolute relevance
  in the UI.
- **Interpretation:** accepting the judge-correlation trade-off of the gate fix.

**Proposed by the agent from evidence, and accepted by the developer:**
- the embedding model (arctic-embed-m), chosen by ablation;
- k=1 nearest-neighbour novelty;
- rank-fusion of signals;
- the corpus-centroid gate signal and adaptive temperature;
- a paraphrase-trained model for sentence surfacing;
- the listwise, one-call-per-topic judge protocol (needed to fit the free-tier API quota);
- the cache-locking and judge-version consistency rules.

Each of these came with the measurement that justified it, and several were checked under the
developer's pre-set rules.

**Produced by the agent:**
- **Application code:** backend and frontend code, API validation, UI components.
- **Scoring and experiments:** scoring modules and config toggles, the experiment harness, diagnosis
  scripts.
- **Tests and docs:** the pytest suite, and `SOLUTION.md` generated from the results files.

---

## 5. Honest notes: where the agent needed correction

Two were caught by the developer:
- **Random-looking stub scores.** The stub produced plausible random scores, and the developer asked
  whether they were real. Later, the judge fallback still showed a made-up number when the API was
  rate-limited. That was replaced by an explicit `unavailable` state with no fake score.
- **A percentile displayed as "Relevance".** The UI showed a within-topic percentile under the label
  "Relevance". The developer caught it from a screenshot, and it now shows absolute relevance.

The rest were caught by the agent's own verification:
- **A fix that silently didn't take effect.** In the embedding-model ablation, four different models
  gave identical metrics. The cause: `embed.py` copied the model name in at import time, so config
  overrides were ignored. Fixed, and the ablation re-run (it changed the winner).
- **A false determinism failure.** C6 first reported MISS. The scores were bit-identical; `np.var`
  left about 1e-32 of floating-point residue. The check was fixed, not the scores.
- **A generated doc that contradicted its own tables.** It called the harmonic gate and linear
  blending "rejected" while its tables showed they won on adversarial rejection. It now states the
  computed trade-off.
- **A judge-cache race.** Concurrent writers could overwrite each other's entries. Writes now use a
  file lock and a re-read-and-merge.
- **Spec assumptions that didn't match the repo.** Examples: modules a spec named that didn't exist,
  five fixture files where one was assumed, and three hand-test cases that were never saved. Each was
  reported rather than papered over.
- **A stub that outlived its phase.** `flags.gaming` stayed hard-coded to false through several
  rounds, and the "stub scoring" badge stayed in the UI header. Both were found by the final
  readiness sweep for stub markers. The flag was resolved using existing signals (a grader-directed
  text check plus the judge's verdict), and the check was tested for zero false positives on the
  250 corpus items.
- **An edit that didn't apply.** A multi-file edit aborted partway on a failed check and left one
  file unchanged. It was caught before any results were produced from it.

Operational limits, stated plainly:
- The free-tier judge quota ran out several times.
- The final benchmark uses cached verdicts from an earlier judge-prompt version, and the results
  record that.
- Several settings were chosen on the same fixtures that are reported. `SOLUTION.md` marks those
  results as in-sample.

---

## Appendix

For concrete evidence of directed use (summarized requirements, correction log with who caught
what, keep/revert decisions with numbers), see
[`docs/agent-collaboration-trace.md`](docs/agent-collaboration-trace.md).
