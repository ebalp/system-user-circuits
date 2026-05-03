# HuggingFace Benchmark Possibilities for Phase 0 v2

Status: exploratory — pick up later
Last updated: 2026-05-01

## TL;DR

Phase 0 v2 is benchmark-worthy. The instruction-hierarchy niche is competitive but
our dataset has differentiating features (counterbalancing, recency control,
5×5 style matrix, calibrated dual verifiers, 7 baseline models). Recommend a
two-stage release: (1) static HF dataset card first; (2) live leaderboard
Space later if interest validates.

---

## How HuggingFace benchmarks work

Two flavors:

1. **Static eval datasets.** Upload to HF Hub with a dataset card describing
   task, splits, metrics. Models are scored via `lm-eval-harness` /
   `lighteval` and can surface on leaderboards like the Open LLM Leaderboard.

2. **Live leaderboard benchmarks** (4 components):
   - Public Gradio Space — submissions UI
   - Private evaluator Space — scores submissions against held-out test set
   - Submissions dataset — records incoming predictions
   - Results dataset — stores evaluation results

   Reference: <https://huggingface.co/blog/hugging-science/building-a-benchmark-or-challenge>

For Open LLM Leaderboard inclusion you typically need: clear task formulation,
programmatic verifier, splits, and a `lighteval` task config.

Useful docs:
- <https://huggingface.co/docs/leaderboards/index>
- <https://huggingface.co/docs/hub/eval-results>
- <https://huggingface.co/docs/hub/leaderboard-data-guide>

---

## Existing instruction-hierarchy benchmarks (competitive landscape)

| Benchmark | Angle | Notes |
|-----------|-------|-------|
| **OpenAI IH-Challenge** | RL training data, simple privilege-conflict scenarios, Python grader | <https://openai.com/index/instruction-hierarchy-challenge/> |
| **ConInstruct** | Conflict detection + resolution within user instructions | <https://arxiv.org/html/2511.14342> |
| **ManyIH-Bench** | Agentic, up to 7 privilege levels, ~12 active constraints/sample | <https://arxiv.org/html/2604.09443> |
| **VerIH** | Verifiable-answer aligned/conflicting system–user pairs | — |
| **Naomibas/llm-system-prompts-benchmark** | Already on HF Hub | <https://huggingface.co/datasets/Naomibas/llm-system-prompts-benchmark> |

Background: "The Instruction Hierarchy" <https://arxiv.org/html/2404.13208v1>;
"Anatomy of an LLM Benchmark" <https://cameronrwolfe.substack.com/p/llm-bench>.

The niche is occupied but not saturated; differentiation matters more than
novelty of topic.

---

## What we have (Phase 0 v2)

- **~787K labeled records** across 7 models:
  - Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, Llama-3.3-70B-Instruct
  - Qwen2.5-7B-Instruct
  - gemma-3-27b-it, gemma-4-31B-it
  - openai/gpt-oss-20b
- **41 conflicts** across language, format, starting word, capitalization,
  emoji, disclaimer, list format, self-reference, person, word count, etc.
- **4 conditions:** A (system baseline), B (user baseline), C (hierarchy
  conflict — main test), D (recency control)
- **Counterbalanced directions** (`a_to_b` and `b_to_a`) for C and D
- **5×5 style matrix:** 5 system framings × 5 user framings
  (safety/helpfulness/etc.)
- **Programmatic dual verifiers** with per-model-calibrated thresholds
  - 8B: BA ≥ 0.95 on 35/40 conflicts (30 perfect)
  - 70B: BA ≥ 0.95 on 41/41 conflicts (40 perfect)
- Each row carries: system prompt, user prompt, response, both verifier
  scores, label (`followed_system` / `followed_user` / `followed_both` /
  `followed_neither`), confidence, expected label, full metadata.

---

## Differentiating angles vs prior work

1. **Counterbalancing (a_to_b vs b_to_a)** isolates positional/recency
   confounds — IH-Challenge and ConInstruct do not do this rigorously.
2. **Condition D (recency control)** lets us separate "follows hierarchy"
   from "follows last-mentioned instruction" — this is rare.
3. **5×5 style matrix** maps robustness to *how* instructions are phrased,
   not just *what* they say — novel axis.
4. **Calibrated dual verifiers** with continuous scores + labels +
   confidence — most IH benchmarks are binary pass/fail.
5. **7-model baseline already computed** — instant leaderboard launch with
   meaningful comparisons across model families and sizes.
6. **Atomic, diagnosable conflicts** — single-constraint-per-prompt is
   complementary to ManyIH-Bench's multi-constraint setup, not redundant.

---

## Weaknesses to address before submission

- **No held-out test set.** For a live leaderboard we need a private split;
  verifiable-format constraints are easy to overfit.
- **Public Python verifiers.** Fine for static eval; weak as an adversarial
  benchmark (a model could be trained to game them). Mitigation: rotate or
  hold back some conflicts; add adversarial evals.
- **Mostly synthetic tasks.** `wildchat_id` exists but coverage is thin.
  Consider expanding real-prompt coverage for external validity.
- **Single-conflict-per-prompt.** Frame as a feature (atomic, diagnosable)
  rather than a limitation.
- **Style matrix size** may inflate dataset without proportional signal —
  consider whether to publish full or subsample.

---

## Recommendation: two-stage release

### Stage 1 — Static HF dataset (low effort, ~1–2 days)

Publish as `IHEval-41` (or similar). Define:

- **Splits:** by condition (A / B / C / D) or by `(condition, direction)`.
- **Primary metric:** balanced accuracy of `label == expected_label` on
  Condition C.
- **Secondary metrics:**
  - Per-conflict BA
  - Recency-confound score: ΔBA between C(a_to_b) and C(b_to_a)
  - Style-robustness score: variance across the 5×5 style matrix
  - Cond-D following rate (control)
- **Baselines table:** include the 7-model results in the dataset card.
- **Artifacts:** dataset card, `lighteval`/`lm-eval-harness` task config,
  example notebook.

This alone is a credible release and validates external interest.

### Stage 2 — Live leaderboard Space (higher effort, ~1–2 weeks)

If Stage 1 gets traction:

- Hold ~10 of the 41 conflicts **private** as the test set.
- Build Gradio submission UI (model name + responses JSONL upload).
- Evaluator Space runs verifiers server-side, writes to results dataset.
- Public leaderboard ranks by primary metric; secondary metrics shown
  per-row.
- Optional: an adversarial track that rotates verifier internals.

---

## Open questions to resolve before publishing

- License — what to publish under? (Tasks, prompts, responses each have
  different considerations; gpt-oss responses may have specific terms.)
- Model-output redistribution — confirm each model's terms allow including
  generated text in a public dataset.
- WildChat licensing for the real-prompt subset.
- Naming — `IHEval-41`? `SysUserConflict-41`? Something tied to the paper
  if/when there is one.
- Whether to release Phase 1 probe directions alongside (could position the
  benchmark as "behavioral + mechanistic").
- Versioning policy — if we add conflicts later, how do we keep leaderboard
  scores comparable across dataset versions?

---

## Concrete next steps (when we pick this up)

1. Decide on Stage 1 scope (full dataset vs subsampled).
2. Resolve licensing for each model's outputs and WildChat prompts.
3. Draft the dataset card (README.md on HF).
4. Write a `lighteval` task config for Condition C balanced accuracy.
5. Pick 3–5 secondary metrics and implement scorers.
6. Decide which conflicts to hold private if pursuing Stage 2.
7. Push a v0.1 to a private HF org for review before going public.
