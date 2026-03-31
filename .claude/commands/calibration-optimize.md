---
description: "Optimize conflict verifiers for semantic label accuracy and run the reverify-analyze-rescore pipeline. Use when the user wants to fix verifier issues, improve condition C label correctness, update thresholds, or apply the full calibration pipeline after code changes. This is the write-heavy command that modifies conflict definitions and verifier code."
---

# Calibration Optimizer

Fix conflict verifiers using cross-model audit evidence, test multiple scoring hypotheses across all models, and apply the best approach. This command **modifies conflict definitions and verifier code**.

## Philosophy

This optimization work exists to produce **semantically correct condition C labels** for downstream mechanistic interpretability (Phase 1 linear probing). The labels "followed_system" and "followed_user" become training signal for probes that identify neural directions separating instruction-following behavior. A mislabeled response — e.g., a refusal labeled "followed_user" — is a corrupted training example.

Metrics (BA, SBR, UCR, Pareto quality, contradiction rate) are **diagnostic tools** that help detect verifier issues, not optimization targets. The real target is: does each condition C label correctly reflect which instruction the model prioritized?

When evaluating hypotheses, weight "how many labels become semantically more correct" over "do aggregate metrics change." A fix that improves 0 metrics but correctly relabels 100 misclassified responses is worth applying.

**Prerequisite:** Run `/calibration-audit-cond-c` first. This command reads the audit JSONs and MD reports — it does NOT audit from scratch.

## Inputs

Parse `$ARGUMENTS` for:

1. **Conflict ID(s)** (required): comma-separated. E.g., `disclaimer_first_vs_none` or `disclaimer_first_vs_none,bullets_and_sub_bullets`
2. **`--models`** (optional): comma-separated model filter using fuzzy matching. Default: all models with audit data for the conflict.

Fuzzy match rules (case-insensitive substring):
- `8b` → `meta-llama_Llama-3.1-8B-Instruct`
- `70b` → `meta-llama_Llama-3.3-70B-Instruct`
- `gemma` → `google_gemma-3-27b-it`
- `1b` → `meta-llama_Llama-3.2-1B-Instruct`
- `qwen` → `Qwen_Qwen2.5-7B-Instruct`
- `gpt` or `oss` → `openai_gpt-oss-20b`

Results files are at `phase0_v2/data/results/{model_label}_results.jsonl`.

If multiple conflicts are specified, process them **sequentially** (complete one conflict before starting the next).

## Step 0: Setup

Get timestamp:
```bash
date +"%m%d_%H%M"
```
Store as `{MMDD_HHMM}`. Also compute `{YYYY-MM-DD HH:MM}` for report headers.

Create output and temp directories:
```bash
mkdir -p phase0_v2/calibration/output/optimize/{conflict_id}
mkdir -p phase0_v2/calibration/_optimize_tmp_{conflict_id}
```

## Step 1: Load audit evidence and structure context

For each conflict:

```python
from phase0_v2.calibration.audit_helpers import load_conflict_audits
audits = load_conflict_audits("{conflict_id}", model_labels)
```

This returns `{model_label: {"json": raw_json_dict, "report_paths": [...]}}`.

If no audit JSONs are found, tell the user to run `/calibration-audit-cond-c` first.

Read **both** the JSON data and the MD highlight reports for each model. The JSONs have structured `suggested_fixes`, error patterns, Pareto metrics, and `response_structure` (per-direction aggregate counts of bare refusal, refusal+content, meta+content, clean rates). The MD reports have qualitative analysis, response excerpts, behavioral taxonomy — complementary context.

Also check existing preprocessing and compute structure data:

```python
import yaml
with open("phase0_v2/config/conflicts.yaml") as f:
    conflict_meta = yaml.safe_load(f).get("{conflict_id}", {})
existing_preprocessing = conflict_meta.get("preprocessing", [])
```

**Structure data:** Check if audit JSONs have `response_structure`. If any model's JSON lacks this field (audits run before structure tooling was added), compute it live by running the CLI for each model:

```bash
# For each model missing response_structure:
uv run python -m phase0_v2.calibration.response_type_analysis \
  {results_file_path} --conflict {conflict_id}
```

This outputs a summary line per direction (refusal%, bare%, meta%, clean%) plus a detailed pattern table. Use these numbers in the overview and pass them to hypothesis subagents.

Show the user an overview that includes structure context:

```markdown
## {conflict_id} — Audit Overview

| Model | Severity | Error% | Top Suggested Fix |

### Response Structure (per model, condition C)

| Model | Dir | Sys Constraint | Refusal% | Bare% | Meta% | Clean% |
{from each audit JSON's response_structure field or computed live}

Existing preprocessing: {existing_preprocessing or "none"}
```

## Step 2: Synthesize hypotheses

The **main agent** (not subagents) reads all audit JSONs and MD reports and:

1. Extracts `suggested_fixes[]` from each model's JSON
2. Groups fixes by approach (same underlying code change → same hypothesis)
3. Merges model-specific details (one model may suggest extra phrases, another a wider window, another a different threshold)
4. Reviews `response_structure` data — if the audit identified stripping as needed (`stripping_needed` is not `false`), include a stripping hypothesis. But **only if the audit found evidence of misclassification** caused by refusal/metacommentary text. Do not propose stripping by default.
5. Synthesizes hypotheses — **not a rigid H1/H2/H3 structure**, but whatever distinct approaches emerge from the evidence. Could be 2, could be 5. Could be all float conversions, or all bool fixes, or a mix. The key is that each hypothesis represents a *distinct scoring approach* worth testing.

For bool conflicts where audits suggest **bool→float conversion**, include that as a hypothesis. Multiple float-based hypotheses are fine (e.g., different scorer designs).

**Structure-aware hypotheses** — when response structure (refusals, metacommentary) affects scorer accuracy, use the preprocessing module (`phase0_v2.conflicts.preprocessing`). Do NOT write ad-hoc regex. Three approaches:

- **Content-only scoring**: `extract_content(response, conflict_id)` strips refusal/meta, returns content only. Use when non-content text contaminates the scorer.
- **Structure-aware short-circuit**: use `tag_response()` / `is_bare_refusal()` to detect specific patterns (bare refusal, refusal + very short content) and return a fixed value, bypassing the scorer entirely. Use when the scorer is irrelevant for these patterns.
- **No preprocessing**: if the scorer already handles structured responses correctly, don't add complexity.

The right approach depends on evidence — the subagents will investigate which mechanism fits.

The preprocessing vocabulary for `conflicts.yaml`:
- `extract_content` — strip refusal prefixes + metacommentary + helpfulness followups via the shared `extract_content()` pipeline (returns content-only text)
- `refusal_prefix_only` — strip only the refusal prefix (custom per-conflict pattern, not the shared pipeline)
- `use_mention_stripping` — strip quoted/emphasized constraint words (use-mention distinction, e.g., "I won't use 'crucial'" → removes quoted keyword)
- `code_fence_unwrap` — extract content from markdown code blocks (```json ... ```)
- `markdown_prefix_stripping` — strip leading markdown formatting characters (#, *, _)
- `parenthetical_stripping` — strip parenthetical ASCII annotations (e.g., pinyin romanizations)

If none of these tags fit the preprocessing a fix introduces, propose a new tag name and description to the user. Only add it after user approval — then update both `conflicts.yaml` and the `PREPROCESSING_VALUES` set in `phase0_v2/config/conflict_config.py`.

Present hypothesis table to user:

```markdown
## Hypotheses for {conflict_id}

| # | Description | Estimated Impact | Complexity |
|---|-------------|------------------|------------|
| H1 | Expand PREFIX_LEN 70→200 | Fixes 70B 9%, Qwen 48% | trivial |
| H2 | Expand PREFIX_LEN + add phrases | Fixes all models | trivial |
| H3 | Bool→float conversion | Eliminates dead zones | moderate |
```

Ask for confirmation before launching subagents. The user may add, remove, or modify hypotheses.

## Step 3: Launch hypothesis-testing subagents

For each hypothesis, launch one Agent (model=opus, run_in_background=true). All launched in parallel.

**Subagent prompt:**

```
You are testing hypothesis "{HYPOTHESIS_LABEL}" for `{conflict_id}`.

**Variables:**
- CONFLICT_ID: {conflict_id}
- HYPOTHESIS_LABEL: {label}
- HYPOTHESIS_DESCRIPTION: {description}
- CODE_CHANGES: {merged code snippets from audit JSONs}
- CONFLICT_FILE: phase0_v2/conflicts/definitions/{conflict_id}.py
- RESULTS_FILES: {dict of model_label: path}
- CURRENT_THRESHOLDS: {dict of model_label: threshold or None}
- TMP_DIR: phase0_v2/calibration/_optimize_tmp_{conflict_id}
- REPORT_DIR: phase0_v2/calibration/output/optimize/{conflict_id}
- EXISTING_PREPROCESSING: {existing_preprocessing list from conflicts.yaml}

**Response structure** (per model, condition C):
{For each model, from the audit JSON's response_structure field or computed live:
  model_label:
    a_to_b (sys=constraint_type): bare=N, ref+cont=N, meta+cont=N, clean=N
    b_to_a (sys=constraint_type): bare=N, ref+cont=N, meta+cont=N, clean=N
    stripping_needed: {value from audit, or "not assessed" if audit predates structure tooling}}

**Audit findings to verify against** (per model):
{For each model, extract from the audit JSON:
  - model_label: severity, error%, condition_C root causes with error counts and signatures,
    per-direction error breakdown (a_to_b errors, b_to_a errors)}

**Read your full instructions from:** `phase0_v2/calibration/optimize_agent_instructions.md`

Read that file now and follow it exactly. It contains the phases, script templates, investigation requirements, and output format. Phase 3 requires you to verify that your hypothesis addresses each root cause listed above.
```

## Step 4: Compare results and decide

After all subagents complete, build a comparison matrix:

```markdown
## Hypothesis Comparison: {conflict_id}

### Hypothesis H1: {description}
| Model | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | C_err% | neither | both | T | d_norm | c_norm | feas |
|-------|--------|--------|--------|--------|----|----|---------|------|---|--------|--------|------|

Qualitative: {subagent's findings}
Confidence: {high/medium/low}
Concerns: {any flagged issues}
```

**Quality assessment** (per hypothesis, per model):

The primary measure of quality is **semantic label correctness** — do the condition C labels accurately reflect which instruction the model followed? Metrics are diagnostic tools, not the goal.

- **Semantic quality** (most important): from subagent's qualitative investigation — how many labels become more correct? Are near-threshold classifications sensible? Are remaining errors genuine edge cases or verifier failures? This is what matters for Phase 1 probing.
- **Label change analysis**: how many labels changed, in which direction, and were those changes correct? A hypothesis that changes 200 labels where 180 are improvements and 20 are regressions is better than one that changes 0 labels.
- **Baseline quality**: SBR/UCR/BA. A model with low baselines may be **model inability** (acceptable) or **verifier bug** (not acceptable). Use the audit evidence to distinguish.
- **Contradiction rate**: `reclassify_condition_c()` reports followed_both/followed_neither counts as a sanity check. These are architecture violations (verifier pair giving inconsistent signals), NOT a semantic error rate. 0% contradiction does NOT mean all labels are correct.
- **Pareto quality (float)**: feasible for all models is ideal. Infeasible thresholds may indicate scorer issues, but if current thresholds produce semantically correct labels, infeasibility is a secondary concern.

**Decision rules:**

1. **Clear winner**: one hypothesis dominates across all models on metrics + semantics → select automatically and tell the user
2. **Multiple good options**: pick lowest max(C_err%) across models, tiebreak by simplicity and semantic quality
3. **Tradeoffs**: different hypotheses best for different models → present to user with full breakdown (metrics + qualitative), ask them to decide
4. **All have issues**: present best options with problems highlighted, ask user to decide or suggest refinement

## Step 5: Write optimization report (initial)

Write the report to `phase0_v2/calibration/output/optimize/{conflict_id}/optimize_{MMDD_HHMM}.md` with the hypothesis comparison and decision. The pipeline results will be appended in Step 8.

```markdown
# Optimization Report: {conflict_id}

**Date:** {YYYY-MM-DD HH:MM}
**Models tested:** {model list}
**Hypotheses tested:** {count}

## Audit Evidence Summary

| Model | Severity | Error% (before) |
|-------|----------|-----------------|

## Hypotheses Tested

### Hypothesis {label}: {description}

**Code changes:** {brief description}

#### Metrics (from hypothesis testing)

| Model | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | C_err% | neither | both | T | d_norm | c_norm | feas |
|-------|--------|--------|--------|--------|----|----|---------|------|---|--------|--------|------|

#### Qualitative Findings

{subagent's qualitative assessment}

#### Confidence: {high/medium/low}

{Repeat for each hypothesis}

## Decision

**Selected:** Hypothesis {label}
**Reason:** {why this was chosen — metrics, semantic quality, tradeoffs}
```

The temp scripts in `_optimize_tmp_{conflict_id}/` are not archived — the report is the permanent record.

## Step 6: Apply the selected fix

Main agent (not subagent) implements the chosen hypothesis. The winning subagent's test script is in `_optimize_tmp_{conflict_id}/` for reference.

1. Modify `phase0_v2/conflicts/definitions/{conflict_id}.py`
2. For bool→float conversion:
   - Update class to use float scoring (add `score_system` / `score_user` methods)
   - Update `phase0_v2/config/thresholds.yaml` with the Pareto-optimal threshold
3. Update `phase0_v2/config/conflicts.yaml`: update `verifier_logic` to describe the current scoring approach (always, even if only the logic changed), and update `type`, `preprocessing`, or `architecture` if those changed
4. Run conflict-specific tests:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_keyword}
   ```
5. Fix any broken tests

## Step 7: Cross-model reverify-analyze pipeline

Three steps, in order:

**1. Reverify** — re-run verify functions with new code on ALL target models:

```bash
# For each model:
uv run python -m phase0_v2.calibration.rescore \
  phase0_v2/data/results/{model}_results.jsonl \
  phase0_v2/data/results/{model}_results.jsonl \
  --reverify --conflicts {conflict_id}
```

**2. Threshold optimization (float conflicts only)** — run per-model Pareto optimization for the specific conflict only. **Always use `--conflicts` to avoid disturbing other conflicts' thresholds.** Each model gets its own Pareto-optimal threshold (no cross-model intersection — thresholds are per-model). This updates `phase0_v2/config/thresholds.yaml`.

```bash
# For each model:
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{model}_results.jsonl \
  --update --conflicts {conflict_id}
```

**3. Rescore** — apply updated thresholds on ALL target models:

```bash
# For each model:
uv run python -m phase0_v2.calibration.rescore \
  phase0_v2/data/results/{model}_results.jsonl \
  phase0_v2/data/results/{model}_results.jsonl
```

## Step 8: Present final results

Cross-model before/after table:

```markdown
## Final Results: {conflict_id}

| Model | BA Before→After | C_err% Before→After | Threshold | Notes |
|-------|-----------------|---------------------|-----------|-------|
```

Include notes for:
- Models where baselines are low due to model inability (not verifier issue)
- Any remaining condition C errors with root cause

**Append** these pipeline results to the optimization report written in Step 5.

Report location: `phase0_v2/calibration/output/optimize/{conflict_id}/optimize_{MMDD_HHMM}.md`

## Step 9: Final audit verification

Run `/calibration-audit-cond-c` on the optimized conflict for all target models. This is the definitive confirmation that the fix works — the audit will re-examine condition C classifications with the now-committed verifier code and produce fresh severity ratings, error%, and root cause analysis.

```
/calibration-audit-cond-c {conflict_id} --models {same models used in optimization}
```

Compare the new audit results to the pre-optimization audit. All previously identified root causes should show as resolved or reduced.

This step is mandatory — do not skip it.

### If the audit finds remaining issues

If the audit flags new or unresolved problems:

1. **Present findings to the user** — show what the audit found, which models are affected, severity
2. **Discuss path of action** — propose fixes (launch new hypothesis subagents, tweak the current approach, adjust thresholds). Get user input before proceeding.
3. **If changes are made** — reverify, rescore, and re-run `/calibration-audit-cond-c` for models that had significant label changes
4. **Append a note to the optimization report** documenting the iteration (what was found, what was changed, new results)
5. **Iterate** until the audit comes back clean or the user decides remaining issues are acceptable (e.g., model inability)

This is a conversation with the user, not an autonomous loop — always discuss before acting.

### Cleanup

Once Step 9 is fully resolved (audit is clean or user accepts remaining issues), delete the temp directory:

```bash
rm -rf phase0_v2/calibration/_optimize_tmp_{conflict_id}
```

## Exclusion policy

NEVER add conflicts to `exclude_conflicts` automatically. Exclusion decisions are made by the human. The optimizer should recommend exclusions with justification but not apply them. Keeping a weak conflict in the data does no harm — removing it loses information.

## Key references

- Subagent instructions: `phase0_v2/calibration/optimize_agent_instructions.md`
- Audit data: `phase0_v2/calibration/output/condition_c_audit/{model_label}/{conflict_id}/`
- Optimization reports: `phase0_v2/calibration/output/optimize/{conflict_id}/`
- Audit helpers: `phase0_v2/calibration/audit_helpers.py` (`measure_baseline_metrics`, `reclassify_condition_c`, `run_pareto`, `load_conflict_audits`)
- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Shared utilities: `phase0_v2/conflicts/verify_utils.py`
- Rescore tool: `phase0_v2/calibration/rescore.py`
- Baseline metrics: `phase0_v2/calibration/_shared.py` (`compute_baseline_rates`, `compute_balanced_accuracy`)
- Audit tool: `phase0_v2/calibration/audit_conflict.py`
- Conflict metadata: `phase0_v2/config/conflicts.yaml`
- Thresholds config: `phase0_v2/config/thresholds.yaml`
- Per-model thresholds: `phase0_v2/calibration/per_model_thresholds.py`

## Related commands

- **`/calibration-audit-cond-c`** — Audit condition C verifier classifications (read-only, prerequisite)
- **`/calibration-per-model-thresholds`** — Per-model Pareto threshold optimization
- **`/calibration-propose`** — Design new conflict definitions
