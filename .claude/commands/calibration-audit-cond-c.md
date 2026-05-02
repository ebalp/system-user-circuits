---
description: "Audit condition C verifier classifications for semantic validity. Use when the user wants to check whether verifiers correctly capture which instruction (system vs user) a model prioritized in hierarchy conflicts, examine followed_both/followed_neither rates, or assess whether condition C labels match human judgment. Auto-applies high-confidence threshold recommendations to thresholds.yaml + rescores; verifier code itself is never modified."
---

# Condition C Verifier Audit

Audit whether conflict verifiers produce **semantically valid** classifications under condition C (hierarchy conflict). Launches subagents per conflict that produce structured JSON + highlight reports. Optionally merges multiple audit runs.

For float conflicts whose per-model Pareto pick is flagged as `ambiguous` (feasible=false, `d_norm>0.01`, `c_norm>0.01`, or `ba<max_ba`), each subagent runs **Phase 2.5 — Semantic threshold derivation** *before* sampling (Phases 4-5). It stratified-samples condition-C responses across score bands (with one-sided early-exit when cond C never crosses the boundary), labels them semantically, and derives `working_T = T_recommended`. Phases 4-5 condition on `working_T` (so diagnosis numbers and severity reflect the **post-fix** state); the audit JSON also carries `diagnosis_at_yaml_t` / `severity_at_yaml_t` for the pre-fix view.

**Phase 6.5 — Auto-apply (subagent-side):** when `recommendation_confidence == "high"` AND `T_recommended != T_pareto`, the subagent calls `apply_audit_recommendation(JSON_PATH)` after writing its outputs. This deterministic helper writes the new threshold to `thresholds.yaml` with full provenance (`source: audit_<MMDD_HHMM>`, `audit_run: <json_path>`, `previous: {full snapshot}`) and rescores the model's JSONL in-place. Medium/low confidence recommendations are NOT auto-applied — they stay in the audit JSON for operator review. Verifier code is never modified.

**Revert:** any auto-applied change can be undone with `revert_audit_recommendation(model_label, conflict_id)`, which restores the `previous` snapshot and re-rescores. Single command, no data loss.

This audit summary serves as the per-model report.

## Inputs

Parse `$ARGUMENTS` for:

1. **Models** (required): comma-separated model IDs, fuzzy-matched. Use `all` to audit all models with existing audit directories.
2. **Conflicts** (optional): `--conflicts X,Y,Z` to audit specific conflicts. If omitted, audit all registered conflicts.
3. **Batch size** (optional): `--batch-size N` (default 4). Agents per batch.

Fuzzy match rules (case-insensitive substring against available results files):
- `8b` → `meta-llama_Llama-3.1-8B-Instruct`
- `70b` → `meta-llama_Llama-3.3-70B-Instruct`
- `gemma` → `google_gemma-3-27b-it`
- `1b` → `meta-llama_Llama-3.2-1B-Instruct`
- `qwen` → `Qwen_Qwen2.5-7B-Instruct`
- `gpt` or `oss` → `openai_gpt-oss-20b`
- `all` → all models with directories in `phase0_v2/calibration/output/condition_c_audit/`

If ambiguous, ask the user.

Available results files:
```bash
ls phase0_v2/data/results/*_results.jsonl
```

### Operating modes

The **summary mode** is determined by input shape:

| Models | Conflicts | Mode | Summary location |
|--------|-----------|------|-----------------|
| 1 model | all or many | **Per-model** | `condition_c_audit/{model}/summary_*.md` |
| many models | specific (1+) | **Cross-model** | `optimize/{conflict}/audit_summary_*.md` |
| many models | all | **Per-model** (each) | `condition_c_audit/{model}/summary_*.md` (one per model) |

- **Per-model mode**: existing behavior — summary shows all conflicts for one model.
- **Cross-model mode**: post-optimization verification — summary shows one conflict across all models. Useful after `/calibration-optimize` to verify a fix works everywhere.

Per-conflict audit JSONs and reports always go to `condition_c_audit/{model}/{conflict}/` regardless of mode.

Examples:
- `8b` → audit all conflicts for 8B (per-model summary)
- `8b --conflicts language_en_zh,language_en_es` → 2 conflicts for 8B (per-model summary)
- `all --conflicts disclaimer_first_vs_none` → 1 conflict across all models (cross-model summary)
- `8b,70b,gemma --conflicts disclaimer_first_vs_none` → 1 conflict across 3 models (cross-model summary)
- `8b --batch-size 8` → larger batches (default is 4)

$ARGUMENTS

## Step 0: Prerequisite — current per-model thresholds in YAML

Before launching audit subagents, confirm each requested model has an up-to-date per-model section in `phase0_v2/config/thresholds.yaml`. The audit's Phase 2.5 (semantic-threshold derivation) only fires on float conflicts where `get_threshold_info()["ambiguous"] == True`, and that flag only appears after `/calibration-per-model-thresholds <model> --update` has been run.

```bash
uv run python -c "
import yaml, sys
from pathlib import Path
data = yaml.safe_load(Path('phase0_v2/config/thresholds.yaml').read_text())
for model_label in {model_list}:  # safe_id form, e.g. google_gemma-4-E2B-it
    section = data.get(model_label, {})
    has_ambiguous = any(
        isinstance(v, dict) and 'ambiguous' in v
        for k, v in section.items() if k != '_meta'
    )
    print(f'{model_label}: ambiguous flag present = {has_ambiguous}')
"
```

If any model returns `False`, **stop and tell the user**: they need to run `/calibration-per-model-thresholds <model>` (with `--update`) before the audit. Don't proceed with a partial Phase 2.5 — silently skipping it for missing flags wastes the audit run (Phases 4-5 would then run at the wrong threshold).

## Step 1: Resolve inputs

Resolve each model ID to its results file path. Confirm files exist.

If `all` is specified, list model directories:
```bash
ls -d phase0_v2/calibration/output/condition_c_audit/*/
```

If `--conflicts` is specified, validate each conflict ID:
```bash
uv run python -c "from phase0_v2.conflicts.registry import get_conflict_ids; print('\n'.join(get_conflict_ids()))"
```

If `--conflicts` is omitted, get all registered conflict IDs from the command above.

Derive `{model_label}` from each results filename (e.g., `meta-llama_Llama-3.1-8B-Instruct`).

Determine operating mode:
- **Cross-model**: multiple models + specific `--conflicts` → set `SUMMARY_MODE=cross-model`
- **Per-model**: single model OR no specific `--conflicts` → set `SUMMARY_MODE=per-model`

## Step 2: Setup

```bash
# Create per-conflict folders for each (model, conflict) pair
for model_label in {model_list}; do
  for conflict_id in {conflict_list}; do
    mkdir -p phase0_v2/calibration/output/condition_c_audit/${model_label}/${conflict_id}
  done
done

# For cross-model mode, also create optimize output dirs
# for conflict_id in {conflict_list}; do
#   mkdir -p phase0_v2/calibration/output/optimize/${conflict_id}
# done
```

Get timestamp:
```bash
date +"%m%d_%H%M"
```

Store as `{MMDD_HHMM}` for filenames. Also compute `{YYYY-MM-DD HH:MM}` for report headers.

## Audit flow (no --merge)

### Step 3: Launch audit agents in batches

For each **(model, conflict) pair**, launch **1 agent** (subagent_type=general-purpose, model="opus", run_in_background=true).

The total number of agents = `len(models) × len(conflicts)`. E.g., 5 models × 1 conflict = 5 agents; 1 model × 41 conflicts = 41 agents.

**Batch concurrency:** Launch up to `batch_size` agents. Wait for ALL agents in the batch to complete before launching the next batch. No backfill — full batch completes, then next batch starts.

```
pairs = [(m, c) for m in models for c in conflicts]
while pairs remaining:
  batch = next batch_size pairs
  launch all agents in batch
  wait for ALL in batch to complete
```

**Audit agent prompt:**

```
You are auditing `{conflict_id}` for condition C semantic validity.

**Variables:**
- CONFLICT_ID: {conflict_id}
- MODEL_LABEL: {model_label}
- RESULTS_FILE: {results_file_path}
- CONFLICT_FOLDER: phase0_v2/calibration/output/condition_c_audit/{model_label}/{conflict_id}/
- JSON_PATH: phase0_v2/calibration/output/condition_c_audit/{model_label}/{conflict_id}/audit_{MMDD_HHMM}.json
- REPORT_PATH: phase0_v2/calibration/output/condition_c_audit/{model_label}/{conflict_id}/audit_{MMDD_HHMM}.md
- TIMESTAMP: {YYYY-MM-DD HH:MM}

**Read your full instructions from:** `phase0_v2/calibration/audit_agent_instructions.md`

Read that file now and follow it exactly. The instructions contain the investigation phases, JSON schema, rubric rules, and self-check.
```

### Step 4: Collect results

After all agents complete, proceed to Step 5 (summary).

## Step 5: Build summary

### Per-model mode (SUMMARY_MODE=per-model)

For each model, use the `build_audit_summary` utility:

```python
from phase0_v2.calibration.audit_helpers import build_audit_summary, load_all_audits

# Generate summary with structured tables
path = build_audit_summary("{model_label}", "{MMDD_HHMM}", human_timestamp="{YYYY-MM-DD HH:MM}")

# Load parsed data for cross-cutting analysis
results = load_all_audits("{model_label}")
```

The utility writes to `phase0_v2/calibration/output/condition_c_audit/{model_label}/summary_{MMDD_HHMM}.md` with:
- Infeasible Thresholds section (if any float conflicts have `feasible: false` — lists fallback strategy and BA)
- Overview table (GREEN/YELLOW/AMBER/RED counts)
- Conflict Health table (sorted by error% descending; includes Feas column Y/N; bool conflicts show `---` for Pareto columns)
- Suggested Fixes Prioritization table (sorted by current error% descending; omitted if no fixes)
- Placeholder sections for Cross-cutting findings and Recommendations

After the utility writes the file, **append** Cross-cutting findings and Recommendations to replace the placeholders:
- Cross-cutting findings: synthesize patterns from `notes` across all conflict JSONs
- Recommendations: prioritize actions grouped by shared root cause

Suggest based on recommended action type:
- **Adjust verifier** or **Redesign scorer** → `/calibration-optimize`
- **Redesign constraint prompts** → `/calibration-propose`
- **Replace with judge** → rubric is in the audit JSON, integrate into pipeline
- **None** → no action needed

### Cross-model mode (SUMMARY_MODE=cross-model)

For each conflict, write a cross-model audit summary to `phase0_v2/calibration/output/optimize/{conflict_id}/audit_summary_{MMDD_HHMM}.md`.

Read the audit JSONs for all models:

```python
from phase0_v2.calibration.audit_helpers import load_conflict_audits
audits = load_conflict_audits("{conflict_id}", model_labels)
```

Write a summary with this structure:

```markdown
# Cross-Model Audit Summary: {conflict_id}

**Date:** {YYYY-MM-DD HH:MM}
**Models audited:** {model list}
**Context:** Post-optimization verification audit

## Overview

| Model | Severity | Error% | BA | Structural | Notes |
|-------|----------|--------|-----|------------|-------|

## Per-Model Findings

### {model_label}
- **Severity:** {GREEN/YELLOW/AMBER/RED}
- **Error rate:** {error%} ({error_count} / {total})
- **Structural errors:** {followed_both} both, {followed_neither} neither
- **Key findings:** {from audit notes}
- **Suggested fixes:** {from audit suggested_fixes, or "None"}

{Repeat for each model}

## Cross-Model Patterns

{Synthesize patterns across models — shared failure modes, model-specific quirks, verifier blind spots}

## Verdict

{Overall assessment: is the verifier performing well across all models? Any remaining issues?}

## Open Questions

{Any open_questions from individual audits that need human input}
```

This summary complements the optimization report (`optimize_*.md`) by providing semantic validation of the fix.

## Step 6: Present results

### Per-model mode

Show the user:
1. Infeasible thresholds (if any) — may indicate scorer/verifier issues or model behavior
2. **Semantic-Threshold Recommendations table** (if any) — float conflicts whose Pareto pick was ambiguous and where Phase 2.5 derived a different `working_T`. Sorted by |Δ| descending. Tell the operator that none of these are auto-applied; the audit JSON's `recommended_action.steps` already lists the exact YAML edit + rescore + re-audit commands.
3. Overview table (GREEN/YELLOW/AMBER/RED counts) — note this reflects **post-fix** severity (computed at `working_T`), so RED conflicts here are ones where even the recommended T leaves serious issues.
4. Conflict Health table — when any conflict has `severity_at_yaml_t`, the table renders dual columns: `Now (YAML T)` + `Err% now` show the pre-fix state; `Post-fix` + `Err% fix` show the state at `working_T`. Operators looking for "what's wrong right now" read the `Now` columns; operators planning the next change read `Post-fix`.
5. RED or AMBER conflicts (post-fix) with brief explanations — these are the ones that need verifier redesign or constraint reconsideration, not just a threshold tweak
6. Suggested Fixes Prioritization table (if any) — per-conflict overrides only; structural changes appear in Open Questions instead
7. Cross-cutting findings
8. Report location: `phase0_v2/calibration/output/condition_c_audit/{model_label}/`

### Cross-model mode

Show the user:
1. Cross-model overview table (severity × model)
2. Any models with AMBER or RED severity — explain why
3. Cross-model patterns
4. Verdict (is the fix working across all models?)
5. Report location: `phase0_v2/calibration/output/optimize/{conflict_id}/`

### Open questions

Check each JSON's `open_questions` array. If any conflict has open questions:

1. List each conflict with open questions
2. For each, show the question, the options, and the current default
3. Ask the user to decide — the JSON uses the agent's default until the user says otherwise
4. After the user decides, note the decision

If no conflicts need human input: "All rubric design decisions were clear from evidence — no human input needed."

### Next steps

Suggest based on results:
- Conflicts needing fixes → `/calibration-optimize`
- Constraints needing redesign → `/calibration-propose`
- Want a second audit run → re-run this command

## Key references

- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Shared utilities: `phase0_v2/conflicts/verify_utils.py`
- Conflict metadata: `phase0_v2/config/conflicts.yaml`
- Audit tool: `phase0_v2/calibration/audit_conflict.py`
- Subagent instructions: `phase0_v2/calibration/audit_agent_instructions.md`
- Per-conflict outputs: `phase0_v2/calibration/output/condition_c_audit/{model_label}/{conflict_id}/`
- Per-model summaries: `phase0_v2/calibration/output/condition_c_audit/{model_label}/summary_*.md`
- Cross-model summaries: `phase0_v2/calibration/output/optimize/{conflict_id}/audit_summary_*.md`

## Related commands

- **`/calibration-optimize`** — Fix verifiers and run the reverify pipeline
- **`/calibration-propose`** — Design new conflict definitions
- **`/calibration-per-model-thresholds`** — Per-model Pareto threshold optimization
