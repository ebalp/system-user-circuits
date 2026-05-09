---
description: "Compute per-model optimal thresholds using Pareto frontier analysis (KDE density + Otsu cost + baseline BA). Use when the user wants to optimize thresholds for a specific model, find model-specific optimal thresholds, or update thresholds.yaml with per-model sections."
---

# Per-Model Threshold Optimization (Pareto)

Compute per-model thresholds via Pareto frontier analysis. For each candidate threshold T, computes three metrics:
- **d_norm**: KDE density cost (lower = closer to valley floor between peaks)
- **c_norm**: Otsu within-class variance cost (lower = better class separation)
- **BA**: balanced accuracy from baseline conditions A/B (higher = better)

Finds the Pareto frontier (no T is better on all three), then selects lexicographically: max BA, then min(d_norm + c_norm).

Each conflict's per-model entry also carries a `max_ba` (best BA achievable on the BA(T) curve) and an `ambiguous` flag. A pick is **ambiguous** when `feasible=false`, `d_norm>0.01`, `c_norm>0.01`, OR `ba < max_ba` (any BA cost). Ambiguous picks should be queued for `/calibration-audit-cond-c`, which runs a semantic-threshold investigation (Phase 4.5) on those conflicts and records a recommended threshold in the audit JSON. This command does **not** auto-apply audit recommendations — operators apply them by re-running this command with `--conflicts <id>` (and relaxed caps if needed) or by hand-editing `thresholds.yaml`.

## Inputs

`$ARGUMENTS` is the model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`), optionally followed by `--conflicts X,Y,Z` to optimize only specific conflicts. If model ID is not provided, ask the user.

$ARGUMENTS

## Prerequisites

The model must have results JSONL at `phase0_v2/data/results/{safe_model_id}_results.jsonl` (where `/` in model ID becomes `_`).

## Step 1: Verify prerequisites

Check that the results file exists.

## Step 2: Run the optimizer (dry run)

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl
```

To optimize only specific conflicts (recommended when called from `/calibration-optimize`):

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --conflicts vocabulary_diversity,formal_vs_casual_tone
```

Default feasibility caps: `d_norm ≤ 0.02`, `c_norm ≤ 0.02`, `BA ≥ 0.95` (tightened 2026-05). Override with:

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --min-ba 0.95 --max-d-norm 0.03 --max-c-norm 0.03
```

Review the output. Each float conflict gets:
- **T_sel**: selected threshold (lexicographic Pareto pick)
- **d_norm / c_norm / BA**: costs at the selected threshold
- **maxBA**: best BA achievable on the BA(T) curve — `BA < maxBA` means the pick sacrifices baseline accuracy
- **Change**: difference from current threshold
- **Ambig**: `[AMBIG]` flag — fires on infeasible OR `d_norm>0.01` OR `c_norm>0.01` OR `BA<maxBA`. Ambiguous picks should be queued for `/calibration-audit-cond-c`.
- **Infeasible**: no threshold meets all three caps — indicates weak model compliance or extreme distribution

## Step 3: Review and confirm

Present the summary table to the user. Highlight:
- Conflicts where threshold changes significantly (> 0.05) from current value
- Infeasible conflicts — these indicate structural problems (e.g., best BA < min_ba)
- Ambiguous (but feasible) picks — these may pass the caps but still don't capture verifier intent; the audit's Phase 4.5 may recommend a different threshold
- If the user wants to relax constraints for specific conflicts, re-run with adjusted `--min-ba`, `--max-d-norm`, or `--max-c-norm`

Ask for confirmation before writing to thresholds.yaml.

## Step 4: Apply thresholds

**Always `--update` before launching `/calibration-audit-cond-c`.** Phase 4.5 (semantic-threshold investigation) reads `info["ambiguous"]` from `get_threshold_info()`, which only returns `True` once the flag is in `thresholds.yaml`. Skip this step and the audit subagents will see `ambiguous: false` for every conflict and silently skip Phase 4.5 — there will be no semantic-threshold recommendations in the report.

This is true even if the audit is the next thing you'll do — do not "wait until after the audit to write the YAML." The thresholds being placeholders is fine; what matters is that the `ambiguous` flag travels through YAML so the audit triggers correctly. After the audit, you re-run `--update --conflicts <ids>` to fold in any recommended thresholds.

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --update [--conflicts X,Y,Z]
```

When `--conflicts` is specified, only those conflicts are updated in the per-model section — other conflicts are preserved as-is, and `_meta.pareto_caps` is preserved (so a partial update can't silently rewrite caps used to produce the rest of the section). Without `--conflicts`, all float conflicts are re-optimized and `_meta.pareto_caps` is refreshed to reflect this run.

This writes the per-model section to `thresholds.yaml` with:
- `_meta.pareto_caps`: the caps used (max_d_norm, max_c_norm, min_ba)
- Per conflict: `threshold`, `d_norm`, `c_norm`, `ba`, `max_ba`, `distribution`, `feasible`, `ambiguous`

After writing, the script shows **suggested default updates** — the median threshold across all per-model sections. Present these to the user. Only apply accepted changes to the `default:` section manually.

## Step 5: Rescore with new thresholds

```bash
uv run python -m phase0_v2.calibration.rescore \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --model {model_id}
```

Only needed for this model. Other models using the `default:` section need rescoring only if defaults were changed.

## Step 6: Generate model report

```bash
uv run python phase0_v2/generate_model_report.py --model {model_id}
```

Open the report and verify the Score vs Threshold plot reflects the new thresholds.

## Related commands

- **`/calibration-audit-cond-c`** — Audit condition C verifier classifications (includes baseline semantic integrity check)
- **`/calibration-optimize`** — Fix verifier code (different from threshold optimization)
- **`/calibration-propose`** — Design new conflict definitions
