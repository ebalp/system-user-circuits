---
description: "Compute per-model optimal thresholds using Pareto frontier analysis (KDE density + Otsu cost + baseline BA). Use when the user wants to optimize thresholds for a specific model, find model-specific optimal thresholds, or update thresholds.yaml with per-model sections."
---

# Per-Model Threshold Optimization (Pareto)

Compute per-model thresholds via Pareto frontier analysis. For each candidate threshold T, computes three metrics:
- **d_norm**: KDE density cost (lower = closer to valley floor between peaks)
- **c_norm**: Otsu within-class variance cost (lower = better class separation)
- **BA**: balanced accuracy from baseline conditions A/B (higher = better)

Finds the Pareto frontier (no T is better on all three), then selects lexicographically: max BA, then min(d_norm + c_norm).

## Inputs

`$ARGUMENTS` is the model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`). If not provided, ask the user.

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

Default feasibility caps: `d_norm ≤ 0.05`, `c_norm ≤ 0.05`, `BA ≥ 0.90`. Override with:

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --min-ba 0.95 --max-d-norm 0.03 --max-c-norm 0.03
```

Review the output. Each float conflict gets:
- **T_sel**: selected threshold (lexicographic Pareto pick)
- **d_norm / c_norm / BA**: costs at the selected threshold
- **Change**: difference from current threshold
- **Infeasible**: no threshold meets all three caps — indicates weak model compliance or extreme distribution

## Step 3: Review and confirm

Present the summary table to the user. Highlight:
- Conflicts where threshold changes significantly (> 0.05) from current value
- Infeasible conflicts — these indicate structural problems (e.g., best BA < min_ba)
- If the user wants to relax constraints for specific conflicts, re-run with adjusted `--min-ba`, `--max-d-norm`, or `--max-c-norm`

Ask for confirmation before writing to thresholds.yaml.

## Step 4: Apply thresholds

```bash
uv run python -m phase0_v2.calibration.per_model_thresholds \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --update
```

This writes the per-model section to `thresholds.yaml` with:
- `_meta.pareto_caps`: the caps used (max_d_norm, max_c_norm, min_ba)
- Per conflict: `threshold`, `d_norm`, `c_norm`, `ba`, `distribution`, `feasible`

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
