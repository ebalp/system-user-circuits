---
description: "Compute cross-model threshold intersections and update thresholds.yaml. Use when the user wants to optimize float thresholds across multiple models, find threshold intersections, update cross-model thresholds, or ensure thresholds work for all models. Even if the user just says 'update thresholds' or 'cross-model thresholds', this skill applies."
---

# Cross-Model Threshold Optimizer

Compute optimal threshold intersections across multiple models and update `thresholds.yaml`. This command reads calibration data, computes per-model optimal ranges, finds cross-model intersections, and updates the shared threshold file.

## Inputs

You need:
1. **Model identifiers** -- comma-separated, fuzzy-matched (e.g., `8b, 70b, gemma`). If `$ARGUMENTS` is empty, default to all models in `experiment.yaml` that have results files.

Resolve fuzzy inputs by matching against available results files:
```bash
ls phase0_v2/data/results/*_results.jsonl
```

Match rules (case-insensitive substring): `8b` → `meta-llama_Llama-3.1-8B-Instruct`, `70b` → `meta-llama_Llama-3.3-70B-Instruct`, `gemma` → `google_gemma-3-27b-it`, `1b` → `meta-llama_Llama-3.2-1B-Instruct`, `qwen` → `Qwen_Qwen2.5-7B-Instruct`, `gpt` or `oss` → `openai_gpt-oss-20b`. If ambiguous, ask the user.

$ARGUMENTS

## Step 1: Run analyze.py for each model

For each resolved model, run calibration analysis to generate fresh CSVs:

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir phase0_v2/calibration/output/{safe_model_id}/ \
  --config phase0_v2/config/experiment.yaml
```

Run these sequentially (they share the conflict registry). Capture output to check for errors.

## Step 2: Run cross-model intersection analysis

All intersection computation, subset analysis, inconsistency reporting, and stale entry detection is handled by the `cross_model_thresholds.py` script. Do NOT compute intersections manually — always use this script.

First, generate the report (dry run, no changes):

```bash
uv run python -m phase0_v2.calibration.cross_model_thresholds \
  phase0_v2/calibration/output/{safe_model_id_1}/calibration_report.csv \
  phase0_v2/calibration/output/{safe_model_id_2}/calibration_report.csv \
  ... \
  --thresholds phase0_v2/config/thresholds.yaml \
  --output phase0_v2/calibration/output/cross_model_thresholds_{MMDD}_{HHMM}.md
```

The script:
- Parses optimal ranges from each model's CSV (handles disjoint regions)
- Computes full intersection across all models using Cartesian product of intervals
- Falls back to subset analysis if full intersection fails (tries N-1, N-2, ... down to 2-model subsets)
- Classifies excluded models by capability tier (strong: 70B/27B, mid: 8B/7B/20B, weak: 1B)
- Detects stale entries in thresholds.yaml (conflicts no longer in registry)
- Generates a markdown report with: main table, inconsistency analysis, stale entries, summary, proposed changes

**Status values in the report:**
- `OK` — current T is in full intersection and equals proposed midpoint
- `UPDATE` — current T is in full intersection but not at midpoint
- `OUTSIDE` — current T is outside full intersection for at least one model
- `SUBSET (N/M)` — no full intersection; best N-of-M model subset used
- `NO INTERSECTION` — even pairwise intersections fail (needs investigation)

**Inconsistency analysis** (for SUBSET/NO INTERSECTION conflicts):
- Shows which models agree and which are excluded, with capability tier
- Computes whether the proposed T is inside or outside the excluded model's range
- Assesses impact: weak-tier outliers are "expected", strong-tier disagreements need investigation

## Step 3: Review report and ask confirmation

Read the generated report and present the key findings to the user:

1. Show the summary counts (OK, UPDATE, OUTSIDE, SUBSET, NO INTERSECTION)
2. Highlight any SUBSET or NO INTERSECTION conflicts with the script's assessment
3. Show the proposed changes table (old → new threshold values)
4. Show stale entries to remove
5. For SUBSET intersections: let the user decide whether to accept (use `--include-subset`) or skip

**Ask the user for confirmation before proceeding.** Do NOT apply changes automatically.

## Step 4: Apply changes

After user confirms, run the script with `--update --confirm`:

```bash
uv run python -m phase0_v2.calibration.cross_model_thresholds \
  phase0_v2/calibration/output/{safe_model_id_1}/calibration_report.csv \
  phase0_v2/calibration/output/{safe_model_id_2}/calibration_report.csv \
  ... \
  --thresholds phase0_v2/config/thresholds.yaml \
  --update --confirm
```

Add `--include-subset` if the user approved subset intersection thresholds too.

The script updates `thresholds.yaml`:
- Sets each threshold to the intersection midpoint
- Skips NO INTERSECTION conflicts (keeps current value)
- By default skips SUBSET conflicts unless `--include-subset` is passed
- Removes stale entries
- Updates the comment header with model list and date
- Keeps entries sorted alphabetically

## Step 8: Rescore all models

For each model, run rescore (NOT `--reverify` — thresholds changed, not verifier code):

```bash
uv run python -m phase0_v2.calibration.rescore \
  phase0_v2/data/results/{safe_model_id}_results.jsonl
```

Then re-run analyze to confirm consistency:

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir phase0_v2/calibration/output/{safe_model_id}/ \
  --config phase0_v2/config/experiment.yaml
```

Check that threshold consistency shows "OK" for all models. If any model shows mismatches, warn the user.

## Step 9: Final summary

After rescoring, present:
- Which thresholds were updated and by how much
- Which stale entries were removed
- Confirmation that all models pass threshold consistency
- For SUBSET intersections: which models were excluded and the rationale
- Any NO INTERSECTION conflicts that need manual attention
- If any weak-tier models (1B) were excluded from subsets, note this is expected and not a verifier problem

## Key references

- Thresholds file: `phase0_v2/config/thresholds.yaml`
- Analysis tool: `phase0_v2/calibration/analyze.py`
- Rescore tool: `phase0_v2/calibration/rescore.py`
- Conflict registry: `phase0_v2/conflicts/registry.py`
- Config: `phase0_v2/config/experiment.yaml`

## Related commands

- **`/calibration-report`** -- Generate per-model calibration report
- **`/calibration-diagnose`** -- Explore root causes of weak conflicts
- **`/calibration-optimize`** -- Fix verifiers and run the reverify pipeline
