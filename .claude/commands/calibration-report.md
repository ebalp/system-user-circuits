---
description: "Generate a calibration report for Phase 0 v2 experiment results. Use this skill when the user asks to: generate a calibration report, run calibration analysis, analyze experiment results for a model, check conflict quality, assess verifier performance, or anything related to calibrating conflicts for a specific model. Even if the user just says 'calibrate' or 'analyze results', this skill applies."
---

# Calibration Report Generator

Generate a structured markdown calibration report from Phase 0 v2 experiment results. This is a **read-only** command -- it analyzes data and produces a report but does NOT modify any conflict definitions or verifier code.

## Inputs

You need:
1. **Model ID** -- e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** -- at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.

Derived paths (all under a per-model directory):
- **Output directory**: `phase0_v2/calibration/output/{safe_model_id}/` -- analysis CSV and anomalies JSONL go here
- **Report file**: `phase0_v2/calibration/output/{safe_model_id}/calibration_report_{MMDD}_{HHMM}.md` (timestamp = current time)

Create the output directory if it doesn't exist: `mkdir -p phase0_v2/calibration/output/{safe_model_id}`

If `$ARGUMENTS` is provided, treat it as the model ID.

## Step 1: Run calibration analysis

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir phase0_v2/calibration/output/{safe_model_id}/ \
  --config phase0_v2/config/experiment.yaml \
  --model-config {model_id}
```

Capture the full console output. It contains:
- Dataset completeness check (COMPLETE or gap list)
- Threshold consistency check (OK or mismatch list)
- Constraint legend
- Baseline rates table
- Float score calibration table (with opt_range column)
- Anomaly summary (by reason and by conflict)

**If threshold consistency shows mismatches**, warn the user: "Stored results were scored with different thresholds. Run `rescore` before generating the report, or the report will reflect stale labels." Ask whether to proceed or stop.

## Step 1b: Check for previous reports

Look for existing reports in the model's output directory:

```bash
ls -t phase0_v2/calibration/output/{safe_model_id}/calibration_report_*.md 2>/dev/null || true
```

If a previous report exists, read it and extract the per-conflict BA and tier assignments from its "Complete conflict status" section. Store these as the baseline for comparison in Step 5 (Section 8: Changes since last report).

If no previous report exists, skip the comparison section.

## Step 2: Parse analysis outputs

Read the two output files:

### calibration_report.csv

Each conflict has up to 4 rows: (constraint=a, role=system), (constraint=a, role=user), (constraint=b, role=system), (constraint=b, role=user). Non-invertible conflicts have only 2 rows (constraint=a only).

Extract per conflict:
- **SBR(a)**: `baseline_rate` where `constraint=a, role=system`
- **UCR(a)**: `baseline_rate` where `constraint=a, role=user`
- **SBR(b)**: `baseline_rate` where `constraint=b, role=system`
- **UCR(b)**: `baseline_rate` where `constraint=b, role=user`
- **BA**: `balanced_accuracy` (same across all 4 rows of a conflict)
- **Type**: `bool` if `trying_mean` is empty, `float` otherwise
- **Threshold**: `threshold` column (for float conflicts)
- **Optimal range**: `optimal_threshold_low`, `optimal_threshold_high`
- **Optimal midpoint**: `optimal_threshold` column (midpoint of optimal range — this is the recommended threshold)

### anomalies.jsonl

Count records by `(conflict_id, anomaly_reason)`. Reasons: `followed_both`, `cond_A_followed_user`, `cond_B_followed_system`. Note: the field name in the JSONL is `anomaly_reason`, not `reason`.

## Step 3: Assign tiers

For each conflict, check baselines and BA:

- **Tier 1**: `min(SBR(a), UCR(a), SBR(b), UCR(b)) >= 0.8` AND `BA >= 0.8`
  - For non-invertible conflicts (no b rows): `min(SBR(a), UCR(a)) >= 0.8`
- **Tier 2**: `BA >= 0.7` but at least one baseline < 0.8
- **Tier 3**: `BA < 0.7` OR `min(baseline) < 0.3`
- **Excluded**: Conflicts in the model's `exclude_conflicts` list in `experiment.yaml`, or conflicts absent from the data entirely

Also check the experiment config for excluded conflicts:
```bash
uv run python -c "
from phase0_v2.src.config import load_config
c = load_config('phase0_v2/config/experiment.yaml')
for m in c.models:
    if m.id == '{model_id}':
        print(m.exclude_conflicts)
"
```

And check for non-invertible conflicts:
```bash
uv run python -c "
from phase0_v2.conflicts.registry import get_all_conflicts
for c in get_all_conflicts():
    if not c.supports_counterbalancing():
        print(c.conflict_id)
"
```

## Step 4: Extract scorer descriptions

Run this command to extract all conflict descriptions at once:

```bash
awk '/# <description>/,/# <\/description>/{print FILENAME": "$0}' phase0_v2/conflicts/definitions/*.py
```

Parse the output to build a map of conflict_id -> (type, constraint_a, constraint_b, scorer, explored). The conflict_id is the filename stem (e.g., `capitalization_all_caps.py` -> `capitalization_all_caps`).

Do NOT create or modify description blocks in this command. If a conflict is missing a description, note it in the report but leave the file untouched.

## Step 5: Write the report

Write to `phase0_v2/calibration/output/{safe_model_id}/calibration_report_{MMDD}_{HHMM}.md` following this exact structure:

### Section 1: Header and dataset status

```markdown
# Calibration Report: {model_id}

**Date**: {today's date}
**Records**: {N} total, {errors} errors
**Conflicts**: {n_in_data} in data, {n_excluded} excluded

## Dataset status

{Copy the completeness check output from analyze.
If COMPLETE: "All conflicts have expected record counts."
If INCOMPLETE: list the gaps as a table.}

## Threshold consistency

{Copy the threshold consistency output from analyze.
If OK: "All float-scored records match current thresholds."
If mismatches: list them and note "Run rescore before using these results."}
```

### Section 2: Constraint legend

```markdown
## Constraint legend

| Conflict | Constraint a | Constraint b | Type | Scorer | Thresh | BA |
|----------|-------------|-------------|------|--------|--------|-----|
```

Sort by BA descending. For threshold: show numeric value for float, `--` for bool.
For non-invertible conflicts, put "(non-invertible)" in Constraint b.

### Section 3: Complete conflict status by tier

```markdown
## Complete conflict status

### Tier 1: Reliable (min baseline >= 0.80, BA >= 0.80)

| Conflict | Thresh | Type | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | Anomalies |
|----------|--------|------|--------|--------|--------|--------|----|-----------|

{N} conflicts.

### Tier 2: Usable with caveats

| Conflict | Thresh | Type | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | Issue |
|----------|--------|------|--------|--------|--------|--------|----|-------|

{N} conflicts.

### Tier 3: Weak (structural problems)

| Conflict | Thresh | Type | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | Root cause |
|----------|--------|------|--------|--------|--------|--------|----|------------|

{N} conflicts.

### Excluded

| Conflict | Reason |
|----------|--------|
```

For Tier 2 "Issue" column: briefly describe which baseline is weak (e.g., "SBR(a)=0.60").
For Tier 3 "Root cause": describe the structural problem (e.g., "Constraint b near-zero: SBR=0.06, UCR=0.08").
For Excluded: state reason (per-model exclusion, non-invertible, etc.).

Format baselines to 2 decimal places, BA to 3.

### Section 4: Float threshold results

```markdown
## Float threshold results

| Conflict | Current T | Optimal mid | Optimal range | BA | Needs change? |
|----------|-----------|-------------|---------------|----|---------------|
```

The **optimal midpoint** (`optimal_threshold` column in the CSV) is the recommended threshold — it maximizes margin from the range edges. "Needs change?" = "Yes" if current threshold differs from the optimal midpoint (not just outside the range). Format the optimal range as `[T_low, T_high]` or just `T` if T_low == T_high.

### Section 5: Anomaly summary

```markdown
## Anomaly summary

| Category | Count | Top contributors |
|----------|-------|------------------|
| followed_both | {n} | {top 3 conflicts with counts} |
| Cond A followed_user | {n} | {top 3} |
| Cond B followed_system | {n} | {top 3} |

Total anomalies: {total}
```

### Section 6: Recommended exclusions

Apply these drop criteria:
- BA < 0.70
- min(baseline) < 0.70

```markdown
## Recommended exclusions

### Drop (both criteria met)

| Conflict | BA | min(BL) | Anomalies | Reason |
|----------|----|---------|-----------|--------|

### Consider dropping (one criterion met)

| Conflict | BA | min(BL) | Anomalies | Reason |
|----------|----|---------|-----------|--------|
```

Write a specific justification per conflict citing the metrics that trigger the recommendation. "Drop" = both BA < 0.70 AND min(baseline) < 0.70. "Consider dropping" = exactly one criterion met.

**IMPORTANT**: These are recommendations only. Do NOT auto-apply exclusions. Exclusion decisions are made by the human. Keeping a weak conflict in the data does no harm; removing it loses visibility.

### Section 7: Suggested improvements

```markdown
## Suggested improvements

{Note any conflicts where:
- Threshold needs updating (current outside optimal range) -- reference the float threshold table
- One baseline side is significantly weaker than the other (>0.3 gap between sides)
- High anomaly count relative to baseline records (>10% of baseline records are anomalous)
}
```

This section is analytical commentary, not a fixed table. Be specific about what to fix and how.

### Section 8: Changes since last report (if previous report exists)

If a previous report was found in Step 1b, include a comparison section:

```markdown
## Changes since last report

**Previous report:** {previous_report_filename}

### BA changes

| Conflict | Previous BA | Current BA | Change | Notes |
|----------|-----------|------------|--------|-------|
```

Only list conflicts where BA changed, tier changed, or exclusion status changed. Sort by largest improvement first.

Summarize:
- Number of conflicts with improved BA, degraded BA, unchanged
- Tier migrations (e.g., "2 conflicts promoted from Tier 2 to Tier 1")
- New exclusions or removals from exclusion list
- Total anomaly count change

If no previous report exists, omit this section entirely.

### Section 9: Diagnostic summary (placeholder)

Always include this section as a placeholder at the end of the report:

```markdown
## Diagnostic summary

_No diagnostics have been run for this report yet. Run `/calibration-diagnose` to populate this section._
```

**After diagnostics are run:** When `/calibration-diagnose` completes in the same conversation, append its compiled diagnostic table and root-cause groupings to this section of the most recent report, replacing the placeholder text. Use the Edit tool to update the report file in place. The appended content should include:

- The diagnostic results summary table (Conflict | BA | Root Cause | Est. BA After | Confidence | Proposed Action)
- The root-cause groupings (model inability, verifier issues, constraint design, threshold issues)
- Links to the individual diagnostic report files in `phase0_v2/calibration/output/{safe_model_id}/diagnosis/`

## Step 6: Present summary and suggest next steps

After writing the report, summarize:
- Conflicts by tier (N Tier 1, N Tier 2, N Tier 3, N Excluded)
- Conflicts recommended for exclusion
- Thresholds that need updating
- Any suggested improvements

Then suggest next steps:
- **`/calibration-diagnose`** -- to explore root causes of weak conflicts without modifying code
- **`/calibration-optimize`** -- to fix verifiers and run the reverify-analyze-rescore pipeline
- **`/calibration-propose`** -- to design and implement new conflict definitions

## Step 7 (optional): Generate PDF

Do NOT generate the PDF automatically. After presenting the summary in Step 6, mention that you can generate a PDF if they want one. Only proceed if the user explicitly asks.

The report contains backslash sequences (`\n`, `\d`) that LaTeX interprets as control sequences, so escape them first with `sed`. Use a LaTeX header to allow line breaks at underscores in table cells (conflict IDs are long).

```bash
sed 's/\\n/\\\\n/g; s/\\d/\\\\d/g' {report_path} > /tmp/re.md && \
pandoc /tmp/re.md \
  -o {report_path%.md}.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=0.75in \
  -V fontsize=9pt \
  -H <(printf '\\let\\oldtextunderscore\\_\n\\renewcommand{\\_}{\\oldtextunderscore\\hspace{0pt}}\n') && \
rm /tmp/re.md
```

The `\hspace{0pt}` after each underscore tells LaTeX it can wrap long identifiers like `no_consecutive_first_letter` across lines instead of overflowing into adjacent columns.

## Key references

- Process doc: `phase0_v2/calibration/iterative_calibration_process.md`
- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Conflict registry: `phase0_v2/conflicts/registry.py`
- Analysis tool: `phase0_v2/calibration/analyze.py`
- Rescore tool: `phase0_v2/calibration/rescore.py`
- Test harness: `phase0_v2/calibration/test_verifier.py`
- Config: `phase0_v2/config/experiment.yaml`

## Related commands

- **`/calibration-diagnose`** -- Explore root causes of weak conflicts (read-only, no code changes)
- **`/calibration-optimize`** -- Fix verifiers and run the reverify-analyze-rescore pipeline
- **`/calibration-propose`** -- Design and implement new conflict definitions
