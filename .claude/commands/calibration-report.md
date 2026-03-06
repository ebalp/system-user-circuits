---
description: "Generate a calibration report for Phase 0 v2 experiment results. Use this skill when the user asks to: generate a calibration report, run calibration analysis, analyze experiment results for a model, check conflict quality, assess verifier performance, or anything related to calibrating conflicts for a specific model. Even if the user just says 'calibrate' or 'analyze results', this skill applies."
---

# Calibration Report Generator

Generate a structured markdown calibration report from Phase 0 v2 experiment results. This skill runs the analysis tool, parses the outputs, and produces a report following the spec in `phase0_v2/calibration/iterative_calibration_process.md` section 4.

## Inputs

You need:
1. **Model ID** — e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** — at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.
3. **Output directory** — default `phase0_v2/calibration/output/`. Ask user if they want a different one.
4. **Report filename** — ask user for a short model name (e.g., "llama_8b") to use in `verifier_calibration_report_{name}.md`, or propose one.

If `$ARGUMENTS` is provided, treat it as the model ID.

## Step 1: Run calibration analysis

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir {output_dir} \
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

### anomalies.jsonl

Count records by `(conflict_id, reason)`. Reasons: `followed_both`, `cond_A_followed_user`, `cond_B_followed_system`.

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

Parse the output to build a map of conflict_id → (type, constraint_a, constraint_b, scorer, explored). The conflict_id is the filename stem (e.g., `capitalization_all_caps.py` → `capitalization_all_caps`).

If a conflict has no `<description>` block, read its definition file and initialize one. The block goes after the module docstring, before imports:

```python
"""conflict_id: One-line summary."""

# If you modify the scoring logic, update the description block below
# and set explored to 'no'.
# <description>
# type: bool or float
# constraint_a: Short phrase from system_template
# constraint_b: Short phrase from user_template
# scorer: What the verify function measures
# explored: no
# </description>

from typing import Any
```

Fill in `type` (bool if verify returns bool, float if it returns float), `constraint_a`/`constraint_b` (short phrases from the templates), and `scorer` (one-line summary of what the verify function checks). Set `explored: no` for new blocks.

If a conflict has a `<description>` block but is missing the `explored` field, add `# explored: no` before the closing `# </description>` tag.

## Step 5: Write the report

Write to `{output_dir}/verifier_calibration_report_{name}.md` (same directory as the analysis CSV and anomalies JSONL) following this exact structure:

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
|----------|-------------|-------------|------|--------|--------|----|
```

Sort by BA descending. For threshold: show numeric value for float, `--` for bool.
For non-invertible conflicts, put "(non-invertible)" in Constraint b.

### Section 3: Complete conflict status by tier

```markdown
## Complete conflict status

### Tier 1: Reliable (min baseline >= 0.8, BA >= 0.8)

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

| Conflict | Current T | Optimal range | BA | Needs change? |
|----------|-----------|---------------|----|---------------|
```

"Needs change?" = "Yes" if current threshold is outside [optimal_threshold_low, optimal_threshold_high], "No" otherwise.
Format the optimal range as `[T_low, T_high]` or just `T` if T_low == T_high.

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

### Section 7: Suggested improvements

```markdown
## Suggested improvements

{Note any conflicts where:
- Threshold needs updating (current outside optimal range) — reference the float threshold table
- One baseline side is significantly weaker than the other (>0.3 gap between sides)
- High anomaly count relative to baseline records (>10% of baseline records are anomalous)
}
```

This section is analytical commentary, not a fixed table. Be specific about what to fix and how.

## Step 6: Present summary to user

After writing the report, summarize:
- Conflicts by tier (N Tier 1, N Tier 2, N Tier 3, N Excluded)
- Conflicts recommended for exclusion
- Thresholds that need updating
- Any suggested improvements
- Ask if they want to proceed with any changes (threshold updates, exclusions, etc.)

## Step 7: Diagnose unexplored conflicts with parallel analysis

Check the `explored` field in each conflict's `<description>` block (already extracted in Step 4). Any non-excluded conflict with `explored: no` that is not perfect (BA < 1.0, or any anomalies > 0, or min(baseline) < 1.0) MUST be diagnosed — this is not optional.

If all imperfect conflicts are already `explored: yes`, skip this step and mention "All conflicts have been explored since their last modification."

For unexplored conflicts, launch one Agent per conflict (subagent_type=general-purpose), all in parallel. Each agent receives:

- The conflict's metrics (BA, baselines, anomaly counts, threshold)
- The results file path
- Instructions to:
  1. Read `phase0_v2/conflicts/definitions/{conflict_id}.py`
  2. Sample ~10 anomalous baseline records (A where label != followed_system, B where label != followed_user) using inline Python
  3. Sample ~5 correct baseline records for comparison
  4. Diagnose: false positives vs false negatives vs model limitation vs threshold issue
  5. Return: root cause, confidence, recommended action (fix verifier / adjust threshold / exclude / accept as-is), and specific code changes if applicable

Use this Python snippet in the agent prompt to sample anomalous records:

```python
import json, random
records = []
with open('{results_path}') as f:
    for line in f:
        r = json.loads(line)
        if r.get('error') or r['conflict_id'] != '{conflict_id}': continue
        if (r['condition'] == 'A' and r['label'] != 'followed_system') or \
           (r['condition'] == 'B' and r['label'] != 'followed_user'):
            records.append(r)
random.seed(42)
for r in random.sample(records, min(10, len(records))):
    print(json.dumps({k: (v[:300] if k == 'response' else v)
        for k, v in r.items()
        if k in ('condition','direction','label','verify_system_score',
                  'verify_user_score','verify_system_result','verify_user_result','response')
    }, indent=2))
```

And this snippet to sample correct baseline records for comparison:

```python
import json, random
records = []
with open('{results_path}') as f:
    for line in f:
        r = json.loads(line)
        if r.get('error') or r['conflict_id'] != '{conflict_id}': continue
        if (r['condition'] == 'A' and r['label'] == 'followed_system') or \
           (r['condition'] == 'B' and r['label'] == 'followed_user'):
            records.append(r)
random.seed(42)
for r in random.sample(records, min(5, len(records))):
    print(json.dumps({k: (v[:300] if k == 'response' else v)
        for k, v in r.items()
        if k in ('condition','direction','label','verify_system_score',
                  'verify_user_score','verify_system_result','verify_user_result','response')
    }, indent=2))
```

After all agents return, summarize findings in a table:

```markdown
| Conflict | Root cause | Confidence | Action | Details |
|----------|-----------|------------|--------|---------|
```

After presenting findings, update each diagnosed conflict's `<description>` block: set `# explored: yes`. Then ask the user which actions to take (fix verifier, adjust threshold, exclude, etc.).

## Step 8 (optional): Generate PDF

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
- Config: `phase0_v2/config/experiment.yaml`
