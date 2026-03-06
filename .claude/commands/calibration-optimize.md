---
description: "Optimize conflict verifiers and run the reverify-analyze-rescore pipeline. Use when the user wants to fix verifier issues, improve BA scores, update thresholds, or apply the full calibration pipeline after code changes. This is the write-heavy command that modifies conflict definitions and verifier code."
---

# Calibration Optimizer

Fix conflict verifiers to improve Balanced Accuracy, then run the full reverify-analyze-rescore pipeline. This command **modifies conflict definitions and verifier code**.

## Inputs

You need:
1. **Model ID** -- e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** -- at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.
3. **Output directory** -- default `phase0_v2/calibration/output/`.

Optional:
- **Specific conflict IDs** -- comma-separated list to optimize only those conflicts (e.g., from a `/calibration-diagnose` recommendation)

If `$ARGUMENTS` is provided, treat it as the model ID.

## Step 1: Identify optimization targets

Run analysis to get current metrics:

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir {output_dir} \
  --config phase0_v2/config/experiment.yaml \
  --model-config {model_id}
```

Parse `calibration_report.csv` and `anomalies.jsonl`. Extract description blocks:

```bash
awk '/# <description>/,/# <\/description>/{print FILENAME": "$0}' phase0_v2/conflicts/definitions/*.py
```

If specific conflicts were requested, use those. Otherwise target all conflicts that are:
- Not excluded in `experiment.yaml`
- `explored: no` in their description block
- Imperfect: BA < 1.0, or any anomalies > 0, or min(baseline) < 1.0

Present the target list with metrics and ask for confirmation.

## Step 2: Launch optimization subagents

For each target conflict, launch one Agent (subagent_type=general-purpose, run_in_background=true). Launch all agents in parallel.

**Agent prompt template:**

```
You are optimizing the `{conflict_id}` conflict verifier to maximize Balanced Accuracy (BA).

**Current metrics:** BA={ba}, SBR(a)={sbr_a}, UCR(a)={ucr_a}, SBR(b)={sbr_b}, UCR(b)={ucr_b}, Threshold={threshold}
**Results file:** {results_path}
**Conflict file:** phase0_v2/conflicts/definitions/{conflict_id}.py
**Scorer functions in:** phase0_v2/conflicts/verify_utils.py

**Your task (all steps required):**

1. **Read** the conflict definition and scorer functions
2. **Sample anomalous records** to understand failure patterns:
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
   for r in random.sample(records, min(15, len(records))):
       print(json.dumps({{k: (v[:400] if k == 'response' else v)
           for k, v in r.items()
           if k in ('condition','direction','label','verify_system_score',
                     'verify_user_score','verify_system_result','verify_user_result','response')
       }}, indent=2))
   ```
3. **Sample correct records** for comparison (same snippet but filter for correct labels)
4. **Diagnose** the root cause of each failure type
5. **Implement fixes** if verifier improvements are possible
6. **Validate** every change with the test harness:
   ```bash
   uv run python -m phase0_v2.calibration.test_verifier \
     {results_path} --conflict {conflict_id}
   ```
7. **Fix any tests** that break due to your changes:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_id_keyword}
   ```
8. **Update the `<description>` block** in the conflict file:
   - Set `# explored: yes`
   - Update `scorer` text if you changed scoring logic
   - If no `<description>` block exists, create one after the module docstring, before imports:
     ```python
     # If you modify the scoring logic, update the description block below
     # and set explored to 'no'.
     # <description>
     # type: bool or float
     # constraint_a: Short phrase from system_template
     # constraint_b: Short phrase from user_template
     # scorer: What the verify function measures
     # explored: yes
     # </description>
     ```
9. **Report:**
   - Root cause classification: **model inability** (model doesn't follow instruction), **verifier issue** (scorer doesn't detect what model does), or **constraint design** (the constraint itself is problematic)
   - If constraint design issue: recommend specific constraint changes
   - Final BA and baseline rates after changes
   - What you changed and why

**Rules:**
- Do NOT overfit. Every aspect of the verifier logic must make sense from a high-level explanation. If a change only helps on this specific dataset but wouldn't generalize, don't do it.
- If the root cause is model inability (model simply can't follow the instruction), accept the BA as-is and report that.
- If you try multiple approaches, keep only the one with the best BA. Revert failed attempts.
- Iterate: if your first fix improves BA but there's room for more, try additional changes.
```

## Step 3: Validate combined changes

After all agents complete:

1. Collect each agent's results (BA before/after, root cause, changes made)
2. **Revert** any changes that caused BA regression
3. Run full test suite:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short
   ```
4. Fix any remaining test failures from the combined changes

Present optimization results:

```markdown
## Optimization results

| Conflict | BA Before | BA After | Root cause | Action taken |
|----------|-----------|----------|------------|--------------|
```

## Step 4: Run reverify-analyze-rescore pipeline

After presenting results, ask user if they want to proceed with the pipeline. Then run:

### Why this order: reverify -> analyze -> optimize thresholds -> rescore

1. **Reverify first** -- Re-runs the actual verify functions on stored response text to produce fresh scores. Must come first because verifier code just changed.

2. **Analyze second** -- Computes metrics and finds optimal thresholds from the new score distributions. Must come after reverify.

3. **Optimize thresholds** -- Update `experiment.yaml` with optimal threshold values from analysis output.

4. **Rescore last** -- Re-applies new thresholds to reverified scores. Lightweight pass, no verify re-runs.

### Commands

```bash
# 1. Reverify: re-run verify functions with new verifier code
uv run python -m phase0_v2.calibration.rescore \
  {results_path} {results_path} --reverify

# 2. Analyze: compute metrics and find optimal thresholds
uv run python -m phase0_v2.calibration.analyze \
  {results_path} --output-dir {output_dir} \
  --config phase0_v2/config/experiment.yaml \
  --model-config {model_id}

# 3. Check the float threshold table in the output for "Needs change?"
# Update experiment.yaml thresholds to optimal values where needed

# 4. Rescore: apply new thresholds to reverified scores
uv run python -m phase0_v2.calibration.rescore \
  {results_path} {results_path} \
  --model-config {model_id}
```

After rescoring, verify threshold consistency is OK:
```bash
uv run python -m phase0_v2.calibration.analyze \
  {results_path} --output-dir {output_dir} \
  --config phase0_v2/config/experiment.yaml \
  --model-config {model_id}
```

The threshold consistency check should report "OK". If not, thresholds were not updated correctly.

## Step 5: Present final results

Show the updated metrics and suggest:
- **`/calibration-report`** to generate a full updated report with the new metrics

### Exclusion policy

NEVER add conflicts to `exclude_conflicts` automatically. Exclusion decisions are made by the human. The optimizer should recommend exclusions with justification but not apply them. Keeping a weak conflict in the data does no harm -- removing it loses information.

## Key references

- Process doc: `phase0_v2/calibration/iterative_calibration_process.md`
- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py`
- Analysis tool: `phase0_v2/calibration/analyze.py`
- Rescore tool: `phase0_v2/calibration/rescore.py`
- Test harness: `phase0_v2/calibration/test_verifier.py`
- Config: `phase0_v2/config/experiment.yaml`

## Related commands

- **`/calibration-report`** -- Generate the full calibration report (run after optimizing to see updated state)
- **`/calibration-diagnose`** -- Explore root causes read-only before committing to fixes
- **`/calibration-propose`** -- Design and implement new conflict definitions
