---
description: "Optimize conflict verifiers and run the reverify-analyze-rescore pipeline. Use when the user wants to fix verifier issues, improve BA scores, update thresholds, or apply the full calibration pipeline after code changes. This is the write-heavy command that modifies conflict definitions and verifier code."
---

# Calibration Optimizer

Implement verifier fixes from diagnostic reports, validate they work without regressions, then run the full reverify-analyze-rescore pipeline. This command **modifies conflict definitions and verifier code**.

**Prerequisite:** Run `/calibration-diagnose` first. This command reads the diagnosis reports and implements the proposed fixes — it does NOT re-diagnose from scratch.

## Inputs

You need:
1. **Model ID** -- e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** -- at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.

Optional:
- **Specific conflict IDs** -- comma-separated list to optimize only those conflicts

If `$ARGUMENTS` is provided, treat it as the model ID.

## Step 1: Identify optimization targets

Find the latest diagnosis reports:

```bash
ls -t phase0_v2/calibration/output/{safe_model_id}/diagnosis/*.md 2>/dev/null || true
```

For each target conflict, find its most recent diagnosis report: `{conflict_id}_*.md` (sorted by timestamp, take latest).

If specific conflicts were requested, use those. Otherwise, read the latest calibration report's diagnostic summary to identify conflicts with actionable verifier fixes (root cause = "verifier issue" or "threshold issue").

Present the target list with: conflict_id, current BA, diagnosed root cause, proposed fix from diagnosis, estimated BA after fix. Ask for confirmation before launching agents.

## Step 2: Launch optimization subagents

For each target conflict, launch one Agent (subagent_type=general-purpose, run_in_background=true). Launch all agents in parallel.

**Agent prompt template:**

```
You are optimizing the `{conflict_id}` conflict verifier to maximize Balanced Accuracy (BA).

**Diagnosis report:** {diagnosis_report_path}
**Current metrics:** BA={ba}, SBR(a)={sbr_a}, UCR(a)={ucr_a}, SBR(b)={sbr_b}, UCR(b)={ucr_b}, Threshold={threshold}
**Results file:** {results_path}
**Conflict file:** phase0_v2/conflicts/definitions/{conflict_id}.py
**Scorer functions in:** phase0_v2/conflicts/verify_utils.py

**Your task:**

1. **Read the diagnosis report** — this is your primary input. It contains root cause analysis, sampled anomalies, and proposed fixes with estimated BA improvements.

2. **Read** the conflict definition file and any scorer functions it references from `phase0_v2/conflicts/verify_utils.py`.

3. **Implement the proposed fix** from the diagnosis report. If the diagnosis proposes multiple options, start with the highest-confidence one.

4. **Validate the fix** with the test harness:
   ```bash
   uv run python -m phase0_v2.calibration.test_verifier \
     {results_path} --conflict {conflict_id} --sample-mismatches 10
   ```
   Check that:
   - BA improved (or at least didn't regress)
   - No new anomaly categories appeared
   - The fix matches the diagnosis estimate

5. **Verify no regressions** by sampling records that were previously correct:
   ```python
   import json, random
   correct, wrong = [], []
   with open('{results_path}') as f:
       for line in f:
           r = json.loads(line)
           if r.get('error') or r['conflict_id'] != '{conflict_id}': continue
           if (r['condition'] == 'A' and r['label'] == 'followed_system') or \
              (r['condition'] == 'B' and r['label'] == 'followed_user'):
               correct.append(r)
           elif r['condition'] in ('A', 'B'):
               wrong.append(r)
   random.seed(42)
   # Check a sample of previously-correct records still pass
   print(f"=== Previously correct: {{len(correct)}} | Previously wrong: {{len(wrong)}} ===")
   for r in random.sample(correct, min(10, len(correct))):
       print(json.dumps({{k: (v[:300] if isinstance(v, str) and k == 'response' else v)
           for k, v in r.items()
           if k in ('condition','direction','label','verify_system_score',
                     'verify_user_score','response')
       }}, indent=2))
   ```
   Re-run the verifier on these samples with a temp script to confirm they still produce correct labels. If any previously-correct record now fails, the fix has a regression — investigate and adjust.

6. **If the diagnosis fix doesn't work or regresses**, try alternative approaches. But always validate against both anomalous AND correct records. Do not chase BA improvements that introduce new failure patterns.

7. **Fix any tests** that break due to your changes:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_id_keyword}
   ```

8. **Update the `<description>` block** in the conflict file:
   - Update `scorer` text if you changed scoring logic
   - If no `<description>` block exists, create one after the module docstring, before imports:
     ```python
     # <description>
     # type: bool or float
     # constraint_a: Short phrase from system_template
     # constraint_b: Short phrase from user_template
     # scorer: What the verify function measures
     # </description>
     ```

9. **Quality targets**: The goal is min(BL) = 1.000 (all four baseline rates at 1.000). The minimum acceptable gate is min(BL) ≥ 0.95:
   - SBR(a) ≥ 0.95, UCR(a) ≥ 0.95, SBR(b) ≥ 0.95, UCR(b) ≥ 0.95
   - If any rate is below 0.95, report which rate(s) failed and the root cause (scorer, template, or model inability)
   - BA alone is not sufficient — a conflict with BA=0.96 but SBR(b)=0.88 is not acceptable for Tier 1
   - Always aim for perfect scores (1.000). Accept ≥ 0.95 only when further iteration shows no improvement.
   - Keep iterating as long as there is measurable progress — no hard iteration limit.

10. **Return a brief summary** (5-8 lines max):
   - BA before → BA after (and per-baseline changes if relevant)
   - Whether all baselines pass the ≥ 0.95 gate
   - What you changed (specific functions/logic)
   - Whether the diagnosis estimate was accurate
   - Any regressions found and how you addressed them
   - If the fix didn't work, explain why and what you tried

**Rules:**
- Do NOT overfit. Every aspect of the verifier logic must make sense from a high-level explanation. If a change only helps on this specific dataset but wouldn't generalize, don't do it.
- If the diagnosis says "model inability" and you were still asked to optimize this conflict, investigate whether there's a small verifier improvement the diagnosis missed. But don't force it — accept the BA if the model genuinely can't follow the instruction.
- Validate against BOTH anomalous and correct records. A fix that resolves anomalies but breaks correct records is worse than no fix.
- Keep only the best approach. Revert failed attempts cleanly.
```

## Step 3: Validate combined changes

After all agents complete:

1. Collect each agent's results (BA before/after, changes made)
2. **Revert** any changes that caused BA regression
3. Run full test suite:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short
   ```
4. Fix any remaining test failures from the combined changes

Present optimization results:

```markdown
## Optimization results

| Conflict | BA Before | BA After | Diagnosis est. | Action taken |
|----------|-----------|----------|----------------|--------------|
```

## Step 4: Run reverify-analyze-rescore pipeline

After presenting results, ask user if they want to proceed with the pipeline. Then run:

### Why this order: reverify -> analyze -> optimize thresholds -> rescore

1. **Reverify first** -- Re-runs the actual verify functions on stored response text to produce fresh scores. Must come first because verifier code just changed.

2. **Analyze second** -- Computes metrics and finds optimal thresholds from the new score distributions. Must come after reverify.

3. **Optimize thresholds** -- Update thresholds in `phase0_v2/config/thresholds.yaml` with the **midpoint** of each optimal threshold range. The midpoint maximizes margin from the range edges, which are more vulnerable to distribution shifts. For each float conflict: `new_threshold = (optimal_threshold_low + optimal_threshold_high) / 2`. The analyze output includes an `opt_mid` column with this value.

4. **Rescore last** -- Re-applies new thresholds to reverified scores. Lightweight pass, no verify re-runs.

### Commands

```bash
# 1. Reverify: re-run verify functions with new verifier code
uv run python -m phase0_v2.calibration.rescore \
  {results_path} {results_path} --reverify

# 2. Analyze: compute metrics and find optimal thresholds
uv run python -m phase0_v2.calibration.analyze \
  {results_path} --output-dir {output_dir} \
  --config phase0_v2/config/experiment.yaml

# 3. Check the float threshold table — for each conflict where current threshold
#    differs from opt_mid, update the threshold in phase0_v2/config/thresholds.yaml.
#    Read the CSV: optimal_threshold column has the midpoint.

# 4. Rescore: apply new thresholds to reverified scores
uv run python -m phase0_v2.calibration.rescore \
  {results_path} {results_path}
```

After rescoring, verify threshold consistency is OK:
```bash
uv run python -m phase0_v2.calibration.analyze \
  {results_path} --output-dir {output_dir} \
  --config phase0_v2/config/experiment.yaml
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
- Diagnosis reports: `phase0_v2/calibration/output/{safe_model_id}/diagnosis/`

## Related commands

- **`/calibration-report`** -- Generate the full calibration report (run after optimizing to see updated state)
- **`/calibration-diagnose`** -- Explore root causes read-only before committing to fixes
- **`/calibration-propose`** -- Design and implement new conflict definitions
