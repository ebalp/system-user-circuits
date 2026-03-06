---
description: "Diagnose root causes of weak conflict verifiers without modifying code. Use when the user wants to investigate why specific conflicts have low BA, explore failure patterns, understand anomalies, or get a diagnostic breakdown before committing to fixes. Read-only exploration -- no permanent code changes."
---

# Calibration Diagnostic Explorer

Explore root causes of weak or imperfect conflicts **without modifying any conflict definitions or verifier code**. This command launches read-only diagnostic subagents that sample records, test hypotheses with temp scripts, and classify root causes.

## Inputs

You need:
1. **Model ID** -- e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** -- at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.

Optional:
- **Specific conflict IDs** -- comma-separated list to diagnose only those conflicts
- **BA threshold** -- diagnose all conflicts with BA below this value (default: diagnose all unexplored imperfect conflicts)

If `$ARGUMENTS` is provided, treat it as the model ID (or conflict IDs if it contains underscores and no slashes).

## Step 1: Get current metrics

Run the analysis to get current metrics:

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir phase0_v2/calibration/output/ \
  --config phase0_v2/config/experiment.yaml \
  --model-config {model_id}
```

Parse `calibration_report.csv` and `anomalies.jsonl` to get per-conflict metrics (BA, baselines, anomaly counts, threshold, type).

## Step 2: Identify diagnostic targets

Filter conflicts by the user's criteria. If no specific criteria given, target all conflicts that are:
- Not excluded in `experiment.yaml`
- Imperfect: BA < 1.0, or any anomalies > 0, or min(baseline) < 1.0
- Optionally filtered by `explored: no` in their description block

Extract descriptions to check explored status:
```bash
awk '/# <description>/,/# <\/description>/{print FILENAME": "$0}' phase0_v2/conflicts/definitions/*.py
```

Present the target list to the user and ask for confirmation before launching agents. Show: conflict_id, BA, min(baseline), anomaly count.

## Step 3: Launch diagnostic subagents

For each target conflict, launch one Agent (subagent_type=general-purpose, run_in_background=true). Launch all agents in parallel.

**Agent prompt template:**

```
You are diagnosing the `{conflict_id}` conflict verifier to understand why it has imperfect Balanced Accuracy. This is a READ-ONLY diagnostic -- do NOT modify any files under `phase0_v2/conflicts/`.

**Current metrics:** BA={ba}, SBR(a)={sbr_a}, UCR(a)={ucr_a}, SBR(b)={sbr_b}, UCR(b)={ucr_b}, Threshold={threshold}, Type={type}
**Anomalies:** followed_both={fb}, cond_A_followed_user={afu}, cond_B_followed_system={bfs}
**Results file:** {results_path}
**Conflict file:** phase0_v2/conflicts/definitions/{conflict_id}.py

**Your task:**

1. **Read** the conflict definition file and any scorer functions it uses from `phase0_v2/conflicts/verify_utils.py`

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

3. **Sample correct baseline records** for comparison (same snippet but filter for correct labels)

4. **Test hypotheses** with temporary scripts in `/tmp/`:
   - Create scripts like `/tmp/test_{conflict_id}.py` to test alternative scoring approaches
   - Use `uv run python -m phase0_v2.calibration.test_verifier {results_path} --conflict {conflict_id}` to check current metrics
   - Try variations in temp scripts to estimate what BA improvements are achievable
   - Clean up temp scripts when done

5. **Classify the root cause** as one of:
   - **model inability**: The model genuinely cannot follow this instruction well. The verifier is correct but the model fails. No verifier fix will help.
   - **verifier issue**: The scorer doesn't properly detect what the model is doing. Specific fix available.
   - **constraint design**: The constraint itself is problematic (e.g., truncation confound, independent properties, positional dependency). May need redesign.
   - **threshold issue**: The threshold is suboptimal. Adjusting it would improve BA without code changes.

6. **Estimate achievable BA** with confidence level (high/medium/low):
   - If verifier issue: estimate BA after proposed fix
   - If model inability: current BA is the ceiling
   - If constraint design: estimate BA after redesign, note complexity

7. **Report** your findings as a structured summary:
   - Root cause classification
   - Evidence (specific examples from sampled records)
   - Proposed fix (description only -- do NOT implement)
   - Estimated BA after fix
   - Confidence level and reasoning

**Rules:**
- Do NOT modify any files under `phase0_v2/`. This is read-only.
- Temp scripts go in `/tmp/` only. Clean them up when done.
- Be thorough in sampling -- look at enough records to be confident in your diagnosis.
- If the root cause is ambiguous, list multiple contributing factors with relative importance.
```

## Step 4: Compile diagnostic table

After all agents complete, collect their reports and compile into a summary table:

```markdown
## Diagnostic Results

| Conflict | BA | Root Cause | Est. BA After | Confidence | Proposed Action |
|----------|----|------------|---------------|------------|-----------------|
```

## Step 5: Present results

Group results by root cause category and present to the user:

### Model inability
Conflicts where the model can't follow the instruction. These are at their ceiling.

### Verifier issues (fixable)
Conflicts where scorer improvements would help. List the proposed fix for each.

### Constraint design issues
Conflicts that need redesign. Describe what's fundamentally wrong.

### Threshold issues
Conflicts where only threshold adjustment is needed (simplest fix).

Then suggest next steps:
- For verifier/threshold issues: **`/calibration-optimize --conflicts X,Y,Z`**
- For constraint design issues: **`/calibration-propose`** to design replacements
- For model inability: accept current BA or consider exclusion

## Key references

- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py`
- Test harness: `phase0_v2/calibration/test_verifier.py`
- Analysis tool: `phase0_v2/calibration/analyze.py`

## Related commands

- **`/calibration-report`** -- Generate the full calibration report (run this first to see current state)
- **`/calibration-optimize`** -- Fix verifiers and run the reverify-analyze-rescore pipeline
- **`/calibration-propose`** -- Design and implement new conflict definitions
