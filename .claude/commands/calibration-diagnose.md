---
description: "Diagnose root causes of weak conflict verifiers without modifying code. Use when the user wants to investigate why specific conflicts have low BA, explore failure patterns, understand anomalies, or get a diagnostic breakdown before committing to fixes. Read-only exploration -- no permanent code changes."
---

# Calibration Diagnostic Explorer

Explore root causes of weak or imperfect conflicts **without modifying any conflict definitions or verifier code**. This command launches read-only diagnostic subagents that sample records, test hypotheses with temp scripts, classify root causes, and write detailed reports to disk.

## Inputs

You need:
1. **Model ID** -- e.g., `meta-llama/Llama-3.1-8B-Instruct`. Ask user if not specified.
2. **Results file** -- at `phase0_v2/data/results/{safe_model_id}_results.jsonl` where `/` in model ID becomes `_`. Confirm it exists.

Optional:
- **Specific conflict IDs** -- comma-separated list to diagnose only those conflicts
- **Selection criteria** -- e.g., "all imperfect", "BA < 0.95", "tier 2 and 3"

If `$ARGUMENTS` is provided, treat it as the model ID (or conflict IDs if it contains underscores and no slashes).

Diagnose whatever the user asks for. There is no automatic filtering by `explored` status -- the user controls the target set.

## Step 1: Get current metrics

Run the analysis to get current metrics:

```bash
uv run python -m phase0_v2.calibration.analyze \
  phase0_v2/data/results/{safe_model_id}_results.jsonl \
  --output-dir phase0_v2/calibration/output/{safe_model_id}/ \
  --config phase0_v2/config/experiment.yaml
```

Parse `calibration_report.csv` and `anomalies.jsonl` from `phase0_v2/calibration/output/{safe_model_id}/` to get per-conflict metrics (BA, baselines, anomaly counts, threshold, type).

## Step 2: Identify diagnostic targets

Apply the user's selection criteria to the metrics. Present the target list and ask for confirmation before launching agents. Show: conflict_id, BA, min(baseline), anomaly count.

Examples of selection criteria:
- "all not perfect" = any conflict with BA < 1.0 or any baseline < 1.0
- "below tier 1" = any conflict with min(baseline) < 0.95 or BA < 0.95
- "these five conflicts: X, Y, Z, W, V"
- "BA < 0.95"
- "tier 2 and 3" = use tier definitions from calibration-report

**Quality gate context:** Tier 1 requires ALL four baselines (SBR(a), UCR(a), SBR(b), UCR(b)) ≥ 0.95 AND BA ≥ 0.95. Conflicts below this are diagnostic targets.

## Step 3: Create output directory

```bash
mkdir -p phase0_v2/calibration/output/{safe_model_id}/diagnosis
```

## Step 4: Launch diagnostic subagents

For each target conflict, launch one Agent (subagent_type=general-purpose, run_in_background=true). Launch all agents in parallel.

The report filename uses the pattern: `{conflict_id}_{MMDD}_{HHMM}.md` where the timestamp is the current time when the agent is launched.

**Agent prompt template:**

```
You are diagnosing the `{conflict_id}` conflict verifier to understand why it has imperfect Balanced Accuracy. This is a READ-ONLY diagnostic -- do NOT modify any files under `phase0_v2/conflicts/`.

**Current metrics:** BA={ba}, SBR(a)={sbr_a}, UCR(a)={ucr_a}, SBR(b)={sbr_b}, UCR(b)={ucr_b}, Threshold={threshold}, Type={type}
**Anomalies:** followed_both={fb}, cond_A_followed_user={afu}, cond_B_followed_system={bfs}
**Results file:** {results_path}
**Conflict file:** phase0_v2/conflicts/definitions/{conflict_id}.py
**Report output path:** phase0_v2/calibration/output/{safe_model_id}/diagnosis/{conflict_id}_{MMDD}_{HHMM}.md

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
       print(json.dumps({{k: (v[:400] if isinstance(v, str) and k == 'response' else v)
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

7. **Write the detailed report** to the output path specified above. The report MUST follow this exact structure:

---

# Diagnostic Report: `{conflict_id}`

**Date:** {YYYY-MM-DD HH:MM}
**Model:** {model_id}
**Results file:** {results_path}

## Conflict Description

{Read from the <description> block in the conflict definition file. Include type, constraint_a description, constraint_b description, scorer description.}

## Constraint Prompts

### Condition C: a_to_b (system=a, user=b)
- **System prompt constraint:** "{system_template with args filled in}"
- **User prompt constraint:** "{user_template with args filled in}"

### Condition C: b_to_a (system=b, user=a)
- **System prompt constraint:** "{inverse_system_template with args filled in}"
- **User prompt constraint:** "{inverse_user_template with args filled in}"

## Verification Logic

{Describe the verify functions in detail:
- What each function checks
- For float scorers: the scoring formula, threshold, and anti-correlation mechanism
- For bool scorers: the exact pass/fail conditions
- Key helper functions used and their logic
- Include relevant code snippets (< 20 lines each)}

## Current Metrics

| Metric | Value |
|--------|-------|
| Balanced Accuracy | {ba} |
| SBR(a) | {sbr_a} |
| UCR(a) | {ucr_a} |
| SBR(b) | {sbr_b} |
| UCR(b) | {ucr_b} |
| Type | {type} |
| Threshold | {threshold or N/A} |
| Optimal range | {opt_range or N/A} |
| Anomalies (cond_A_followed_user) | {afu} |
| Anomalies (cond_B_followed_system) | {bfs} |
| Anomalies (followed_both) | {fb} |

## Error Analysis

### Baseline failure inventory

{Table showing every baseline failure bucket:
| Direction | Condition | Constraint tested | Expected | n | Failures | Rate | Pattern |}

### Sampled anomalous records

{For each sampled anomaly, show:
- condition, direction, label
- verify scores (system and user)
- response excerpt (first 300 chars)
- what went wrong (verifier error? model error? edge case?)
Group by failure pattern if patterns emerge.}

### Sampled correct records (for comparison)

{Brief summary of what correct records look like, to contrast with failures.}

## Root Cause Classification

**Primary:** {model inability | verifier issue | constraint design | threshold issue}
**Secondary:** {if applicable}

{Detailed explanation with evidence. Reference specific sampled records.}

## Proposed Improvements

{For each proposed fix:
1. What to change (specific function, threshold, logic)
2. Why it helps (which failures it addresses)
3. Estimated BA improvement
4. Risk of regression
5. Complexity (trivial / moderate / significant / requires redesign)}

## Estimated Achievable BA

| Scenario | BA | Confidence | Notes |
|----------|-----|------------|-------|
| Current | {ba} | -- | |
| {fix 1} | {est} | {H/M/L} | {notes} |
| {fix 2} | {est} | {H/M/L} | {notes} |
| Theoretical ceiling | {est} | {H/M/L} | {notes} |

## Key Files

- {List all files read during diagnosis with brief description}

---

8. **Return a brief summary** (5-8 lines max) to the parent agent with:
   - Root cause classification
   - Current BA and estimated achievable BA
   - One-line description of the top proposed fix
   - Confidence level

**Rules:**
- Do NOT modify any files under `phase0_v2/conflicts/`. This is read-only.
- Temp scripts go in `/tmp/` only. Clean them up when done.
- Be thorough in sampling -- look at enough records to be confident in your diagnosis.
- If the root cause is ambiguous, list multiple contributing factors with relative importance.
- The report file is the primary output. Make it thorough and self-contained.
```

## Step 5: Compile diagnostic table

After all agents complete, collect their brief summaries and compile into a summary table:

```markdown
## Diagnostic Results

| Conflict | BA | Root Cause | Est. BA After | Confidence | Proposed Action |
|----------|----|------------|---------------|------------|-----------------|
```

Include a note that detailed reports are available at:
```
phase0_v2/calibration/output/{safe_model_id}/diagnosis/{conflict_id}_{MMDD}_{HHMM}.md
```

## Step 6: Present results

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
- For detailed analysis: read reports in `phase0_v2/calibration/output/{safe_model_id}/diagnosis/`

## Step 7: Append diagnostic summary to most recent calibration report

Find the most recent calibration report for this model:

```bash
ls -t phase0_v2/calibration/output/{safe_model_id}/calibration_report_*.md 2>/dev/null | head -1
```

If a report exists, use the Edit tool to replace the diagnostic placeholder section:

Replace:
```
## Diagnostic summary

_No diagnostics have been run for this report yet. Run `/calibration-diagnose` to populate this section._
```

With the compiled diagnostic results: the summary table, root-cause groupings, and links to the individual report files. This makes the calibration report a single self-contained document that includes both the metrics snapshot and the diagnostic analysis.

If no calibration report exists for this model, skip this step.

## Key references

- Conflict definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py`
- Test harness: `phase0_v2/calibration/test_verifier.py`
- Analysis tool: `phase0_v2/calibration/analyze.py`
- Diagnostic reports: `phase0_v2/calibration/output/{safe_model_id}/diagnosis/`

## Related commands

- **`/calibration-report`** -- Generate the full calibration report (run this first to see current state)
- **`/calibration-optimize`** -- Fix verifiers and run the reverify-analyze-rescore pipeline
- **`/calibration-propose`** -- Design and implement new conflict definitions
