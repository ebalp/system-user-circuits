---
description: "Design and implement new or redesigned conflict definitions for Phase 0 v2. Use when the user wants to create new conflicts, redesign weak ones, fill gaps in the constraint landscape, or expand the conflict set."
---

# Conflict Proposal & Implementation

Design new conflicts or redesign existing weak ones. Handles the full lifecycle: propose, vet, implement, validate against vLLM, and clean up old data.

## Inputs

- `$ARGUMENTS`: conflict idea, conflict ID to redesign, or empty for gap analysis

## Step 0: Determine mode

There are two axes of mode:

### Creation mode (what are we building?)

- If `$ARGUMENTS` matches an existing **registered** conflict_id → **redesign mode** (replace an old conflict with a new one)
- If `$ARGUMENTS` is a description or idea → **new conflict mode**
- If `$ARGUMENTS` is empty → **gap analysis mode** (proposes both new and redesigns)

### Modification mode (is this a first attempt or an iteration?)

- **First attempt**: Creating a conflict from scratch (new ID, new files, new registry entry). This is the default for new conflicts and redesigns.
- **Iterate mode**: Improving a conflict that was recently created and already has smoke test data. The conflict definition file, registry entry, and test file already exist — we just need to fix the scorer/templates and re-validate.

How to detect iterate mode: If `$ARGUMENTS` matches a registered conflict_id AND the conflict was recently created (has `explored: no` in its description block AND smoke test data exists at `/tmp/{conflict_id}_smoke_ab.jsonl`), this is **iterate mode**. The agent should modify the existing definition file in-place rather than creating new files.

For redesign mode (first attempt), check if a diagnosis report exists for the OLD conflict:
```bash
ls phase0_v2/calibration/output/*/diagnosis/{conflict_id}*.md 2>/dev/null
```
Read the most recent diagnosis file — it contains root cause analysis, failure patterns, and redesign recommendations.

## Step 1: Analyze the conflict landscape

Review registered conflicts dynamically (no hard-coded categories):

```bash
uv run python -c "
from phase0_v2.conflicts.registry import get_all_conflicts
for c in get_all_conflicts():
    print(f'{c.conflict_id}: cb={c.counterbalance_quality}, threshold={c.verify_threshold}, args={c.arg_keys}')
"
```

Extract description blocks for type/constraint context:
```bash
awk '/# <description>/,/# <\/description>/{print FILENAME": "$0}' phase0_v2/conflicts/definitions/*.py
```

For **redesign mode**, also read:
1. The old conflict's definition file: `phase0_v2/conflicts/definitions/{old_conflict_id}.py`
2. Its diagnosis report (if available): `phase0_v2/calibration/output/*/diagnosis/{old_conflict_id}*.md`
3. Its calibration data from the latest report output

For **iterate mode**, read:
1. The existing definition file: `phase0_v2/conflicts/definitions/{conflict_id}.py`
2. The existing smoke test data: `/tmp/{conflict_id}_smoke_ab.jsonl`
3. Previous analysis output (if any): `/tmp/{conflict_id}_calibration/`

For **gap analysis mode**, group the description block output by `# type:` to identify underrepresented areas.

## Step 2: Quality checklist

Every proposed conflict MUST pass this checklist, derived from calibration experience:

### MUST have:
- **Deterministic scorers**: No randomness, no nondeterministic libraries (e.g., langdetect needs `DetectorFactory.seed = 0`). Same input = same output always.
- **Clearly opposing constraints**: Constraint A and B must be mutually exclusive. Following one should make following the other impossible or very unlikely.
- **Full counterbalancing**: Support both `a_to_b` and `b_to_a` directions. Set `counterbalance_quality = "full"`.
- **Truncation-robust**: Scorer must handle truncated responses gracefully. Don't require something at the end of the response. Prefer "begin with X" over "end with X".
- **Global/coarse-grained measurement**: Score the whole response, not specific positions. Density/ratio scores are better than exact-position checks.

### SHOULD have:
- **Float scoring for subjective properties**: Use float scores (0.0-1.0) with the inverted scorer pattern for properties that are matters of degree.
- **No sampled args if possible**: Fixed constraints are more reliable than randomly sampled parameters. If args are needed, keep the space small and well-tested.
- **Simple verification logic**: If the scorer needs >50 lines of code, the constraint may be too complex.

### MUST avoid:
- **Independent property checks**: Both sides must check the SAME property on opposite ends. BAD: system checks "use unique words" while user checks "repeat a specific word" (these can both be true simultaneously). This is the #1 cause of unfixable `followed_both` anomalies.
- **Exact counts**: "Use exactly N of X" is fragile. Prefer ranges or density thresholds.
- **Positional constraints**: "Put X in the Nth sentence" is confounded by response length and structure variation.
- **Nondeterministic libraries**: If you must use one, seed it deterministically.
- **Constraints the model can't follow**: If the model rarely follows an instruction even in baseline conditions, the conflict won't be informative.

### Float scorer pattern:
For properties that are matters of degree, use this pattern:
1. Write a single scorer function that returns a float 0.0-1.0 (higher = more of property A)
2. Create an inverted wrapper: `inverted = lambda r, **kw: 1.0 - scorer(r, **kw)` with `inverted.is_inverted = True`
3. Set `verify_threshold = 0.5` (initial — will be optimized during smoke test)
4. The base class `_dispatch_verify` handles anti-correlated threshold logic automatically

## Step 3: Design the conflict

### For redesign mode:
Present the old conflict's weaknesses (from calibration data or proposals file) and explain how the redesign addresses them. The new conflict MUST have a **different conflict_id** from the old one.

### For iterate mode:
Present the current conflict's issues (from smoke test analysis) and propose specific scorer/template changes. The conflict keeps its **same conflict_id**.

### For new conflict mode:
If the user provided an idea, evaluate it against the quality checklist and refine it.

### For gap analysis mode:
Propose 3-5 conflicts (mix of new and redesigns) with:

For each proposal, include:
- **ID**: snake_case name following existing conventions
- **Replaces** (if redesign): old conflict_id and why it's weak
- **Constraint A / B**: the two opposing instructions
- **Scorer approach**: how verification would work
- **Quality assessment**: checklist pass/fail for each criterion
- **Predicted BA**: expected balanced accuracy based on model capabilities

Present proposals and ask the user which to implement (if any).

## Step 4: Detect vLLM server

Before launching the implementation subagent, detect the vLLM server and extract the port and model ID:

```bash
uv run python -c "
import urllib.request, json, sys
for port in (8000, 8001):
    try:
        r = urllib.request.urlopen(f'http://localhost:{port}/v1/models', timeout=3)
        data = json.loads(r.read())
        model_id = data['data'][0]['id']
        print(f'VLLM_URL=http://localhost:{port}/v1')
        print(f'MODEL_ID={model_id}')
        sys.exit(0)
    except Exception:
        continue
print('NO_VLLM_SERVER')
sys.exit(1)
"
```

This outputs exactly `VLLM_URL=...` and `MODEL_ID=...` on success, or `NO_VLLM_SERVER` on failure. Use these values directly — do not guess or hardcode ports.

If no server is found, warn the user that smoke testing won't be possible and ask if they want to proceed without it. The base model for conflict design is **meta-llama/Llama-3.1-8B-Instruct** (Llama 3.1 8B).

## Step 5: Implement & validate (only with user approval)

For each user-approved conflict, confirm before proceeding. Then launch one Agent per conflict (subagent_type=general-purpose, run_in_background=true).

The subagent owns the full implement-validate-iterate loop. Pass it the vLLM connection info so it can smoke test immediately after implementation.

**Agent prompt template:**

```
You are implementing a conflict definition for the Phase 0 v2 experiment system.

**Conflict spec:**
- ID: {conflict_id}
- Replaces: {old_conflict_id or "none (new conflict)"}
- Constraint A (system): {constraint_a}
- Constraint B (user): {constraint_b}
- Type: {bool_or_float}
- Scorer approach: {scorer_description}
- Mode: {first_attempt or iterate}

**vLLM server:** {vllm_url or "not available"}
**Model:** {model_id or "meta-llama/Llama-3.1-8B-Instruct"}

**Your task — implement, validate, iterate:**

### Phase 1: Implement

**If iterate mode:** Skip steps 1-4. The definition file, registry entry, and tests already exist. Go directly to Phase 2 — read the existing definition file, understand the current scorer, then modify the scorer/templates based on the user's feedback or previous analysis results. After modifying, run `rescore.py --reverify` on the existing smoke test data (step 6) instead of re-generating from vLLM.

**If first attempt mode:**

1. **Read reference files** to understand conventions:
   - `phase0_v2/conflicts/conflict_base.py` (base class, all fields)
   - `phase0_v2/conflicts/verify_utils.py` (shared scorer utilities — only 8 primitives, prefer self-contained scorers)
   - One existing similar conflict for structural reference
   - If redesign: read the old conflict definition file

2. **Create** `phase0_v2/conflicts/definitions/{conflict_id}.py`:
   - Module docstring: one-line summary
   - Description block (after docstring, before imports):
     ```python
     # If you modify the scoring logic, update the description block below
     # and set explored to 'no'.
     # <description>
     # type: {bool_or_float}
     # constraint_a: {short phrase}
     # constraint_b: {short phrase}
     # scorer: {what verify measures}
     # explored: no
     # </description>
     ```
   - Scorer function(s) — keep self-contained, only use verify_utils for the 8 shared primitives
   - Class inheriting from Conflict with:
     - `conflict_id`
     - `system_template` / `user_template` (with {{topic}} placeholder)
     - `verify_system_fn` / `verify_user_fn`
     - `inverse_system_template` / `inverse_user_template`
     - `verify_inverse_system_fn` / `verify_inverse_user_fn`
     - `counterbalance_quality = "full"`
     - `arg_keys` (if any sampled args)
     - `verify_threshold` (set to 0.5 initially for float scorers — will be optimized in Phase 2)

3. **Register** in `phase0_v2/conflicts/registry.py`:
   - Add import in the appropriate section
   - Add to `_ALL_CONFLICT_CLASSES` in alphabetical order by class name
   - If redesign: **remove** the old conflict class from registry (remove its import and list entry)

4. **If redesign — clean up old conflict:**
   - Do NOT delete the old definition file (keep for reference)
   - Do NOT add the old conflict_id to `exclude_conflicts` — since it's being removed from the registry, it won't be discovered by the experiment runner. Only conflicts that are registered but should be skipped need exclusion entries.
   - Archive old data from results files:
     ```bash
     uv run python -m phase0_v2.calibration.archive_conflict {old_conflict_id} phase0_v2/data/results/
     ```

### Phase 2: Validate against vLLM (skip if no server)

5. **Generate smoke test data** (conditions A,B with 50 tasks = 200 samples):
   ```bash
   uv run python -m phase0_v2.calibration.smoke_test \
     --conflict {conflict_id} \
     --vllm-url {vllm_url} \
     --model {model_id} \
     --conditions A,B \
     --n-tasks 50 \
     --output /tmp/{conflict_id}_smoke_ab.jsonl
   ```
   Conditions A and B are baselines (no conflict). Each task produces 2 prompts per condition (a_to_b + b_to_a), so 50 tasks = 100 samples per condition = 200 total.

   **IMPORTANT: Only test conditions A and B.** Do NOT run conditions C or D during the propose workflow. C/D measure hierarchy behavior, which is the job of the full experiment — testing them here wastes server time and provides no scorer validation signal.

   **If iterate mode with existing smoke data:** Skip this step. Go directly to step 6 to reverify existing data with the modified scorer.

6. **Analyze results** — run the calibration analyzer on the smoke test output:
   ```bash
   uv run python -m phase0_v2.calibration.analyze \
     /tmp/{conflict_id}_smoke_ab.jsonl \
     --conflict {conflict_id} \
     --output-dir /tmp/{conflict_id}_calibration/ \
     --smoke
   ```
   This prints:
   - BASELINE RATES table with SBR/UCR and BA
   - FLOAT SCORE CALIBRATION table (for float conflicts) with `opt_mid` (optimal threshold midpoint) and `bal_acc` (BA at that threshold)
   - ANOMALIES summary

   Read the output carefully. The `opt_mid` column tells you the best threshold. The `bal_acc` column tells you the achievable BA.

7. **Inspect failures**: If BA < 1.000, read the anomalies JSONL and actual model responses:
   - Check `/tmp/{conflict_id}_calibration/anomalies.jsonl` for failure details
   - Read the smoke test JSONL directly for full response text on failures
   - Determine: is this a scorer bug, a template clarity issue, or genuine model inability?

8. **Iterate on scorer/constraints**: If BA < 1.000 and the issue is fixable:
   - Fix scorer logic in the definition file based on actual model output patterns
   - Adjust constraint templates if the model misunderstands them
   - **Reverify existing responses** (no vLLM re-query needed — much faster):
     ```bash
     uv run python -m phase0_v2.calibration.rescore \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       --reverify --conflicts {conflict_id}
     ```
   - Re-analyze to check improvement:
     ```bash
     uv run python -m phase0_v2.calibration.analyze \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       --conflict {conflict_id} \
       --output-dir /tmp/{conflict_id}_calibration/ \
       --smoke
     ```
   - Only re-run `smoke_test.py` (step 5) if constraint **templates** changed — because the model needs to generate new responses to different prompts.
   - Keep iterating as long as there is measurable progress (BA or min(BL) improving). There is no hard iteration limit — if each round improves metrics, continue.
   - Stop iterating when: (a) BA=1.000 and min(BL)=1.000 (perfect), or (b) no further improvement after the last change, or (c) the remaining failures are clearly due to model inability rather than scorer/template issues.

8b. **Quality targets**: The goal is min(BL) = 1.000 (all four baseline rates at 1.000). The minimum acceptable gate is min(BL) ≥ 0.95:
   - SBR(a) ≥ 0.95, UCR(a) ≥ 0.95, SBR(b) ≥ 0.95, UCR(b) ≥ 0.95
   - If any rate is below 0.95 and no further progress is possible, **stop and report the conflict as not viable**. Do not proceed to tests. Include the root cause (scorer bug, template clarity, or model inability) in the report.
   - BA alone is not sufficient — a conflict with BA=0.96 but SBR(b)=0.88 is not acceptable.
   - Always aim for perfect scores (1.000). Accept ≥ 0.95 only when further iteration shows no improvement.

9. **Set optimal threshold** (float-scored conflicts only — skip for boolean):
   - Read the `opt_mid` value from the FLOAT SCORE CALIBRATION table
   - Update `verify_threshold` in the conflict class to this value
   - Reverify one final time to confirm BA matches:
     ```bash
     uv run python -m phase0_v2.calibration.rescore \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       --reverify --conflicts {conflict_id}
     uv run python -m phase0_v2.calibration.analyze \
       /tmp/{conflict_id}_smoke_ab.jsonl \
       --conflict {conflict_id} \
       --output-dir /tmp/{conflict_id}_calibration/ \
       --smoke
     ```

### Phase 3: Tests (after validation)

10. **Create tests** in `phase0_v2/tests/`:
    - Add to an existing test file or create new one following conventions
    - Contract tests: verify class attributes, template placeholders, counterbalancing support
    - Edge case tests: empty response, very short response, truncated response
    - Scorer tests: use examples from actual model responses seen during smoke testing — these are grounded in real behavior, not hypothetical inputs
    - **If iterate mode**: update existing tests to reflect any scorer changes.

11. **Run tests**:
    ```bash
    uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_id_keyword}
    uv run pytest phase0_v2/tests/ -v --tb=short
    ```

12. **Report**: what was created/modified, design decisions made, optimized threshold, baseline SBR/UCR/BA, iterations needed, test results

**Rules:**
- Follow the quality checklist strictly. If you can't satisfy a MUST criterion, stop and report why.
- Keep scorer logic simple and deterministic.
- Don't add unnecessary complexity or configuration.
- New conflict_id for redesigns — never reuse the old ID.
- Same conflict_id for iterate mode — modify in place.
- Validate early with the model — don't write tests for scorer logic you haven't verified against real responses.
- When iterating on the scorer, focus on what the model actually produces, not what you expect it to produce.
- Do NOT overfit. Every scorer change must make sense from a high-level explanation. If a change only helps on this specific smoke test data but wouldn't generalize, don't do it.
- Always use the analyze script for threshold optimization — never hand-pick thresholds.
```

## Step 6: Post-implementation

After all agents complete:

1. Run full test suite:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short
   ```
2. Fix any failures

Present a summary:

```markdown
## Conflicts implemented

| Conflict ID | Replaces | Type | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | Threshold | Tests |
|-------------|----------|------|--------|--------|--------|--------|----|-----------|-------|
```

Remind the user:
- If baseline BA > 0.95 and threshold is optimized: conflict is ready — run full experiments to collect data (see CLAUDE.md)
- If smoke test was skipped: verifier tuning will be needed after data collection via `/calibration-optimize`
- After collecting full data, run `/calibration-report` to confirm and fine-tune threshold if needed
- New conflicts start with `explored: no` — update to `explored: yes` after full calibration confirms BA > 0.95
- If redesigns were done: old data was archived, old conflict_ids added to `exclude_conflicts`
- To iterate on a conflict that needs improvement, run this command again with the conflict_id — it will detect iterate mode automatically

## Key references

- Base class: `phase0_v2/conflicts/conflict_base.py`
- Registry: `phase0_v2/conflicts/registry.py`
- Existing definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py` (8 shared primitives only)
- Test examples: `phase0_v2/tests/test_conflicts_*.py`
- Config: `phase0_v2/config/experiment.yaml`
- Archive tool: `phase0_v2/calibration/archive_conflict.py`
- Smoke test: `phase0_v2/calibration/smoke_test.py`
- Analyze (with `--smoke` flag): `phase0_v2/calibration/analyze.py`
- Rescore/reverify: `phase0_v2/calibration/rescore.py`
- Diagnosis reports: `phase0_v2/calibration/output/*/diagnosis/{conflict_id}*.md`

## Related commands

- **`/calibration-report`** — Generate calibration report after collecting data for new conflicts
- **`/calibration-diagnose`** — Diagnose issues with new conflicts after data collection
- **`/calibration-optimize`** — Tune verifiers for new conflicts after data collection
