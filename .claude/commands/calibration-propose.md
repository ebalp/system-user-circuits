---
description: "Design and implement new conflict definitions for Phase 0 v2. Use when the user wants to create new conflicts, fill gaps in the constraint landscape, or expand the conflict set. For fixing existing conflicts, use /calibration-optimize instead."
---

# New Conflict Proposal & Implementation

Design and implement new conflicts. Handles the full lifecycle: brainstorm with the user, implement, smoke test against real models, iterate until quality targets are met.

For fixing or improving **existing** conflicts, use `/calibration-optimize` instead.

## Inputs

- `$ARGUMENTS`: conflict idea or description, or empty for gap analysis

## Step 1: Brainstorm with the user

### If the user provided an idea:

Evaluate the idea against the quality checklist (Step 2). Refine it collaboratively — propose concrete constraint A/B wording, scorer approach, and flag any checklist concerns.

### If no idea provided (gap analysis):

Review the current conflict landscape:

```bash
uv run python -c "
import yaml
with open('phase0_v2/config/conflicts.yaml') as f:
    data = yaml.safe_load(f)
for cid, meta in data.items():
    print(f'{cid}: type={meta.get(\"type\")}, a={meta.get(\"constraint_a\")}, b={meta.get(\"constraint_b\")}')
"
```

Group conflicts by `type` to identify underrepresented areas. Propose 3-5 new conflicts with:

- **ID**: snake_case name following existing conventions
- **Constraint A / B**: the two opposing instructions
- **Scorer approach**: how verification would work
- **Quality assessment**: checklist pass/fail for each criterion

Present proposals and ask the user which to implement (if any).

$ARGUMENTS

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
3. Add the conflict to `phase0_v2/config/thresholds.yaml` with threshold `0.5` (initial — will be optimized during smoke test)
4. The base class `_dispatch_verify` handles anti-correlated threshold logic automatically

## Step 3: Confirm design with user

Present the final design for each conflict:

```markdown
| Field | Value |
|-------|-------|
| ID | {conflict_id} |
| Constraint A (system) | {constraint_a} |
| Constraint B (user) | {constraint_b} |
| Type | {bool or float} |
| Scorer approach | {description} |
| Checklist | {all MUST criteria pass} |
```

Ask the user to confirm before proceeding to implementation.

## Step 4: Detect inference server

The smoke test uses `VLLMClient` which speaks the OpenAI-compatible API — it works with vLLM, HF TGI, or any OpenAI-compatible endpoint.

Check for a local server first:

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
print('NO_SERVER')
sys.exit(1)
"
```

If no local server is found, ask the user for an endpoint URL and model ID. They may provide an HF Inference Endpoint, a remote vLLM server, or another OpenAI-compatible API.

If no server is available at all, warn the user that smoke testing won't be possible and ask if they want to proceed without it (implementation only, validation deferred).

## Step 5: Implement & validate (only with user approval)

For each user-approved conflict, launch one Agent (subagent_type=general-purpose, model=opus, run_in_background=true).

The subagent owns the full implement-validate-iterate loop. Pass it the server connection info so it can smoke test immediately after implementation.

**Agent prompt template:**

```
You are implementing a new conflict definition for the Phase 0 v2 experiment system.

**Conflict spec:**
- ID: {conflict_id}
- Constraint A (system): {constraint_a}
- Constraint B (user): {constraint_b}
- Type: {bool_or_float}
- Scorer approach: {scorer_description}

**Inference server:** {server_url or "not available"}
**Model:** {model_id}

**Your task — implement, validate, iterate:**

### Phase 1: Implement

1. **Read reference files** to understand conventions:
   - `phase0_v2/conflicts/conflict_base.py` (base class, all fields)
   - `phase0_v2/conflicts/verify_utils.py` (shared scorer utilities — only 8 primitives, prefer self-contained scorers)
   - One existing similar conflict for structural reference

2. **Create** `phase0_v2/conflicts/definitions/{conflict_id}.py`:
   - Module docstring: one-line summary
   - Scorer function(s) — keep self-contained, only use verify_utils for the 8 shared primitives
   - Create/update the entry in `phase0_v2/config/conflicts.yaml` with type, constraint_a, constraint_b, scorer description
   - Class inheriting from Conflict with:
     - `conflict_id`
     - `system_template` / `user_template`
     - `verify_system_fn` / `verify_user_fn`
     - `inverse_system_template` / `inverse_user_template`
     - `verify_inverse_system_fn` / `verify_inverse_user_fn`
     - `counterbalance_quality = "full"`
     - `arg_keys` (if any sampled args)
     - Add threshold to `phase0_v2/config/thresholds.yaml` (set to 0.5 initially for float scorers — will be optimized in Phase 2)

3. **Register** in `phase0_v2/conflicts/registry.py`:
   - Add import in the appropriate section
   - Add to `_ALL_CONFLICT_CLASSES` in alphabetical order by class name

### Phase 2: Validate with smoke test (skip if no server)

4. **Generate smoke test data** (all 4 conditions):
   ```bash
   uv run python -m phase0_v2.calibration.smoke_test \
     --conflict {conflict_id} \
     --vllm-url {server_url} \
     --model {model_id} \
     --n-tasks 50 \
     --output /tmp/{conflict_id}_smoke.jsonl
   ```
   Tests all 4 conditions. A,B validate the scorer on clean baselines. C,D are critical because model behavior under conflict is much stranger — refusals, metacommentary, partial compliance, hedging — and the scorer must handle these gracefully. A scorer that works on baselines can completely break under condition C.

   The smoke test prints a full report: baseline rates (SBR/UCR), baseline BA, per-condition accuracy, condition C label distribution, and condition C BA.

   **Read the output carefully:**
   - Baseline BA: does the scorer reliably detect the constraint on clean responses? (target: 1.000)
   - Condition C labels: check for excessive `followed_both` or `followed_neither` — these suggest the constraints aren't truly mutually exclusive or the scorer has gaps with conflicted responses.
   - Condition C responses: read actual C responses to see if the scorer handles refusals, meta-commentary, and partial compliance correctly.

5. **Inspect failures**: If baseline BA < 1.000:
   - Read the smoke test JSONL directly for full response text on failures
   - Determine: is this a scorer bug, a template clarity issue, or genuine model inability?

6. **Iterate on scorer/constraints**: If baseline BA < 1.000 and the issue is fixable:
   - Fix scorer logic in the definition file based on actual model output patterns
   - Adjust constraint templates if the model misunderstands them
   - **Reverify existing responses** (no re-query needed — much faster):
     ```bash
     uv run python -m phase0_v2.calibration.rescore \
       /tmp/{conflict_id}_smoke.jsonl \
       /tmp/{conflict_id}_smoke.jsonl \
       --reverify --conflicts {conflict_id}
     ```
   - Re-run the smoke test report to check improvement:
     ```bash
     uv run python -m phase0_v2.calibration.audit_conflict \
       /tmp/{conflict_id}_smoke.jsonl --conflict {conflict_id}
     ```
   - Only re-run `smoke_test.py` (step 4) if constraint **templates** changed — because the model needs to generate new responses to different prompts.
   - Keep iterating as long as there is measurable progress (baseline BA or min(BL) improving).
   - Stop iterating when: (a) min(BL)=1.000 (perfect baselines), or (b) no further improvement after the last change, or (c) the remaining failures are clearly due to model inability rather than scorer/template issues.

7. **Quality targets**: The goal is min(BL) = 1.000 (all four baseline rates at 1.000). The minimum acceptable gate is min(BL) ≥ 0.95:
   - SBR(a) ≥ 0.95, UCR(a) ≥ 0.95, SBR(b) ≥ 0.95, UCR(b) ≥ 0.95
   - If any rate is below 0.95 and no further progress is possible, **stop and report the conflict as not viable**. Do not proceed to tests. Include the root cause (scorer bug, template clarity, or model inability) in the report.
   - BA alone is not sufficient — a conflict with BA=0.96 but SBR(b)=0.88 is not acceptable.
   - Always aim for perfect scores (1.000). Accept ≥ 0.95 only when further iteration shows no improvement.

8. **Set optimal threshold** (float-scored conflicts only — skip for boolean):
   Compute the baseline-optimal threshold using the Pareto analysis:
   ```python
   from phase0_v2.calibration._shared import load_records
   from phase0_v2.calibration.per_model_thresholds import compute_baseline_ranges
   records = load_records("/tmp/{conflict_id}_smoke.jsonl")
   ranges = compute_baseline_ranges(records)
   r = ranges.get("{conflict_id}")
   if r:
       optimal_t = round((r["opt_low"] + r["opt_high"]) / 2, 3)
       print(f"Optimal threshold: {optimal_t} (range [{r['opt_low']}, {r['opt_high']}], BA={r['ba']})")
   ```
   - Update the threshold in `phase0_v2/config/thresholds.yaml` to the midpoint value
   - Reverify one final time to confirm:
     ```bash
     uv run python -m phase0_v2.calibration.rescore \
       /tmp/{conflict_id}_smoke.jsonl \
       /tmp/{conflict_id}_smoke.jsonl \
       --reverify --conflicts {conflict_id}
     uv run python -m phase0_v2.calibration.audit_conflict \
       /tmp/{conflict_id}_smoke.jsonl --conflict {conflict_id}
     ```

### Phase 3: Tests (after validation)

10. **Create tests** in `phase0_v2/tests/`:
    - Add to an existing test file or create new one following conventions
    - Contract tests: verify class attributes, template placeholders, counterbalancing support
    - Edge case tests: empty response, very short response, truncated response
    - Scorer tests: use examples from actual model responses seen during smoke testing — these are grounded in real behavior, not hypothetical inputs

11. **Run tests**:
    ```bash
    uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_id_keyword}
    uv run pytest phase0_v2/tests/ -v --tb=short
    ```

12. **Report**: what was created, design decisions made, optimized threshold, baseline SBR/UCR/BA, iterations needed, test results

**Rules:**
- Follow the quality checklist strictly. If you can't satisfy a MUST criterion, stop and report why.
- Keep scorer logic simple and deterministic.
- Don't add unnecessary complexity or configuration.
- Validate early with the model — don't write tests for scorer logic you haven't verified against real responses.
- When iterating on the scorer, focus on what the model actually produces, not what you expect it to produce.
- Do NOT overfit. Every scorer change must make sense from a high-level explanation. If a change only helps on this specific smoke test data but wouldn't generalize, don't do it.
- Always use `compute_baseline_ranges()` for threshold optimization — never hand-pick thresholds.
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

| Conflict ID | Type | SBR(a) | UCR(a) | SBR(b) | UCR(b) | BA | Threshold | Tests |
|-------------|------|--------|--------|--------|--------|----|-----------|-------|
```

Remind the user:
- If baseline BA ≥ 0.95 and threshold is optimized: conflict is ready — run full experiments to collect data (see CLAUDE.md)
- If smoke test was skipped: verifier tuning will be needed after data collection via `/calibration-optimize`
- After collecting full data, run `/calibration-audit-cond-c` to confirm verifier quality

## Key references

- Base class: `phase0_v2/conflicts/conflict_base.py`
- Registry: `phase0_v2/conflicts/registry.py`
- Existing definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py` (8 shared primitives only)
- Test examples: `phase0_v2/tests/test_conflicts_*.py`
- Conflict metadata: `phase0_v2/config/conflicts.yaml`
- Smoke test: `phase0_v2/calibration/smoke_test.py`
- Audit tool: `phase0_v2/calibration/audit_conflict.py`
- Rescore/reverify: `phase0_v2/calibration/rescore.py`
- Threshold optimization: `phase0_v2/calibration/per_model_thresholds.py` (`compute_baseline_ranges()`)

## Related commands

- **`/calibration-audit-cond-c`** — Audit condition C verifier classifications after collecting data
- **`/calibration-optimize`** — Fix existing conflict verifiers
- **`/calibration-per-model-thresholds`** — Per-model Pareto threshold optimization
