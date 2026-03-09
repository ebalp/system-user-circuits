---
description: "Design and implement new or redesigned conflict definitions for Phase 0 v2. Use when the user wants to create new conflicts, redesign weak ones, fill gaps in the constraint landscape, or expand the conflict set."
---

# Conflict Proposal & Implementation

Design new conflicts or redesign existing weak ones. Handles the full lifecycle: propose, vet, implement, smoke test, and clean up old data.

## Inputs

- `$ARGUMENTS`: conflict idea, conflict ID to redesign, or empty for gap analysis

## Step 0: Determine mode

Decide whether this is a **new conflict** or a **redesign** of an existing one.

- If `$ARGUMENTS` matches an existing registered conflict_id → **redesign mode**
- If `$ARGUMENTS` is a description or idea → **new conflict mode**
- If `$ARGUMENTS` is empty → **gap analysis mode** (proposes both new and redesigns)

For redesign mode, check if a diagnosis report exists for the conflict:
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
3. Set `verify_threshold = 0.5` (or tune after data collection)
4. The base class `_dispatch_verify` handles anti-correlated threshold logic automatically

## Step 3: Design the conflict

### For redesign mode:
Present the old conflict's weaknesses (from calibration data or proposals file) and explain how the redesign addresses them. The new conflict MUST have a **different conflict_id** from the old one.

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

## Step 4: Implement (only with user approval)

For each user-approved conflict, confirm before proceeding. Then launch one Agent per conflict (subagent_type=general-purpose, run_in_background=true).

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

**Your task:**

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
     - `verify_threshold` (for float scorers)

3. **Register** in `phase0_v2/conflicts/registry.py`:
   - Add import in the appropriate section
   - Add to `_ALL_CONFLICT_CLASSES` in alphabetical order by class name
   - If redesign: **remove** the old conflict class from registry (remove its import and list entry)

4. **If redesign — clean up old conflict:**
   - Do NOT delete the old definition file (keep for reference)
   - Add the old conflict_id to `exclude_conflicts` for ALL models in `phase0_v2/config/experiment.yaml`
   - Archive old data from results files:
     ```bash
     uv run python -m phase0_v2.calibration.archive_conflict {old_conflict_id} phase0_v2/data/results/
     ```

5. **Create tests** in `phase0_v2/tests/`:
   - Add to an existing test file or create new one following conventions
   - Contract tests: verify class attributes, template placeholders, counterbalancing support
   - Edge case tests: empty response, very short response, truncated response
   - Scorer tests: known inputs with expected outputs

6. **Run tests**:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short -k {conflict_id_keyword}
   uv run pytest phase0_v2/tests/ -v --tb=short
   ```

7. **Report**: what was created, any design decisions made, test results

**Rules:**
- Follow the quality checklist strictly. If you can't satisfy a MUST criterion, stop and report why.
- Keep scorer logic simple and deterministic.
- Don't add unnecessary complexity or configuration.
- New conflict_id for redesigns — never reuse the old ID.
```

## Step 5: Smoke test & verifier tuning (requires vLLM)

After implementation, if a vLLM server is available, iteratively test and tune the conflict. The goal is **BA > 0.95** — conflicts below this threshold are not useful for the experiment.

### 5a. Initial smoke test

Run the smoke test to get a first BA estimate:

```bash
uv run python -m phase0_v2.calibration.smoke_test \
  --conflict {conflict_id} \
  --vllm-url {vllm_url} \
  --model {model_id}
```

This generates ~24 prompts (3 tasks × 4 conditions × 2 directions), sends them to vLLM, and computes BA with per-condition breakdown.

### 5b. Verifier tuning loop

If BA < 0.95, inspect the smoke test output to understand failures:
- Which conditions fail? (A/B baselines should be near 100% — if not, the verifier is wrong)
- Are scores clustered near the threshold? (threshold tuning may help)
- Are there systematic misclassifications? (scorer logic needs fixing)

Fix the verifier code, then re-run the smoke test. Use `--conditions A,B` with more samples to focus on baseline accuracy:

```bash
uv run python -m phase0_v2.calibration.smoke_test \
  --conflict {conflict_id} \
  --vllm-url {vllm_url} \
  --model {model_id} \
  --conditions A,B \
  --n-tasks 10
```

Repeat until:
- **Conditions A and B**: near 100% correct (verifier accurately detects constraint compliance)
- **Overall BA > 0.95**: conflict is ready for full experiment

If BA stays below 0.95 after 2-3 tuning rounds, the conflict design may be fundamentally flawed — reconsider the constraints or scorer approach before proceeding.

### 5c. No vLLM available

If no vLLM server is available, skip smoke testing and note it as deferred. The conflict will need tuning via `/calibration-optimize` after full data collection, but this is less efficient — prefer smoke testing when possible.

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

| Conflict ID | Replaces | Type | Counterbalancing | Smoke BA | Tests |
|-------------|----------|------|------------------|----------|-------|
```

Remind the user:
- If smoke test BA > 0.95: conflict is ready — run full experiments to collect data (see CLAUDE.md)
- If smoke test was skipped: verifier tuning will be needed after data collection via `/calibration-optimize`
- After collecting full data, run `/calibration-report` to confirm quality
- New conflicts start with `explored: no` — update to `explored: yes` after full calibration confirms BA > 0.95
- If redesigns were done: old data was archived, old conflict_ids added to `exclude_conflicts`

## Key references

- Base class: `phase0_v2/conflicts/conflict_base.py`
- Registry: `phase0_v2/conflicts/registry.py`
- Existing definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py` (8 shared primitives only)
- Test examples: `phase0_v2/tests/test_conflicts_*.py`
- Config: `phase0_v2/config/experiment.yaml`
- Archive tool: `phase0_v2/calibration/archive_conflict.py`
- Smoke test: `phase0_v2/calibration/smoke_test.py`
- Diagnosis reports: `phase0_v2/calibration/output/*/diagnosis/{conflict_id}*.md`

## Related commands

- **`/calibration-report`** — Generate calibration report after collecting data for new conflicts
- **`/calibration-diagnose`** — Diagnose issues with new conflicts after data collection
- **`/calibration-optimize`** — Tune verifiers for new conflicts after data collection
