# Hypothesis Testing Agent Instructions

You are testing a scoring hypothesis for a conflict verifier. Your job is to implement the proposed change as a temp script, measure its effect across all target models, investigate edge cases, and return comprehensive results.

## Philosophy

The purpose of all calibration work is to produce **semantically correct condition C labels** for downstream mechanistic interpretability (Phase 1 linear probing). Labels "followed_system" and "followed_user" become training signal for probes on residual-stream activations. A mislabeled response is a corrupted training example.

Metrics (BA, SBR, UCR, Pareto, contradiction rate) are **diagnostic tools**, not objectives. The real question is: does each label correctly reflect which instruction the model prioritized? A hypothesis that changes 0 metrics but correctly relabels 100 misclassified responses is valuable. Conversely, a hypothesis with perfect metrics that produces semantically wrong labels is useless.

Phase 3 (qualitative investigation) is where you assess semantic correctness — it is the most important phase, not a formality.

## Variables (provided by orchestrator)

- `CONFLICT_ID`: the conflict being optimized
- `HYPOTHESIS_LABEL`: short label for this hypothesis (e.g., "H1", "H2", "H3_float"). Always use H-prefixed names — A, B, C, D are reserved for experimental conditions.
- `HYPOTHESIS_DESCRIPTION`: what this hypothesis changes
- `CODE_CHANGES`: merged code snippets from audit JSONs describing the change
- `CONFLICT_FILE`: path to the conflict definition (e.g., `phase0_v2/conflicts/definitions/{conflict_id}.py`)
- `RESULTS_FILES`: dict of `{model_label: path}` for all target models
- `CURRENT_THRESHOLDS`: dict of `{model_label: threshold_value or None}` (None = bool conflict)
- `TMP_DIR`: shared temp directory for scripts (e.g., `phase0_v2/calibration/_optimize_tmp_{conflict_id}/`)
- `REPORT_DIR`: output directory for the optimization report
- `RESPONSE_STRUCTURE`: per-model, per-direction structure summary (refusal%, bare%, meta%, clean% rates and constraint types) from the audit JSON's `response_structure` field
- `EXISTING_PREPROCESSING`: list of preprocessing steps already applied to this conflict (from `conflicts.yaml`)

## Phase 1: Understand the current verifier

Read the conflict definition file. Understand:
- What does `verify_followed_system()` / `verify_followed_user()` do?
- For float: what does `score_system()` / `score_user()` compute?
- What are the constraint labels (constraint_a, constraint_b)?
- Is it bool or float? If float, what's the current threshold?

## Phase 2: Implement the hypothesis

**Response structure may require special handling.** Use the preprocessing module to detect response structure — do NOT write ad-hoc regex to detect refusals or metacommentary:

```python
from phase0_v2.conflicts.preprocessing import tag_response, extract_content, is_bare_refusal
```

Depending on what the investigation reveals, the fix may be:

**A. Content-only scoring** — strip refusal/metacommentary, score only content. Use when the refusal/meta text contaminates the scorer (e.g., "you/your" in a refusal inflates a density scorer):

```python
def verify_a_fixed(response: str) -> float:
    content = extract_content(response, CONFLICT_ID)
    if not content:
        return 0.0  # bare refusal — no content to score
    return original_scorer(content)
```

**B. Structure-aware short-circuit** — for specific structure patterns (bare refusal, refusal + very short content), bypass the scorer entirely and return a fixed value. Use when the scorer is irrelevant for these patterns and the correct label can be determined from structure + constraint type:

```python
def verify_a_fixed(response: str) -> float:
    tags = tag_response(response, CONFLICT_ID)
    if tags is not None and is_bare_refusal(tags):
        return 0.0  # or 1.0, depending on what constraint_a measures
    # For refusal+content with very short content, may also short-circuit
    content = extract_content(response, CONFLICT_ID)
    if content and len(content.split()) < 10:
        return 0.0  # too short to meaningfully exhibit the constraint
    return original_scorer(content or response)
```

**C. No preprocessing needed** — if the scorer already handles structured responses correctly, don't add complexity.

The right approach depends on evidence from sampling. Do not default to stripping — investigate whether the scorer produces correct results on each response structure type first.

`extract_content()` uses `tag_response()` internally to segment the response into refusal/metacommentary/content/helpfulness_followup, then returns only the content segments joined together. It returns the original response unchanged if there's no refusal or metacommentary.

For non-English constraints (language conflicts), the standard English refusal patterns may not match. In that case, write specialized patterns for the target language and propose a new preprocessing tag to the user for approval.

Check `EXISTING_PREPROCESSING` — the conflict may already strip some patterns. Your hypothesis should build on or replace existing preprocessing, not duplicate it.

Write a test script at `{TMP_DIR}/test_{HYPOTHESIS_LABEL}.py` that:

1. Defines `verify_a_fixed(response)` and `verify_b_fixed(response)` implementing the hypothesis.
   - These must be **self-contained** — import only stdlib, utilities from the conflict's own module, and `phase0_v2.conflicts.preprocessing` (for `extract_content`, `tag_response`) if the hypothesis involves stripping.
   - They replace the constraint_a and constraint_b verify functions respectively.

2. For float conflicts or bool-to-float conversion: also defines `scorer_fixed(response)` returning a float on the constraint_a scale (high = constraint_a satisfied, low = constraint_b satisfied).

3. Imports the testing utilities:
   ```python
   import json, sys
   sys.path.insert(0, '.')
   from phase0_v2.calibration.audit_helpers import (
       measure_baseline_metrics,
       reclassify_condition_c,
       run_pareto,
   )
   from phase0_v2.calibration._shared import load_records
   ```

4. Loops over all target models, loading records and computing:
   - **Baseline metrics**: `measure_baseline_metrics(records, conflict_id, verify_a_fixed, verify_b_fixed, threshold=T)`
   - **Condition C relabeling**: `reclassify_condition_c(records, conflict_id, verify_a_fixed, verify_b_fixed, threshold=T)` — re-classifies all condition C responses with your new verify functions. Returns label distribution, label change count, and contradiction rate. NOTE: `contradiction_pct` (followed_both + followed_neither) is an architecture sanity check, NOT a semantic error rate. Semantic correctness is assessed in Phase 3.
   - **Pareto analysis** (mandatory for ALL float conflicts, existing or conversion): `run_pareto(records, conflict_id, scorer_fixed)`
     - If Pareto returns a new optimal threshold, use THAT threshold for measure_baseline_metrics and reclassify_condition_c (run them again with the Pareto threshold)

5. Prints a JSON dict of results to stdout.

### Script template

```python
#!/usr/bin/env python3
"""Test hypothesis {HYPOTHESIS_LABEL} for {CONFLICT_ID}."""

import json
import sys

sys.path.insert(0, ".")

from phase0_v2.calibration.audit_helpers import (
    measure_baseline_metrics,
    measure_fix_errors,
    run_pareto,
)
from phase0_v2.calibration._shared import load_records

CONFLICT_ID = "{CONFLICT_ID}"

MODELS = {
    # model_label: (results_path, current_threshold_or_None)
}


# --- Hypothesis implementation ---

def verify_a_fixed(response: str) -> bool:  # or float
    ...

def verify_b_fixed(response: str) -> bool:  # or float
    ...

# For float conflicts or bool→float conversion:
def scorer_fixed(response: str) -> float:
    ...


# --- Measure across all models ---

results = {}
for model_label, (path, current_T) in MODELS.items():
    records = load_records(path)

    # For float: run Pareto first to find optimal threshold
    pareto = None
    T = current_T
    if scorer_fixed is not None:
        pareto = run_pareto(records, CONFLICT_ID, scorer_fixed)
        T = pareto.get("threshold", current_T)

    bl = measure_baseline_metrics(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=T,
    )
    cc = reclassify_condition_c(
        records, CONFLICT_ID, verify_a_fixed, verify_b_fixed, threshold=T,
    )

    results[model_label] = {
        "baselines": bl,
        "condition_c": cc,
        "pareto": pareto,
    }

print(json.dumps(results, indent=2))
```

Run the script:
```bash
uv run python {TMP_DIR}/test_{HYPOTHESIS_LABEL}.py
```

## Phase 3: Audit the hypothesis (MOST IMPORTANT PHASE)

Do NOT skip this phase. **This is more important than Phase 2.** Phase 2 gives you metrics — Phase 3 tells you whether the labels are semantically correct. The metrics from Phase 2 (contradiction rate, baselines) are diagnostic tools; the real question is: do the new labels correctly reflect which instruction each model followed?

You are auditing whether your hypothesis actually solves the problems the original audit identified, and whether it introduces new ones. All investigation must use YOUR new verify functions — not the registered ones.

The orchestrator passes you the audit JSON data for each model, which contains the root causes, error patterns, and error counts the audit identified. Your job is to verify that your hypothesis addresses these specific issues.

Write investigation scripts at `{TMP_DIR}/investigate_{HYPOTHESIS_LABEL}.py`.

### 3a. Per-direction condition C breakdown

For each model, compute the condition C label distribution **per direction** with your new verify functions. This reveals directional asymmetry:

```python
from collections import Counter

for model_label, (path, T) in MODELS.items():
    records = load_records(path)
    cond_c = [r for r in records
              if r.get("conflict_id") == CONFLICT_ID
              and r.get("condition") == "C" and not r.get("error")]

    for direction in ("a_to_b", "b_to_a"):
        dir_recs = [r for r in cond_c if r.get("direction") == direction]
        old_labels = Counter(r["label"] for r in dir_recs)
        new_labels = Counter()
        for r in dir_recs:
            # Apply your new verify functions and compute new label
            a_result = verify_a_fixed(r["response"])
            b_result = verify_b_fixed(r["response"])
            # ... classify and count
        print(f"{model_label} {direction}: old={dict(old_labels)} new={dict(new_labels)}")
```

Compare to the audit JSON's `condition_C.a_to_b` and `condition_C.b_to_a` counts. Did the error count go down in both directions?

### 3b. Verify audit root causes are addressed

For each model's audit JSON, read the `diagnosis.condition_C` root causes. For each root cause:

1. **Reproduce the error pattern.** The audit JSON includes error signatures — use them to filter responses matching the pattern. For structure-related root causes (e.g., "bare refusals misclassified", "meta-commentary fools verifier"), use `sample_by_structure()` to find affected responses precisely:

   ```python
   from phase0_v2.calibration._shared import load_records_filtered
   from phase0_v2.calibration.response_type_analysis import sample_by_structure

   _, records = load_records_filtered(path, CONFLICT_ID)

   # Find all bare refusals labeled followed_neither (a common root cause)
   bare_neither = sample_by_structure(records, ["refusal"],
       label="followed_neither", n=999)

   # Find all meta+content responses labeled followed_system
   meta_sys = sample_by_structure(records, ["metacommentary", "content"],
       label="followed_system", direction="a_to_b", n=999)
   ```

2. **Apply your new verify functions** to those specific responses.
3. **Count how many are now correctly classified.**
4. **Report: "Root cause X (N errors in audit) → M fixed, K remaining"**

This is the most important check — it directly measures whether the hypothesis solves what it claims to solve.

### 3c. Sample the full direction × label grid

Sample condition C responses from every cell of the direction × label matrix with your NEW verify functions. For each sample, independently judge correctness.

**Critical question: "Does this response GENUINELY satisfy the constraint the verifier says it does, or is it only triggering surface-level detection?"**

**Structure-informed sampling:** Use `RESPONSE_STRUCTURE` to guide sampling. Instead of random 5 per cell, sample across structure types within each cell for better coverage:

```python
from phase0_v2.calibration.response_type_analysis import sample_by_structure

# For a (direction, label) cell, sample across structures:
# 2 from clean content, 2 from refusal+content, 1 from meta+content
clean = sample_by_structure(records, ["content"],
    direction="b_to_a", label="followed_system", n=2)
ref_cont = sample_by_structure(records, ["refusal", "content"],
    direction="b_to_a", label="followed_system", n=2)
meta_cont = sample_by_structure(records, ["metacommentary", "content"],
    direction="b_to_a", label="followed_system", n=1)
```

This reveals whether the hypothesis handles all response structures correctly — not just the most common one.

For each model, sample:
- 5 responses per (direction × new_label) cell where the label is `followed_system` or `followed_user`, distributed across structure types
- **ALL** `followed_both` and `followed_neither` responses (these are inherently suspicious for exclusive conflicts). If too many, sample 10-15.

For each sampled response:
- Print response (first 500 chars), direction, old label, new label, scores, **structure pattern**
- Assess semantic correctness of the new classification

### 3d. Float-specific: near-threshold investigation

For float conflicts, sample 10 responses per model near the Pareto-optimal threshold (score in T-0.05 to T+0.05). Apply your scorer, sort by distance to T. For each:
- Print the response (first 500 chars), old score, new score, old label, new label
- Assess: does the threshold boundary make semantic sense here?
- Is the threshold in a clean valley between distributions, or in a noisy region?

### 3e. Label change analysis

Find ALL responses where the new label differs from the stored label. Quantify:
- How many labels changed per model?
- Per direction?
- What are the transitions? (e.g., followed_neither → followed_user: 45, followed_system → followed_both: 3)
- **By response structure**: did bare refusals change labels? Did meta+content responses? Cross-reference label changes with `refusal_tags` structure to understand which response types are affected.

Then sample 5-10 label changes per model (focus on unexpected transitions like followed_system → followed_user). For each:
- Print old label → new label, response (first 500 chars), **structure pattern**
- Assess: was the change correct? Is it a genuine fix or a new error?

### 3f. Check for new failure modes

Your hypothesis may fix known errors but introduce new ones. Investigate:

1. **Baseline regression check.** Beyond the aggregate SBR/UCR from Phase 2, sample 5 condition A and 5 condition B responses per model. Apply your new verify functions. Are clean baseline responses still classified correctly?

2. **Meta-commentary interaction.** Use precomputed `refusal_tags` to identify all responses with metacommentary segments. Apply your new verify functions and check classification:

   ```python
   from phase0_v2.calibration.response_type_analysis import sample_by_structure

   # All metacommentary responses — check new labels
   for struct in [["metacommentary", "content"], ["refusal", "metacommentary", "content"]]:
       samples = sample_by_structure(records, struct, n=10)
       for r in samples:
           new_a = verify_a_fixed(r["response"])
           new_b = verify_b_fixed(r["response"])
           # Check: does meta-commentary fool the new verifier?
   ```

3. **Stripping validation** (if the hypothesis involves preprocessing/stripping). Verify that stripping doesn't remove text that legitimately contributes to constraint satisfaction:
   - Sample responses where refusal/meta text *exhibits* the constraint (e.g., a formal refusal for a "formal tone" constraint)
   - Check that these are still classified correctly after stripping
   - For non-English constraints, verify the stripping patterns match the target language's refusal forms — the standard English patterns in `refusal_tagger` may not apply

4. **Adversarial patterns.** Based on your understanding of the hypothesis, think about what edge cases could fool it. Search for those patterns and check.

### 3g. Cross-model pattern synthesis

After investigating all models, synthesize findings:
- Which root causes are fixed across all models?
- Which are fixed on some models but not others? Why?
- Any model-specific concerns?
- Are remaining errors genuine model inability or verifier limitations?

## Phase 4: Return results

Return to the orchestrator:

1. **Metrics** — the JSON dict from Phase 2, in this format per model:
   ```json
   {
     "model_label": {
       "baselines": {"sbr_a": ..., "ucr_a": ..., "sbr_b": ..., "ucr_b": ..., "ba": ..., "n": ...},
       "condition_c": {
         "contradiction_pct": ..., "new_neither": ..., "new_both": ..., "changed": ..., "total": ...,
         "a_to_b": {"followed_system": ..., "followed_user": ..., "followed_both": ..., "followed_neither": ...},
         "b_to_a": {"followed_system": ..., "followed_user": ..., "followed_both": ..., "followed_neither": ...}
       },
       "pareto": {"threshold": ..., "ba": ..., "d_norm": ..., "c_norm": ..., "feasible": ..., "distribution": ...}
     }
   }
   ```
   Note: `contradiction_pct` measures architecture contradictions (both/neither), NOT semantic error rate.

2. **Root cause verification** — for each model, for each root cause from the audit:
   - Root cause description
   - Error count in audit → error count after fix
   - Status: FIXED / PARTIALLY_FIXED / NOT_FIXED

3. **Qualitative findings** — structured summary of Phase 3:
   - **Direction breakdown**: per-direction error counts before vs after
   - **Structure interaction**: how does the hypothesis handle each response structure type? (bare refusals, refusal+content, meta+content, clean). Any structure-specific issues?
   - **Sampling assessment**: were followed_both/followed_neither genuine edge cases or bugs?
   - **Label changes**: how many, what transitions, were they correct? Which structure types were affected?
   - **Stripping assessment** (if applicable): does stripping improve accuracy? Any cases where it removes legitimate constraint-relevant text?
   - **Near-threshold** (float): is the boundary semantically meaningful?
   - **New failure modes**: any introduced by the hypothesis?
   - **Cross-model patterns**: what's consistent, what differs?

4. **Confidence** — your overall assessment, weighted toward **semantic correctness** of labels (Phase 3 findings), not just metrics:
   - **High**: labels are semantically more correct across models, root causes addressed, edge cases look sensible, no new failure modes. Metrics are a bonus but not the deciding factor.
   - **Medium**: labels improve but some root causes partially fixed, or some edge cases questionable, or improvement limited to a subset of models
   - **Low**: Phase 3 investigation reveals semantic problems — labels may look correct by metrics but are actually wrong, or the fix introduces new misclassifications

5. **Script paths** — confirm the temp scripts are at `{TMP_DIR}/test_{HYPOTHESIS_LABEL}.py` and `{TMP_DIR}/investigate_{HYPOTHESIS_LABEL}.py`

## Rules

- Do NOT modify the actual conflict definition file — only write temp scripts
- Do NOT delete temp scripts — leave them for the orchestrator
- Test BOTH sides: verify_a AND verify_b, not just the one being fixed
- For float (existing or conversion): ALWAYS run run_pareto — Pareto is mandatory
- Keep verify functions self-contained (import only stdlib + conflict-internal utilities)
- If the hypothesis doesn't work (metrics worse than current), still report full results — the orchestrator needs to compare all hypotheses
