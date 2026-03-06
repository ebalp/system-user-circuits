---
description: "Design and implement new conflict definitions for Phase 0 v2. Use when the user wants to create new conflicts, fill gaps in the constraint landscape, propose new behavioral tests, or expand the conflict set. Includes quality checklist based on calibration learnings."
---

# Conflict Proposal & Implementation

Design new conflict definitions based on gap analysis and quality principles learned from calibration. This command helps propose, vet, and implement new conflicts.

## Inputs

Optional:
- **Conflict idea** from the user (description of what they want to test)
- If no idea provided, this command analyzes the landscape and proposes based on gaps

If `$ARGUMENTS` is provided, treat it as the conflict idea description.

## Step 1: Analyze the conflict landscape

Review all registered conflicts to understand coverage:

```bash
uv run python -c "
from phase0_v2.conflicts.registry import get_all_conflicts
for c in get_all_conflicts():
    print(f'{c.conflict_id}: cb={c.counterbalance_quality}, args={c.arg_keys}')
"
```

Extract descriptions for type/category context:
```bash
awk '/# <description>/,/# <\/description>/{print FILENAME": "$0}' phase0_v2/conflicts/definitions/*.py
```

Categorize existing conflicts by type:
- **Language**: language_en_es, bilingual_english_plus
- **Format**: format_json_yaml, json_only_vs_plain, template_response
- **List/Structure**: list_bullets_vs_numbered, bullets_and_sub_bullets, numbered_sections_vs_prose, short_paragraphs_vs_single_block, stairs_indent, each_word_new_line
- **Style**: capitalization_all_caps, title_case_vs_sentence_case, formal_vs_casual_tone, first_vs_third_person, active_vs_passive_voice, direct_answer_vs_hedging
- **Content**: emoji_use_vs_avoid, disclaimer_add_vs_none, self_reference_ai_mention, starting_word_hello_greetings, questions_vs_statements
- **Word/Count**: forbidden_words, keyword_exact_count, keyword_in_nth_sentence, min_unique_words, min_pronoun_count, word_count_range, max_sentence_length, exact_number_count, max_word_repeat
- **Pattern**: alphabetical_first_letters, paragraph_end_same_word, sentence_chaining, no_consecutive_first_letter, odd_even_syllables, repeat_answer_twice, italics_thesis

Identify underrepresented categories or missing dimensions.

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

## Step 3: Propose conflicts

If the user provided an idea, evaluate it against the quality checklist and refine it.

If no idea, propose 3-5 new conflicts with:
- **ID**: snake_case name following existing conventions
- **Category**: which type category it fills
- **Gap rationale**: why this is needed (what dimension is underrepresented)
- **Constraint A / B**: the two opposing instructions
- **Scorer approach**: how verification would work
- **Quality assessment**: checklist pass/fail for each criterion
- **Predicted difficulty**: easy/medium/hard based on model capability expectations

Present proposals and ask the user which to implement (if any).

## Step 4: Implement (only with user approval)

For each user-approved conflict, confirm before proceeding. Then launch one Agent per conflict (subagent_type=general-purpose, run_in_background=true).

**Agent prompt template:**

```
You are implementing a new conflict definition for the Phase 0 v2 experiment system.

**Conflict spec:**
- ID: {conflict_id}
- Constraint A (system): {constraint_a}
- Constraint B (user): {constraint_b}
- Type: {bool_or_float}
- Scorer approach: {scorer_description}

**Your task:**

1. **Read reference files** to understand conventions:
   - `phase0_v2/conflicts/conflict_base.py` (base class, all fields)
   - `phase0_v2/conflicts/verify_utils.py` (shared scorer utilities)
   - One existing similar conflict for structural reference

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
   - Scorer function(s) -- put shared logic in verify_utils.py if reusable
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
   - Add import in the appropriate batch section
   - Add to `_ALL_CONFLICT_CLASSES` in alphabetical order by class name

4. **Check compatibility** -- if the conflict could interfere with existing ones (e.g., both constrain formatting), note it for the compatibility matrix

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
- Use existing verify_utils functions where applicable.
- Don't add unnecessary complexity or configuration.
```

## Step 5: Post-implementation

After all agents complete:

1. Run full test suite:
   ```bash
   uv run pytest phase0_v2/tests/ -v --tb=short
   ```
2. Fix any failures

Present a summary of what was implemented:

```markdown
## New conflicts created

| Conflict ID | Type | Counterbalancing | Tests |
|-------------|------|------------------|-------|
```

Remind the user:
- Run experiments to collect data: see CLAUDE.md for experiment running instructions
- After collecting data, run `/calibration-report` to assess the new conflicts
- New conflicts start with `explored: no` -- use `/calibration-optimize` after data collection to tune them

## Key references

- Base class: `phase0_v2/conflicts/conflict_base.py`
- Registry: `phase0_v2/conflicts/registry.py`
- Existing definitions: `phase0_v2/conflicts/definitions/*.py`
- Scorer utilities: `phase0_v2/conflicts/verify_utils.py`
- Test examples: `phase0_v2/tests/test_conflicts_*.py`
- Config: `phase0_v2/config/experiment.yaml`

## Related commands

- **`/calibration-report`** -- Generate calibration report after collecting data for new conflicts
- **`/calibration-diagnose`** -- Diagnose issues with new conflicts after data collection
- **`/calibration-optimize`** -- Tune verifiers for new conflicts after data collection
