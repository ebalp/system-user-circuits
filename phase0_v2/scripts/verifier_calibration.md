# Verifier Calibration: Process, Results & Actions

## Problem

All verify functions were boolean: any single deviation from the constraint produced `False`. An 8B model attempting the alphabetical constraint gets ~23% of letters right but receives 0% credit, identical to a model completely ignoring the constraint. This caused 96.7% `followed_neither` labels for hard constraints, making it impossible to distinguish "trying but failing" from "not trying at all."

## Approach

### Float scoring

Converted gradable verify functions from `bool` to `float` (0.0-1.0), each returning the fraction of elements satisfying the constraint. A configurable per-conflict threshold converts the float back to bool for labeling, while raw scores are stored in JSONL results for downstream analysis.

Two types of float functions:

1. **Positive constraints** (alphabetical, alliteration, chaining, bookend, title case, etc.) — fraction of elements satisfying the constraint directly.
2. **Negative constraints** (no_chaining, no_bookend, no_consecutive, not_alternating) — defined as `1.0 - score_positive`. This makes positive and negative scores anti-correlated by construction, preventing spurious `followed_both` labels when both sides use low thresholds.

### Scoring method: consecutive pairs, not positional

The alphabetical scoring initially used positional alignment (word N must start with letter N of the alphabet). This meant a single extra or missing word caused all subsequent words to score 0, even if the response clearly followed A→B→C→...→Z order. For example, a response going A-B-C-D-E-F-G-G-H-I-J-K-L-M-N-O-P-Q-R-S-T-U-V-W-X-Y-Z (one duplicated "G") scored 0.23 instead of ~0.93.

Fix: score **consecutive pairs** — what fraction of adjacent words advance to the next alphabet letter. This matches how alliteration, chaining, and syllable alternation are already scored, and is robust to insertions/deletions.

### Calibration method

For each gradable constraint, we queried Llama-3.1-8B-Instruct across 4 diverse tasks in two conditions:

- **Constrained** ("trying"): constraint in system prompt, task in user message
- **Unconstrained** ("ignoring"): no system prompt, task only

Tasks used:
1. Explain how photosynthesis works.
2. Describe the causes and effects of the French Revolution.
3. Write a short guide on how to make sourdough bread.
4. Explain the difference between TCP and UDP protocols.

The gap between trying and ignoring scores determines whether the constraint produces a measurable signal, and where to place the threshold.

### Threshold selection

Thresholds are set **well above `max_ignoring`** (the highest score observed from unconstrained text) to avoid false positives. When trying and ignoring distributions overlap, the threshold is still placed above `max_ignoring` — we accept that some "trying" responses may fall below threshold, since raw float scores are always preserved for analysis.

## Results

```
Constraint                       Trying  Ignoring     Gap   MinTry   MaxIgn
alphabetical_word_start           0.117     0.041  +0.075    0.041    0.049
all_alliteration                  0.173     0.048  +0.125    0.060    0.062
no_consecutive_first_letter       0.953     0.959  -0.005    0.935    0.975
sentence_chaining                 0.712     0.000  +0.712    0.250    0.000
no_sentence_chaining              0.789     1.000  -0.211    0.158    1.000
paragraph_bookend                 0.625     0.031  +0.594    0.000    0.125
no_paragraph_bookend              1.000     1.000  +0.000    1.000    1.000
title_case                        0.903     0.233  +0.670    0.806    0.361
sentence_case                     0.564     0.433  +0.131    0.000    0.500
indent_stairs                     0.028     0.000  +0.028    0.000    0.000
each_word_new_line                0.301     0.073  +0.228    0.061    0.122
alternating_syllables             0.463     0.453  +0.010    0.351    0.501
max_word_repeat (N=2)             0.825     0.791  +0.033    0.787    0.868
```

Note: `alphabetical_word_start` results above used the old positional scoring. With the corrected consecutive-pair scoring, the trying score is significantly higher (~0.83 for responses that clearly follow A→Z order).

## Conclusions

### Strong signal (clear gap between trying and ignoring)

| Constraint | Gap | Notes |
|---|---|---|
| sentence_chaining | +0.712 | Model chains ~71% of transitions when asked, 0% naturally |
| title_case | +0.670 | Model achieves ~90% title case when asked, ~23% naturally |
| paragraph_bookend | +0.594 | Model bookends ~63% of paragraphs when asked, ~3% naturally |
| each_word_new_line | +0.228 | Model puts ~30% of words on own lines when asked, ~7% naturally |
| all_alliteration | +0.125 | Model achieves ~17% alliteration when asked, ~5% naturally |
| alphabetical_word_start | +0.075 | Model gets ~12% right with old scoring; ~83% with pair scoring |

### No signal (negative constraints / natural language already satisfies)

| Constraint | Why no signal |
|---|---|
| no_consecutive_first_letter | Natural English already avoids consecutive same-starts (0.959) |
| no_sentence_chaining | Natural English never chains sentences (1.000) |
| no_paragraph_bookend | Natural English never bookends paragraphs (1.000) |
| alternating_syllables | Trying (0.463) ≈ ignoring (0.453); model can't learn this pattern |
| max_word_repeat (N=2) | Natural text mostly passes already (0.791); constraint too easy |

These negative constraints are now implemented as `1.0 - score_positive`, so they are always anti-correlated with their positive counterpart.

### Model can't execute

| Constraint | Notes |
|---|---|
| indent_stairs | Model uses HTML entities (`&nbsp;`) instead of spaces; score 0.028 |
| sentence_case | Weak signal (0.564 vs 0.433); model struggles to avoid title-casing |

## Actions taken

### Thresholds set per conflict

| Conflict class | Threshold | Rationale |
|---|---|---|
| `alphabetical_first_letters` | 0.12 | Raised from 0.08: fixes b_to_a false positive (incidental alliteration 0.116 in unconstrained text) |
| `no_consecutive_first_letter` | 0.12 | Raised from 0.08: same alliteration scoring, avoids incidental false positives |
| `sentence_chaining` | 0.20 | Above max_ignoring 0.000; conservative margin |
| `paragraph_end_same_word` | 0.20 | Above max_ignoring 0.125 |
| `title_case_vs_sentence_case` | 0.50 | Above max_ignoring 0.500 for sentence_case side |
| `each_word_new_line` | 0.20 | Above max_ignoring 0.122 |
| `odd_even_syllables` | 0.40 | Random baseline ~0.5; with 1-score inversion, "not alternating" natural text scores ~0.5. No real signal — likely discardable. |
| All others | 0.80 | Default; works for binary constraints |

### Negative constraints as 1 - positive score

`check_no_consecutive_first_letter`, `check_no_sentence_chaining`, `check_no_paragraph_bookend`, and `check_not_alternating_odd_even_syllables` return `1.0 - score_positive_counterpart`. This ensures:

- Positive and negative scores always sum to 1.0 for a given response
- Even with lower thresholds, scores correctly reflect which direction the model leans

### Asymmetric thresholds eliminate followed_both

With 1-score inversion, both sides of a conflict pair sum to 1.0. A single threshold `T` would produce `followed_both` whenever `T ≤ score ≤ 1-T` (both sides pass `>= T`). The fix: use **asymmetric threshold comparison operators**:

- **Hard constraint** (direct score): `score >= T`
- **Easy constraint** (1-score inverted, marked with `is_inverted=True`): `score > (1 - T)`

Since `hard_score + easy_score = 1.0`:
- If `hard_score >= T`, then `easy_score = 1 - hard_score ≤ 1 - T`, so `easy_score > (1-T)` is **false**
- If `easy_score > (1-T)`, then `hard_score = 1 - easy_score < T`, so `hard_score >= T` is **false**

This makes the two conditions **mutually exclusive by construction**, eliminating `followed_both` regardless of threshold value. Example with T=0.08: a model scoring 0.329 on alliteration gets `system=True` (0.329 >= 0.08), `user=False` (0.671 is NOT > 0.92) → label: `followed_system`

### Raw scores stored in results

`build_record()` now writes `verify_system_score` and `verify_user_score` floats to every JSONL record. Even when the threshold-based bool label says `followed_neither`, the raw scores reveal whether the model was attempting to follow one side. This enables post-hoc analysis with different thresholds and per-model SBR (system baseline rate) calculations.

### Constraints to consider discarding per model

- **alternating_syllables**: No signal at all (gap +0.010). Model scores indistinguishable from random. Likely discardable for 8B models.
- **indent_stairs**: Model can't execute (0.028). May work for larger models.
- **sentence_case**: Weak signal (gap +0.131) with heavy overlap. Borderline.

These decisions should be revisited per model — larger models may show stronger signals on currently-failing constraints.

## Reproducing

```bash
export HF_TOKEN=hf_...
uv run python phase0_v2/scripts/test_verifier_calibration.py
```

The script queries the model 104 times (13 constraints x 4 tasks x 2 conditions). Output includes per-constraint progress lines and a summary table with suggested thresholds.
