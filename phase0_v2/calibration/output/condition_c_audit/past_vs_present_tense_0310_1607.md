# Condition C Audit: past_vs_present_tense

**Date:** 2026-03-10
**Threshold:** 0.692 (cross-model intersection midpoint)
**Type:** float (inverted pair: past_score / (past + present))

## Architecture

- **Scorer:** `score_past_tense(text)` returns `past_count / (past_count + present_count)`, where:
  - **Past indicators:** irregular past forms (~50 words) + regular `-ed` endings (>= 4 chars, with exception list)
  - **Present indicators:** present be/aux (`is`, `are`, `am`, `do`, `does`, `has`, `have`) + contraction stems + base-form verbs (~45) + 3rd-person forms (~200+ precomputed)
- **Inverted pair:** `_score_present_tense = 1 - score_past_tense`. Threshold T=0.692 means `past >= 0.692` = past, `present >= 0.692` (i.e., past <= 0.308) = present.
- **Mutual exclusivity:** Mathematically guaranteed (complementary scores, T > 0.5). Confirmed: 0 "both high" cases across all models.
- **Ambiguous band:** scores in (0.308, 0.692) -- a wide 0.384 range.

## Condition C Classification Summary

| Model | Direction | Followed System | Followed User | Ambiguous | Total |
|-------|-----------|----------------|---------------|-----------|-------|
| 8B | a_to_b (sys=past) | 292 (23.4%) | 527 (42.2%) | 431 (34.5%) | 1250 |
| 8B | b_to_a (sys=present) | 136 (10.9%) | 985 (78.8%) | 129 (10.3%) | 1250 |
| 70B | a_to_b (sys=past) | 54 (4.3%) | 1039 (83.1%) | 157 (12.6%) | 1250 |
| 70B | b_to_a (sys=present) | 0 (0.0%) | 1250 (100.0%) | 0 (0.0%) | 1250 |
| Gemma-27B | a_to_b (sys=past) | 377 (30.2%) | 513 (41.0%) | 360 (28.8%) | 1250 |
| Gemma-27B | b_to_a (sys=present) | 166 (13.3%) | 868 (69.4%) | 216 (17.3%) | 1250 |

## Key Findings

### 1. MAJOR: Strong direction asymmetry in condition C

All three models show massive direction asymmetry:
- **b_to_a direction** (system=present, user=past): models overwhelmingly produce past tense (78-100% "followed user")
- **a_to_b direction** (system=past, user=present): more mixed, with high ambiguous rates (12-34%)

This asymmetry suggests a **past-tense bias** in these models -- when user requests past tense, models comply much more readily than when system requests it.

**Condition D (recency control) confirms this is partly genuine behavior:**
- 8B Cond D: both directions show ~75% past-tense mean (past bias regardless of placement)
- 70B Cond D: a_to_b mean=0.500, b_to_a mean=0.844 (user position effect, but also past bias)

**Verdict:** The direction asymmetry is primarily a genuine behavioral signal, not a verifier artifact. Models have a natural affinity for past tense, especially for factual/historical topics.

### 2. MODERATE: High ambiguous rate, especially for 8B and Gemma-27B

- 8B: 560/2500 (22.4%) ambiguous across condition C
- Gemma-27B: 576/2500 (23.0%)
- 70B: 157/2500 (6.3%)

The wide ambiguous band (0.308 to 0.692) means that many genuinely mixed-tense responses cannot be classified. Manual inspection of ambiguous cases shows they are **truly mixed** -- the model blends past and present tense within a single response, which is a valid "neither fully followed" state.

### 3. MODERATE: -ed adjective false positives inflate past-tense counts

Words like "organized", "focused", "designed", "balanced", "used", "based", "connected", "united", "structured" are counted as past-tense verbs but frequently function as adjectives. The adversarial test confirms:

- "The focused, organized approach is based on a structured, balanced methodology..." scores 0.833 (past) despite being written in present tense with "is" and "remains" as the actual verbs.

**Impact quantified:**
- 8B: 1218 records affected, **204 would change classification label** if adjective-ed words were excluded
- 70B: 712 affected, **65 would change label**
- Gemma-27B: 1151 affected, **158 would change label**

This is a real but bounded issue (~8% of records would flip). The most impactful false-positive words are: `used`, `based`, `called`, `designed`, `organized`, `focused`, `balanced`, `united`, `connected`, `located`.

### 4. MINOR: Present perfect ambiguity

"The internet has been developing. Scientists have discovered many things." scores 0.400 -- the auxiliary "has/have" counts as present, while the participle "-ed" counts as past. This is linguistically correct behavior for a ratio-based scorer (present perfect is genuinely between tenses), but it means present-perfect-heavy responses reliably fall in the ambiguous band.

### 5. LOW: Baseline performance

Baselines (conditions A/B) show the verifier works well for clear tense:
- Cond A a_to_b (system=past only): 100% correctly above threshold for all models
- Cond B b_to_a (past only): 100% above threshold

However, present-tense baselines are weaker:
- Cond A b_to_a (system=present only, 8B): only 60% classified as present (>= T). 20/50 fail, with scores in the 0.32-0.59 range.
- This appears partly due to -ed adjectives and present perfect constructions inflating past counts even when the model attempts present tense.

### 6. Topic sensitivity (8B only)

For 8B, historical topics (french_revolution, world_war_2, etc.) have a mean past-tense score of 0.760 vs 0.557 for non-historical topics (+0.203 difference). This effect is negligible for 70B and Gemma-27B (<0.02 difference), suggesting 8B has more difficulty maintaining present tense for inherently historical content.

## Score Distributions (Condition C)

| Model | [0.0-0.3) | [0.3-0.5) | [0.5-0.7) | [0.7-1.0] |
|-------|-----------|-----------|-----------|-----------|
| 8B | 634 (25%) | 437 (17%) | 163 (7%) | 1266 (51%) |
| 70B | 1017 (41%) | 171 (7%) | 8 (0.3%) | 1304 (52%) |
| Gemma-27B | 636 (25%) | 450 (18%) | 175 (7%) | 1239 (50%) |

70B shows a bimodal distribution (strong past or strong present), while 8B and Gemma-27B have more mass in the middle.

## Severity Assessment

| Finding | Severity | Impact on Condition C validity |
|---------|----------|-------------------------------|
| Direction asymmetry | Informational | Genuine behavior, not a verifier bug |
| High ambiguous rate | Moderate | 22% of data unclassifiable; reduces effective sample size |
| -ed adjective false positives | Moderate | ~8% label flips; systematically inflates past-tense scores |
| Present perfect ambiguity | Minor | Contributes to ambiguous band but is linguistically reasonable |
| Baseline present-tense weakness | Moderate | 40% baseline failure (8B present) undermines calibration |

## Recommendations

1. **Expand the -ed exception list** to include common participial adjectives: `used`, `based`, `called`, `organized`, `focused`, `designed`, `balanced`, `structured`, `connected`, `united`, `located`, `advanced`, `specialized`, `integrated`, `dedicated`, `limited`, `related`, `combined`, `improved`, `increased`, `reduced`, `decreased`, `expected`, `required`, `supposed`, `considered`, `experienced`, `interested`, `detailed`, `automated`, `isolated`, `associated`, `fixed`, `mixed`, `named`, `alleged`, `complicated`, `sophisticated`, `unlimited`. This would require POS-tag disambiguation or a heuristic (e.g., only count -ed words that follow a subject/aux pattern).
2. **Consider POS tagging** (e.g., spaCy or NLTK POS tagger) to distinguish "the organized team" (adjective) from "she organized the files" (verb). This would be a significant architecture change but would materially improve accuracy.
3. **Accept the current state** if the ~8% label-flip rate is tolerable for the research questions being investigated. The verifier's core design (ratio of past vs present indicators) is sound; the noise is at the margins.

## Verdict

**PASS with caveats.** The verifier correctly identifies clear past-tense and present-tense responses. The main weaknesses are: (a) participial adjectives counted as past tense (~8% label impact), and (b) wide ambiguous band reducing classifiable sample size by ~22% for 8B/Gemma. The direction asymmetry observed in condition C is a genuine behavioral signal, not a verifier artifact. The conflict is usable for measuring instruction-following at the population level, but individual-response classification near the threshold boundary has meaningful noise.
