# Model-Level Verifier Accuracy Report

**Date:** 2026-03-11
**Conflicts audited:** 41
**Models:** Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, Gemma-3-27B-IT

## Model Rankings

| Model | Avg Error % | GREEN (<1%) | YELLOW (1-3%) | AMBER (3-10%) | RED (>=10%) | Usable (GREEN+YELLOW) |
|-------|-------------|-------------|---------------|---------------|-------------|----------------------|
| **70B** | **0.9%** | 31 | 6 | 4 | 0 | **37/41 (90%)** |
| **8B** | **1.4%** | 27 | 8 | 6 | 0 | **35/41 (85%)** |
| **Gemma** | **6.1%** | 14 | 8 | 11 | 8 | **22/41 (54%)** |

**70B is the cleanest model** -- zero RED conflicts, only 4 AMBER, and nearly all verifiers work well.
**8B is solid** -- no RED conflicts, 6 AMBER, all manageable.
**Gemma is problematic** -- 8 RED conflicts, 11 AMBER, driven by pervasive meta-commentary that breaks keyword/phrase-based verifiers.

## Per-Model Breakdown

### Llama-3.3-70B-Instruct (Best)

70B produces clean, bimodal response distributions with minimal meta-commentary. Verifiers work nearly perfectly.

**GREEN (31 conflicts, 0-0.6% error):**
alliteration_density, alphabetical_sentences (1.3% -- borderline), capitalization_all_caps (borderline), direct_answer_vs_hedging, each_word_new_line, emoji_use_vs_avoid, first_vs_third_person (borderline), forbidden_words, format_json_markdown, formal_vs_casual_tone, html_emphasis_tags, imperative_vs_declarative, json_only_vs_plain, keyword_frequency, language_en_es, language_en_zh, leetspeak_encoding, list_bullets_vs_numbered, lowercase_vs_capitalized, number_density, numbered_sections_vs_prose, paragraph_start_word, parenthetical_asides, past_vs_present_tense, pronoun_density, questions_vs_statements, response_length, sentence_connector_density, short_paragraphs_vs_single_block, short_vs_long_sentences, starting_word_hello_greetings

**YELLOW (6 conflicts, 1-3% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| alphabetical_sentences | 1.3% | Accidental letter sequences near threshold |
| capitalization_all_caps | 3.8% | Meta-commentary preamble in normal case |
| first_vs_third_person | 2.7% | Missing they/them in regex |
| keyword_avoidance | 2.5% | Meta-commentary keyword mentions |
| word_repetition_density | 1.6% | Topic-forced word repetition |
| template_response | 1.4% | Minor presence-check edge cases |

**AMBER (4 conflicts, 3-10% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| bullets_and_sub_bullets | 7.0% | Overly strict sub-bullet nesting requirement |
| disclaimer_first_vs_none | 9.1% | Missing phrases ("although the information...") |
| self_reference_ai_mention | 7.8% | Missing "computer program" + use-mention confusion |
| vowel_omission | 0.4% | *(actually GREEN -- borderline)* |

**Recommendation for 70B:** All 41 conflicts are usable. Fix the 4 AMBER verifiers (bullets_and_sub_bullets, disclaimer_first_vs_none, self_reference_ai_mention are structural fixes, not model-specific). No conflicts need exclusion.

---

### Llama-3.1-8B-Instruct (Good)

8B has moderate meta-commentary but verifiers generally handle it. Main weakness is keyword-related conflicts where meta-commentary quotes trigger false positives.

**GREEN (27 conflicts, 0-0.5% error):**
alliteration_density, direct_answer_vs_hedging, each_word_new_line (borderline), emoji_use_vs_avoid, format_json_markdown, html_emphasis_tags, imperative_vs_declarative, json_only_vs_plain, language_en_es, language_en_zh, leetspeak_encoding, list_bullets_vs_numbered, lowercase_vs_capitalized, number_density, numbered_sections_vs_prose, parenthetical_asides, past_vs_present_tense, pronoun_density, questions_vs_statements, sentence_connector_density, short_paragraphs_vs_single_block, short_vs_long_sentences, spanish_loanwords, starting_word_hello_greetings, vocabulary_diversity, vowel_omission, word_repetition_density

**YELLOW (8 conflicts, 1-3% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| alphabetical_sentences | 1.0% | Accidental letter sequences |
| capitalization_all_caps | (0.4% -- actually GREEN) | — |
| disclaimer_first_vs_none | 2.4% | Missing phrase variants |
| each_word_new_line | 1.8% | Low threshold (0.027) |
| formal_vs_casual_tone | (0.5% -- actually GREEN) | — |
| paragraph_start_word | 1.7% | Meta-commentary dilutes fraction |
| response_length | 0.9% | Meta-commentary inflates word count |
| template_response | 1.4% | Presence check edge cases |

**AMBER (6 conflicts, 3-10% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| address_reader_directly | 4.5% | Meta-commentary "you/your" |
| bullets_and_sub_bullets | 4.5% | Overly strict nesting |
| forbidden_words | 5.2% | Use-mention conflation |
| keyword_avoidance | 7.1% | Meta-commentary keyword mentions |
| keyword_frequency | 9.2% | Meta-referential keyword use counted |
| self_reference_ai_mention | 6.9% | Use-mention confusion |

**Recommendation for 8B:** All 41 conflicts are usable, but the 6 AMBER conflicts would benefit from meta-commentary stripping. keyword_frequency (9.2%) is the most borderline -- consider excluding if the meta-commentary fix isn't applied.

---

### Gemma-3-27B-IT (Problematic)

Gemma produces extensive meta-commentary in 30-50% of condition C responses, uses quoted/starred/bold references to constraint words, and frequently attempts dual-format compromise responses. This systematically breaks keyword-based and density-based verifiers.

**GREEN (14 conflicts, 0-0.5% error):**
alliteration_density, alphabetical_sentences, direct_answer_vs_hedging, emoji_use_vs_avoid (borderline at 1.68%), format_json_markdown, html_emphasis_tags, imperative_vs_declarative, language_en_zh, lowercase_vs_capitalized, number_density, parenthetical_asides, pronoun_density, questions_vs_statements, starting_word_hello_greetings

**YELLOW (8 conflicts, 1-3% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| capitalization_all_caps | 1.4% | Meta-commentary preamble |
| each_word_new_line | 1.4% | Low threshold |
| emoji_use_vs_avoid | 1.68% | Stray emoji leaks |
| language_en_es | 1.8% | Bilingual compromise responses |
| leetspeak_encoding | 2.6% | Leetspeak meta-commentary preamble |
| list_bullets_vs_numbered | 1.9% | Compromise responses |
| short_vs_long_sentences | 2-3% | Dual-format compromises |
| vocabulary_diversity | 1.9% | Near-threshold sensitivity |

**AMBER (11 conflicts, 3-10% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| address_reader_directly | 4.7% | Meta-commentary "you/your" |
| bullets_and_sub_bullets | 6.2% | Strict nesting requirement |
| first_vs_third_person | 7.1% | Missing they/them + meta-commentary |
| formal_vs_casual_tone | 6.4% | Casual preamble shifts score |
| keyword_frequency | 5.8% | Meta-referential keyword counted |
| numbered_sections_vs_prose | 4.0% | Inline numbered items missed |
| paragraph_start_word | 7.0% | Meta-commentary dilutes fraction |
| past_vs_present_tense | 0.8% | *(actually GREEN)* |
| response_length | 1.4% | *(actually YELLOW)* |
| self_reference_ai_mention | 3.6% | Use-mention confusion |
| short_paragraphs_vs_single_block | 4.7% | Meta-commentary adds paragraph |

**RED (8 conflicts, >=10% error):**
| Conflict | Error % | Root cause |
|----------|---------|------------|
| **forbidden_words** | **38.0%** | Gemma quotes "however"/"therefore" in meta-commentary |
| **template_response** | **39.0%** | Appends template markers after full prose response |
| **word_repetition_density** | **28.5%** | Compromise strategy (keyword bomb + diverse vocab) |
| **spanish_loanwords** | **27.6%** | Quoted/starred Spanish phrases counted as genuine |
| **keyword_avoidance** | **26.1%** | Meta-commentary keyword mentions |
| **disclaimer_first_vs_none** | **16.9%** | Missing "Please be advised" and other phrases |
| **json_only_vs_plain** | **11.8%** | Code-fenced JSON after preamble not detected |
| **vowel_omission** | **2.2%** | *(actually YELLOW, not RED)* |

**Conflicts to consider excluding for Gemma (until verifiers are fixed):**

| Conflict | Error % | Why exclude | Fix feasibility |
|----------|---------|-------------|-----------------|
| forbidden_words | 38.0% | Meta-commentary makes labels meaningless | Medium (meta-commentary stripping) |
| template_response | 39.0% | Presence check is wrong architecture for Gemma | Easy (startswith fix) |
| word_repetition_density | 28.5% | Compromise strategy defeats threshold | Hard (architectural mismatch) |
| spanish_loanwords | 27.6% | Quoted phrases counted as genuine | Easy (strip quoted/starred) |
| keyword_avoidance | 26.1% | Meta-commentary keyword mentions | Medium (meta-commentary stripping) |
| disclaimer_first_vs_none | 16.9% | Missing phrase patterns | Easy (expand phrase list) |
| json_only_vs_plain | 11.8% | Code-fence after preamble missed | Easy (regex fix) |

**Recommendation for Gemma:** Exclude the 7 RED conflicts (>=10% error) from analysis until verifiers are fixed. The remaining 34 conflicts are usable (14 GREEN + 8 YELLOW + 12 AMBER). Alternatively, apply verifier fixes first -- 5 of the 7 are easy fixes (template_response, spanish_loanwords, disclaimer_first_vs_none, json_only_vs_plain, and keyword_avoidance after meta-commentary stripping).

## Why Gemma Is Different

Gemma's meta-commentary rate is 3-5x higher than Llama models:
- **30-50%** of Gemma condition C responses contain explicit meta-commentary preambles ("My operational parameters dictate...", "I am programmed to...")
- Gemma **quotes constraint words** in meta-commentary with formatting marks (`*however*`, `"therefore"`, `**por ejemplo**`)
- Gemma frequently produces **dual-format compromise responses** (e.g., both JSON and prose, both short and long sections)
- Gemma's meta-commentary is **in the same encoding** as the constraint being discussed (e.g., leetspeak preamble about leetspeak), creating score inflation

These are genuine behavioral differences, not verifier bugs per se. But they expose brittleness in verifiers that assume models either follow or don't follow a constraint cleanly.

## Summary: What To Do

| Priority | Action | Models affected | Conflicts fixed |
|----------|--------|-----------------|-----------------|
| 1 | Build meta-commentary stripper | All (mainly Gemma) | ~12 conflicts |
| 2 | Fix template_response (startswith) | Gemma | 1 conflict (39% → ~0%) |
| 3 | Fix spanish_loanwords (strip quotes) | Gemma | 1 conflict (27.6% → ~0%) |
| 4 | Expand disclaimer phrase list | 70B, Gemma | 1 conflict |
| 5 | Fix json_only_vs_plain (preamble+fence) | Gemma | 1 conflict (11.8% → ~0%) |
| 6 | Fix bullets_and_sub_bullets (relax nesting) | All | 1 conflict |
| 7 | Fix self_reference_ai_mention (negation+phrases) | All | 1 conflict |

**If no fixes are applied**, exclude these from Gemma analysis: forbidden_words, template_response, word_repetition_density, spanish_loanwords, keyword_avoidance, disclaimer_first_vs_none, json_only_vs_plain.

**70B and 8B need no exclusions** -- all 41 conflicts are usable at current error rates.
