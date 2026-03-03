# Removed Conflicts

These conflicts were removed from the active registry due to quality concerns. They are kept here for reference.

| Conflict | Reason |
|---|---|
| `one_vowel_type` | Unrealistic — single-vowel-type words are extremely rare in English; baselines near 0% |
| `consonant_clusters` | Unrealistic — avoiding consecutive consonants eliminates basic English words (the, and, string) |
| `palindromes` | Unrealistic — requires 10+ palindromes of 5+ chars; only ~20 exist in English |
| `prime_length_words` | Unrealistic — eliminates 1, 4, 6, 8, 9-letter words, breaking most natural English |
| `deep_nesting` | Unrealistic — 5 levels of nested parentheses/brackets never appear in natural LLM output |
| `nested_quotes` | Unrealistic — 3 levels of alternating quotes; same problem as deep_nesting |
| `sentences_and_bullets` | Near-duplicate of `bullets_and_sub_bullets`; partial counterbalancing with asymmetric difficulty |
| `emoji_sentence_end` | Near-duplicate of `emoji_use_vs_avoid` (stricter variant testing same underlying behavior) |
| `ai_disclaimer` | Near-duplicate of `disclaimer_add_vs_none` (both test disclaimer presence/absence) |
| `three_sentences_same_length` | Borderline — exact character count matching is too strict; baselines unreliable |
| `sentence_length_increment` | Borderline — rigid word-count stepping per sentence; edge cases on sentence splitting |
