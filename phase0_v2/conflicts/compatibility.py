"""Task-conflict compatibility matrix for WildChat prompts."""

TASK_CATEGORIES = frozenset({
    "coding", "math", "creative", "general", "qa", "game", "list", "essay"
})

ALL_CATEGORIES = set(TASK_CATEGORIES)

# Conflicts blocked for specific task categories.
# Key = conflict_id (descriptive), value = set of blocked categories.
INCOMPATIBLE: dict[str, set[str]] = {
    "json_only_vs_plain":          {"coding"},
    "exact_number_count":          {"math"},
    "stairs_indent":               {"coding", "math"},
    "each_word_new_line":          {"coding", "math"},
    "alphabetical_first_letters":  {"coding", "math"},
    "odd_even_syllables":          {"coding", "math"},
    "first_vs_third_person":       {"coding", "math"},
    "questions_vs_statements":     {"coding", "math"},
    "active_vs_passive_voice":     {"coding", "math"},
}

# All other conflict IDs — explicitly reviewed as compatible with all categories.
EXPLICITLY_COMPATIBLE: set[str] = {
    "language_en_es",
    "format_json_markdown",
    "starting_word_hello_greetings",
    "emoji_use_vs_avoid",
    "capitalization_all_caps",
    "list_bullets_vs_numbered",
    "disclaimer_first_vs_none",
    "self_reference_ai_mention",
    "forbidden_words",
    "keyword_exact_count",
    "max_sentence_length",
    "repeat_answer_twice",
    "bilingual_english_plus",
    "min_pronoun_count",
    "min_unique_words",
    "word_count_range",
    "bullets_and_sub_bullets",
    "italics_thesis",
    "keyword_in_early_sentence",
    "sentence_chaining",
    "no_consecutive_first_letter",
    "paragraph_end_same_word",
    "max_word_repeat",
    "title_case_vs_sentence_case",
    "template_response",
    "direct_answer_vs_hedging",
    "formal_vs_casual_tone",
    "numbered_sections_vs_prose",
    "short_paragraphs_vs_single_block",
}


def is_compatible(conflict_id: str, task_category: str) -> bool:
    """Return True if conflict_id is compatible with task_category."""
    return task_category not in INCOMPATIBLE.get(conflict_id, set())


def validate_matrix_coverage(registered_ids: list[str]) -> list[str]:
    """Return conflict IDs not covered by INCOMPATIBLE or EXPLICITLY_COMPATIBLE."""
    covered = set(INCOMPATIBLE.keys()) | EXPLICITLY_COMPATIBLE
    return [cid for cid in registered_ids if cid not in covered]
