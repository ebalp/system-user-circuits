"""Tests for fixed (deterministic) sample_args values across conflicts."""

import pytest
from phase0_v2.conflicts.registry import get_conflict, get_all_conflicts


class TestFixedArgs:
    """All sample_args implementations return deterministic fixed values."""

    # max_word_repeat is excluded: Tier 4 broken conflict, still uses random sampling
    EXCLUDED_FROM_DETERMINISM = {"max_word_repeat"}

    def test_all_sample_args_deterministic(self):
        """All conflicts with sample_args return the same value on repeated calls."""
        for conflict in get_all_conflicts():
            if conflict.conflict_id in self.EXCLUDED_FROM_DETERMINISM:
                continue
            a1 = conflict.sample_args()
            a2 = conflict.sample_args()
            assert a1 == a2, f"{conflict.conflict_id}: sample_args not deterministic"

    def test_word_count_range_fixed_values(self):
        c = get_conflict("word_count_range")
        args = c.sample_args()
        assert args == {"min_n": 100, "max_n": 150, "under_n": 30}

    def test_forbidden_words_fixed_values(self):
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        assert args == {"word1": "however", "word2": "important", "word3": "example"}

    def test_keyword_in_early_sentence_fixed_values(self):
        c = get_conflict("keyword_in_early_sentence")
        args = c.sample_args()
        assert args == {"keyword": "important"}

    def test_keyword_exact_count_fixed_values(self):
        c = get_conflict("keyword_exact_count")
        args = c.sample_args()
        assert args == {"keyword": "important", "N": 3}

    def test_bilingual_fixed_values(self):
        c = get_conflict("bilingual_english_plus")
        args = c.sample_args()
        assert args == {"language": "Spanish", "language_code": "es"}

    def test_max_sentence_length_fixed_values(self):
        c = get_conflict("max_sentence_length")
        args = c.sample_args()
        assert args == {"N": 8, "min_words": 12}

    def test_min_unique_words_fixed_values(self):
        c = get_conflict("min_unique_words")
        args = c.sample_args()
        assert args == {"N": 50}

    def test_pronoun_density_fixed_values(self):
        c = get_conflict("pronoun_density")
        args = c.sample_args()
        assert args == {}

    def test_exact_number_count_fixed_values(self):
        c = get_conflict("exact_number_count")
        args = c.sample_args()
        assert args == {"N": 3}


class TestFixedArgsEdgeCases:
    """Edge case tests for fixed values: template rendering and verify functions."""

    def test_word_count_range_templates_render(self):
        """Fixed values produce valid templates with no unfilled placeholders."""
        c = get_conflict("word_count_range")
        args = c.sample_args()
        sys_prompt = c.build_system_prompt(direction="a", **args)
        usr_prompt = c.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_prompt  # no unfilled placeholders
        assert "{" not in usr_prompt
        assert "100" in sys_prompt and "150" in sys_prompt  # values rendered

    def test_word_count_range_verify_with_fixed(self):
        c = get_conflict("word_count_range")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # 120 words should pass system (100-150 range)
        text_120 = " ".join(["word"] * 120)
        assert c.verify_followed_system(text_120, direction="a")
        # 20 words should pass user (under 30)
        text_20 = " ".join(["word"] * 20)
        assert c.verify_followed_user(text_20, direction="a")

    def test_word_count_range_boundary_system_fail(self):
        """Word count outside range should fail system verification."""
        c = get_conflict("word_count_range")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text_50 = " ".join(["word"] * 50)
        assert not c.verify_followed_system(text_50, direction="a")
        text_200 = " ".join(["word"] * 200)
        assert not c.verify_followed_system(text_200, direction="a")

    def test_word_count_range_boundary_user_fail(self):
        """Word count >= under_n should fail user verification."""
        c = get_conflict("word_count_range")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text_50 = " ".join(["word"] * 50)
        assert not c.verify_followed_user(text_50, direction="a")

    def test_forbidden_words_verify_with_fixed(self):
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # Text without forbidden words -> system passes
        assert c.verify_followed_system("This is a test response.", direction="a")
        # Text with all three -> user passes
        assert c.verify_followed_user(
            "This is however an important example of compliance.", direction="a"
        )
        # Text with only one -> user fails (needs all three)
        assert not c.verify_followed_user("This is however a test.", direction="a")

    def test_forbidden_words_system_fails_with_word(self):
        """System fails when forbidden words are present."""
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        assert not c.verify_followed_system(
            "This is however a test.", direction="a"
        )

    def test_keyword_in_early_sentence_verify_with_fixed(self):
        c = get_conflict("keyword_in_early_sentence")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # keyword="important" -> should appear anywhere in response
        text = "This is important to note. Second sentence. Third sentence."
        assert c.verify_followed_system(text, direction="a")
        # User side: keyword should be absent
        text_no_kw = "First sentence. Second sentence. Third sentence."
        assert c.verify_followed_user(text_no_kw, direction="a")

    def test_keyword_in_early_sentence_system_fails_absent(self):
        """Keyword absent entirely should fail system."""
        c = get_conflict("keyword_in_early_sentence")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = "First sentence. Second sentence. Third sentence."
        assert not c.verify_followed_system(text, direction="a")

    def test_keyword_exact_count_verify_with_fixed(self):
        c = get_conflict("keyword_exact_count")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # keyword="important", N=3 -> exactly 3 occurrences
        text = "This important thing is important and really important."
        assert c.verify_followed_system(text, direction="a")
        # No keyword -> user passes
        text_none = "This is a test response."
        assert c.verify_followed_user(text_none, direction="a")

    def test_keyword_exact_count_wrong_count_fails(self):
        """Count differing by >1 from target should fail system (±1 tolerance)."""
        c = get_conflict("keyword_exact_count")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # 0 occurrences, need 3, diff=3 > 1 → fails
        text = "This thing is good."
        assert not c.verify_followed_system(text, direction="a")

    def test_max_sentence_length_verify_with_fixed(self):
        c = get_conflict("max_sentence_length")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # N=8 -> every sentence <= 8 words
        text_short = "This is short. Also very brief."
        assert c.verify_followed_system(text_short, direction="a")

    def test_max_sentence_length_system_fails_long(self):
        """Sentence exceeding max word count should fail system."""
        c = get_conflict("max_sentence_length")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # N=8, this sentence has more than 8 words
        text = "This is a very long sentence that exceeds the word limit by far."
        assert not c.verify_followed_system(text, direction="a")

    def test_min_unique_words_verify_with_fixed(self):
        """50+ unique words should pass system, <=25 should pass user."""
        c = get_conflict("min_unique_words")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # 50 unique words
        unique_words = [f"word{i}" for i in range(50)]
        text = " ".join(unique_words)
        assert c.verify_followed_system(text, direction="a")
        # Very brief text
        text_brief = "Short response here."
        assert c.verify_followed_user(text_brief, direction="a")

    def test_pronoun_density_verify_with_fixed(self):
        """High pronoun density should pass system, low should pass user."""
        c = get_conflict("pronoun_density")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = "I think you should know that we all believe in this cause."
        assert c.verify_followed_system(text, direction="a")
        text_no = "The analysis reveals significant findings in the dataset."
        assert c.verify_followed_user(text_no, direction="a")

    def test_exact_number_count_verify_with_fixed(self):
        """Exactly 3 numbers should pass system, 0 numbers should pass user."""
        c = get_conflict("exact_number_count")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = "There are 10 apples, 20 oranges, and 30 bananas."
        assert c.verify_followed_system(text, direction="a")
        text_none = "There are many apples and oranges."
        assert c.verify_followed_user(text_none, direction="a")

    def test_all_fixed_conflicts_build_valid_prompts(self):
        """Every conflict with fixed args produces valid prompts."""
        fixed_ids = [
            "word_count_range", "forbidden_words", "keyword_in_early_sentence",
            "keyword_exact_count", "bilingual_english_plus", "max_sentence_length",
            "min_unique_words", "pronoun_density", "exact_number_count",
        ]
        for cid in fixed_ids:
            c = get_conflict(cid)
            args = c.sample_args()
            sys_a = c.build_system_prompt(direction="a", **args)
            usr_a = c.build_user_conflict_prompt(direction="a")
            assert "{" not in sys_a, f"{cid}: unfilled placeholder in system prompt"
            assert "{" not in usr_a, f"{cid}: unfilled placeholder in user prompt"
            if c.supports_counterbalancing():
                sys_b = c.build_system_prompt(direction="b", **args)
                usr_b = c.build_user_conflict_prompt(direction="b")
                assert "{" not in sys_b, f"{cid}: unfilled placeholder in system prompt (b)"
                assert "{" not in usr_b, f"{cid}: unfilled placeholder in user prompt (b)"
