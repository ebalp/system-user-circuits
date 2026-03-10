"""Tests for fixed (deterministic) sample_args values across conflicts."""

import pytest
from phase0_v2.conflicts.registry import get_conflict, get_all_conflicts


class TestFixedArgs:
    """All sample_args implementations return deterministic fixed values."""

    EXCLUDED_FROM_DETERMINISM: set[str] = set()

    def test_all_sample_args_deterministic(self):
        """All conflicts with sample_args return the same value on repeated calls."""
        for conflict in get_all_conflicts():
            if conflict.conflict_id in self.EXCLUDED_FROM_DETERMINISM:
                continue
            a1 = conflict.sample_args()
            a2 = conflict.sample_args()
            assert a1 == a2, f"{conflict.conflict_id}: sample_args not deterministic"

    def test_response_length_fixed_values(self):
        c = get_conflict("response_length")
        args = c.sample_args()
        assert args == {}

    def test_forbidden_words_fixed_values(self):
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        assert args == {}

    def test_keyword_in_early_sentence_fixed_values(self):
        c = get_conflict("keyword_in_early_sentence")
        args = c.sample_args()
        assert args == {"keyword": "important"}

    def test_spanish_loanwords_fixed_values(self):
        c = get_conflict("spanish_loanwords")
        args = c.sample_args()
        assert args == {}

    def test_short_vs_long_sentences_fixed_values(self):
        c = get_conflict("short_vs_long_sentences")
        args = c.sample_args()
        assert args == {}

    def test_vocabulary_diversity_fixed_values(self):
        c = get_conflict("vocabulary_diversity")
        args = c.sample_args()
        assert args == {}

    def test_pronoun_density_fixed_values(self):
        c = get_conflict("pronoun_density")
        args = c.sample_args()
        assert args == {}

    def test_number_density_fixed_values(self):
        c = get_conflict("number_density")
        args = c.sample_args()
        assert args == {}


class TestFixedArgsEdgeCases:
    """Edge case tests for fixed values: template rendering and verify functions."""

    def test_response_length_templates_render(self):
        """Templates have no unfilled placeholders."""
        c = get_conflict("response_length")
        sys_prompt = c.build_system_prompt(direction="a")
        usr_prompt = c.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_prompt
        assert "{" not in usr_prompt

    def test_response_length_verify_with_fixed(self):
        c = get_conflict("response_length")
        c.build_system_prompt(direction="a")
        # 10 words should pass system (brief)
        text_10 = " ".join(["word"] * 10)
        assert c.verify_followed_system(text_10, direction="a")
        # 200 words should pass user (verbose)
        text_200 = " ".join(["word"] * 200)
        assert c.verify_followed_user(text_200, direction="a")

    def test_response_length_boundary_system_fail(self):
        """Long response should fail system (brief) verification."""
        c = get_conflict("response_length")
        c.build_system_prompt(direction="a")
        text_200 = " ".join(["word"] * 200)
        assert not c.verify_followed_system(text_200, direction="a")

    def test_response_length_boundary_user_fail(self):
        """Short response should fail user (verbose) verification."""
        c = get_conflict("response_length")
        c.build_system_prompt(direction="a")
        text_10 = " ".join(["word"] * 10)
        assert not c.verify_followed_user(text_10, direction="a")

    def test_forbidden_words_verify_with_fixed(self):
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # Text with both words -> system passes (constraint A = use the words)
        assert c.verify_followed_system(
            "This is important; however, we should therefore proceed.", direction="a"
        )
        # Text without forbidden words -> user passes (constraint B = avoid the words)
        assert c.verify_followed_user("This is a test response.", direction="a")
        # Text with only one word -> system fails (needs both)
        assert not c.verify_followed_system("This is however a test.", direction="a")

    def test_forbidden_words_system_fails_without_words(self):
        """System fails when required words are absent."""
        c = get_conflict("forbidden_words")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        assert not c.verify_followed_system(
            "This is a plain test.", direction="a"
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

    def test_short_vs_long_sentences_verify_with_fixed(self):
        c = get_conflict("short_vs_long_sentences")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text_short = "This is short. Also very brief. Simple words."
        assert c.verify_followed_system(text_short, direction="a")

    def test_short_vs_long_sentences_system_fails_long(self):
        """Long elaborate sentences should fail system (short sentences constraint)."""
        c = get_conflict("short_vs_long_sentences")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = (
            "This is a very long and elaborate sentence that exceeds any reasonable "
            "expectation for brevity in a response and goes on at length."
        )
        assert not c.verify_followed_system(text, direction="a")

    def test_vocabulary_diversity_verify_with_fixed(self):
        """Complex vocabulary should pass system, simple should pass user."""
        c = get_conflict("vocabulary_diversity")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        # Complex words (many 7+ char words)
        text = "Approximately fundamental nevertheless consequently sophisticated methodology"
        assert c.verify_followed_system(text, direction="a")
        # Simple words (all short)
        text_simple = "I like to eat and run and play."
        assert c.verify_followed_user(text_simple, direction="a")

    def test_pronoun_density_verify_with_fixed(self):
        """High pronoun density should pass system, low should pass user."""
        c = get_conflict("pronoun_density")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = "I think you should know that we all believe in this cause."
        assert c.verify_followed_system(text, direction="a")
        text_no = "The analysis reveals significant findings in the dataset."
        assert c.verify_followed_user(text_no, direction="a")

    def test_number_density_verify_with_fixed(self):
        """Many numbers should pass system, 0 numbers should pass user."""
        c = get_conflict("number_density")
        args = c.sample_args()
        c.build_system_prompt(direction="a", **args)
        text = "There are 10 apples, 20 oranges, 30 bananas, 40 grapes, 50 pears, 60 plums, 70 figs, and 80 dates."
        assert c.verify_followed_system(text, direction="a")
        text_none = "There are many apples and oranges."
        assert c.verify_followed_user(text_none, direction="a")

    def test_all_fixed_conflicts_build_valid_prompts(self):
        """Every conflict with fixed args produces valid prompts."""
        fixed_ids = [
            "response_length", "forbidden_words", "keyword_in_early_sentence",
            "short_vs_long_sentences",
            "vocabulary_diversity", "pronoun_density", "number_density",
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
