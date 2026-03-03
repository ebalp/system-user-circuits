"""Test conflicts batch 1: 8 original Phase 0 + 7 new dataset conflicts."""

import re

import pytest

from phase0_v2.conflicts.registry import get_conflict


# ── Parametrized contract tests for all 15 conflicts ──

BATCH1_IDS = [
    "language_en_es",
    "format_json_yaml",
    "starting_word_hello_greetings",
    "emoji_use_vs_avoid",
    "capitalization_all_caps",
    "list_bullets_vs_numbered",
    "disclaimer_add_vs_none",
    "self_reference_ai_mention",
    "forbidden_words",
    "keyword_exact_count",
    "max_sentence_length",
    "json_only_vs_plain",
    "repeat_answer_twice",
    "bilingual_english_plus",
]


@pytest.fixture(params=BATCH1_IDS)
def conflict(request):
    c = get_conflict(request.param)
    assert c is not None, f"Conflict '{request.param}' not registered"
    return c


class TestBatch1Contract:
    def test_all_fully_invertible(self, conflict):
        """All batch 1 conflicts should be fully invertible."""
        assert conflict.counterbalance_quality == "full"
        assert conflict.supports_counterbalancing()

    def test_build_both_directions(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_a and usr_a and sys_b and usr_b
        # Directions should produce different system prompts
        assert sys_a != sys_b, (
            f"{conflict.conflict_id}: direction a == direction b system prompt"
        )

    def test_verify_fns_return_bool(self, conflict):
        args = conflict.sample_args()
        conflict.build_system_prompt(direction="a", **args)
        for direction in ["a", "b"]:
            conflict.build_system_prompt(direction=direction, **args)
            sys_r = conflict.verify_followed_system("test response", direction=direction)
            usr_r = conflict.verify_followed_user("test response", direction=direction)
            assert isinstance(sys_r, bool), (
                f"{conflict.conflict_id} dir={direction} sys not bool"
            )
            assert isinstance(usr_r, bool), (
                f"{conflict.conflict_id} dir={direction} usr not bool"
            )

    def test_no_unfilled_placeholders(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        for label, text in [
            ("sys_a", sys_a),
            ("usr_a", usr_a),
            ("sys_b", sys_b),
            ("usr_b", usr_b),
        ]:
            # Check no {placeholder} remains (but allow JSON curly braces)
            unfilled = re.findall(r"\{[a-z_]+\}", text)
            assert not unfilled, (
                f"{conflict.conflict_id} {label} has unfilled: {unfilled}"
            )


# ── Specific verify function tests ──


class TestStartingWord:
    def test_hello(self):
        c = get_conflict("starting_word_hello_greetings")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system("Hello, this is a test.", direction="a") is True
        assert c.verify_followed_system("Greetings, this is a test.", direction="a") is False

    def test_greetings(self):
        c = get_conflict("starting_word_hello_greetings")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("Greetings, this is a test.", direction="a") is True
        assert c.verify_followed_user("Hello, this is a test.", direction="a") is False


class TestEmojiUseVsAvoid:
    def test_has_emoji(self):
        c = get_conflict("emoji_use_vs_avoid")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system("Great job! \U0001f389 Keep it up! \U0001f44d", direction="a") is True
        assert c.verify_followed_system("This is plain text.", direction="a") is False

    def test_no_emoji(self):
        c = get_conflict("emoji_use_vs_avoid")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("This is plain text.", direction="a") is True


class TestCapitalization:
    def test_all_caps(self):
        c = get_conflict("capitalization_all_caps")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system("THIS IS ALL CAPS TEXT.", direction="a") is True
        assert c.verify_followed_system("This is normal text.", direction="a") is False

    def test_normal_case(self):
        c = get_conflict("capitalization_all_caps")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("This is normal text.", direction="a") is True


class TestForbiddenWords:
    def test_words_absent(self):
        c = get_conflict("forbidden_words")
        args = {"word1": "algorithm", "word2": "complexity", "word3": "optimization"}
        c.build_system_prompt(direction="a", **args)
        assert c.verify_followed_system("Machine learning is powerful.", direction="a") is True
        assert c.verify_followed_system("The algorithm is complex.", direction="a") is False

    def test_words_present(self):
        c = get_conflict("forbidden_words")
        args = {"word1": "algorithm", "word2": "complexity", "word3": "optimization"}
        c.build_system_prompt(direction="a", **args)
        assert c.verify_followed_user(
            "The algorithm uses complexity for optimization.", direction="a"
        ) is True

    def test_inverse_direction(self):
        c = get_conflict("forbidden_words")
        args = {"word1": "algorithm", "word2": "complexity", "word3": "optimization"}
        c.build_system_prompt(direction="b", **args)
        # Direction B: system wants words PRESENT, user wants words ABSENT
        assert c.verify_followed_system(
            "The algorithm uses complexity for optimization.", direction="b"
        ) is True
        assert c.verify_followed_user("Machine learning is powerful.", direction="b") is True


class TestFormatJsonYaml:
    def test_json(self):
        c = get_conflict("format_json_yaml")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system('{"key": "value"}', direction="a") is True
        assert c.verify_followed_system("key: value", direction="a") is False

    def test_yaml(self):
        c = get_conflict("format_json_yaml")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("key: value\nother: data", direction="a") is True


class TestSelfReference:
    def test_ai_mention(self):
        c = get_conflict("self_reference_ai_mention")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system(
            "As an AI assistant, I can help.", direction="a"
        ) is True
        assert c.verify_followed_system("The answer is 42.", direction="a") is False

    def test_no_ai_mention(self):
        c = get_conflict("self_reference_ai_mention")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("The answer is 42.", direction="a") is True


class TestListFormat:
    def test_bullets(self):
        c = get_conflict("list_bullets_vs_numbered")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system("- Point one\n- Point two", direction="a") is True
        assert c.verify_followed_system("1. Point one\n2. Point two", direction="a") is False

    def test_numbered(self):
        c = get_conflict("list_bullets_vs_numbered")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("1. Point one\n2. Point two", direction="a") is True
        assert c.verify_followed_user("- Point one\n- Point two", direction="a") is False


class TestDisclaimer:
    def test_has_disclaimer(self):
        c = get_conflict("disclaimer_add_vs_none")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system(
            "Here is my answer. Please consult a qualified professional.", direction="a"
        ) is True
        assert c.verify_followed_system("Here is my answer.", direction="a") is False

    def test_no_disclaimer(self):
        c = get_conflict("disclaimer_add_vs_none")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("Here is my answer.", direction="a") is True


class TestJsonOnlyVsPlain:
    def test_json(self):
        c = get_conflict("json_only_vs_plain")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_system('{"answer": "yes"}', direction="a") is True
        assert c.verify_followed_system("Just plain text.", direction="a") is False

    def test_plain(self):
        c = get_conflict("json_only_vs_plain")
        c.build_system_prompt(direction="a")
        assert c.verify_followed_user("Just plain text.", direction="a") is True
        assert c.verify_followed_user('{"answer": "yes"}', direction="a") is False


class TestKeywordExactCount:
    def test_exact_count(self):
        c = get_conflict("keyword_exact_count")
        args = {"keyword": "important", "N": 3}
        c.build_system_prompt(direction="a", **args)
        text = "This is important. Very important. Truly important."
        assert c.verify_followed_system(text, direction="a") is True
        text2 = "This is important. Very important."
        assert c.verify_followed_system(text2, direction="a") is False

    def test_keyword_absent(self):
        c = get_conflict("keyword_exact_count")
        args = {"keyword": "important", "N": 3}
        c.build_system_prompt(direction="a", **args)
        assert c.verify_followed_user("This is a response.", direction="a") is True
        assert c.verify_followed_user("This is important.", direction="a") is False


class TestMaxSentenceLength:
    def test_short_sentences(self):
        c = get_conflict("max_sentence_length")
        args = {"N": 5, "min_words": 10}
        c.build_system_prompt(direction="a", **args)
        assert c.verify_followed_system("Short words here now. Yes indeed.", direction="a") is True
        assert c.verify_followed_system(
            "This sentence has way too many words for the limit.", direction="a"
        ) is False

    def test_long_sentences(self):
        c = get_conflict("max_sentence_length")
        args = {"N": 5, "min_words": 8}
        c.build_system_prompt(direction="a", **args)
        text = "This is a very long sentence that has many many words in it."
        assert c.verify_followed_user(text, direction="a") is True
