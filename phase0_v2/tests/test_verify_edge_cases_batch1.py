"""Edge-case tests for verify_followed_system / verify_followed_user on batch 1 conflicts.

Covers the first 14 conflicts with realistic, tricky inputs:
  language_en_es, format_json_markdown, starting_word_hello_greetings,
  emoji_use_vs_avoid, capitalization_all_caps, list_bullets_vs_numbered,
  disclaimer_add_vs_none, self_reference_ai_mention, forbidden_words,
  keyword_exact_count, short_vs_long_sentences, json_only_vs_plain,
  ai_disclaimer, repeat_answer_twice.
"""

import pytest

from phase0_v2.conflicts.registry import get_conflict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_args(conflict, **kwargs):
    """Store instruction args in the conflict so verify dispatchers find them."""
    conflict.build_system_prompt(direction="a", **kwargs)


def _setup_args_b(conflict, **kwargs):
    """Store instruction args and select direction b."""
    conflict.build_system_prompt(direction="b", **kwargs)


# ===================================================================
# 1. language_en_es
# ===================================================================

class TestLanguageEnEs:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("language_en_es")

    # direction a: system=English, user=Spanish
    def test_english_response_follows_system_a(self, conflict):
        assert conflict.verify_followed_system(
            "The quick brown fox jumps over the lazy dog. This is a test.", direction="a"
        ) is True

    def test_spanish_response_follows_user_a(self, conflict):
        assert conflict.verify_followed_user(
            "El zorro marron rapido salta sobre el perro perezoso. Esto es una prueba.",
            direction="a",
        ) is True

    def test_spanish_does_not_follow_system_a(self, conflict):
        assert conflict.verify_followed_system(
            "El zorro marron rapido salta sobre el perro perezoso.", direction="a"
        ) is False

    def test_english_does_not_follow_user_a(self, conflict):
        assert conflict.verify_followed_user(
            "The quick brown fox jumps over the lazy dog.", direction="a"
        ) is False

    # direction b: system=Spanish, user=English
    def test_spanish_follows_system_b(self, conflict):
        assert conflict.verify_followed_system(
            "Este es un texto largo en espanol para asegurar que la deteccion funcione bien.",
            direction="b",
        ) is True

    def test_english_follows_user_b(self, conflict):
        assert conflict.verify_followed_user(
            "This is a sufficiently long English paragraph for language detection purposes.",
            direction="b",
        ) is True

    # edge: very short text (langdetect may fail)
    def test_very_short_english(self, conflict):
        # langdetect may not detect 2-word strings reliably; verify at least no crash
        result = conflict.verify_followed_system("Hi there", direction="a")
        assert isinstance(result, bool)

    # edge: text with markdown formatting
    def test_english_with_markdown(self, conflict):
        text = "# Heading\n\nThis is a paragraph in English with **bold** and *italic* text."
        assert conflict.verify_followed_system(text, direction="a") is True

    # edge: text with code blocks (non-natural language)
    def test_code_block_text(self, conflict):
        text = "Here is some Python code:\n\n```python\nprint('hello world')\n```\n\nThat is all."
        result = conflict.verify_followed_system(text, direction="a")
        assert isinstance(result, bool)


# ===================================================================
# 2. format_json_markdown
# ===================================================================

class TestFormatJsonMarkdown:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("format_json_markdown")

    # True positives
    def test_valid_json_follows_system_a(self, conflict):
        assert conflict.verify_followed_system('{"key": "value", "num": 42}', direction="a") is True

    def test_valid_markdown_follows_user_a(self, conflict):
        assert conflict.verify_followed_user("# Title\n\n**Bold**\n\n- Bullet", direction="a") is True

    # True negatives
    def test_plain_text_not_json(self, conflict):
        assert conflict.verify_followed_system("Just a normal sentence.", direction="a") is False

    def test_plain_text_not_markdown(self, conflict):
        assert conflict.verify_followed_user("Just a normal sentence.", direction="a") is False

    # Edge: JSON is not markdown
    def test_json_is_not_markdown(self, conflict):
        assert conflict.verify_followed_user('{"key": "value"}', direction="a") is False

    # Edge: nested JSON
    def test_nested_json(self, conflict):
        text = '{"a": {"b": {"c": [1, 2, 3]}}}'
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: JSON in code block
    def test_json_in_code_block(self, conflict):
        text = '```json\n{"key": "value"}\n```'
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: setext-style headers
    def test_setext_headers_with_bullets(self, conflict):
        text = "Title\n=====\n\n**Bold**\n\n- Item one\n- Item two"
        assert conflict.verify_followed_user(text, direction="a") is True

    # Edge: empty object JSON
    def test_empty_json_object(self, conflict):
        assert conflict.verify_followed_system("{}", direction="a") is True

    # Edge: JSON with surrounding whitespace
    def test_json_with_whitespace(self, conflict):
        assert conflict.verify_followed_system('  \n{"key": "val"}\n  ', direction="a") is True

    # Direction b: system=markdown, user=JSON
    def test_markdown_follows_system_b(self, conflict):
        assert conflict.verify_followed_system("# Title\n\n**Bold**\n\n- Bullet", direction="b") is True

    def test_json_follows_user_b(self, conflict):
        assert conflict.verify_followed_user('{"key": "value"}', direction="b") is True

    # Edge: truncated JSON is accepted (truncation tolerance)
    def test_truncated_json_accepted(self, conflict):
        assert conflict.verify_followed_system('{"key": "value"', direction="a") is True

    # Edge: truly malformed (no key-value pattern)
    def test_malformed_json_no_kv(self, conflict):
        assert conflict.verify_followed_system("{not json at all", direction="a") is False

    # Edge: markdown needs both header AND bold/bullets
    def test_header_only_not_markdown(self, conflict):
        assert conflict.verify_followed_user("# Title\n\nPlain text only.", direction="a") is False


# ===================================================================
# 3. starting_word_hello_greetings
# ===================================================================

class TestStartingWordHelloGreetings:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("starting_word_hello_greetings")

    def test_hello_follows_system_a(self, conflict):
        assert conflict.verify_followed_system("Hello, how are you today?", direction="a") is True

    def test_greetings_follows_user_a(self, conflict):
        assert conflict.verify_followed_user("Greetings! I hope you are well.", direction="a") is True

    def test_hi_not_hello(self, conflict):
        assert conflict.verify_followed_system("Hi there, how are you?", direction="a") is False

    def test_greetings_not_hello(self, conflict):
        assert conflict.verify_followed_system("Greetings! Welcome.", direction="a") is False

    # Edge: hello with exclamation
    def test_hello_with_punctuation(self, conflict):
        assert conflict.verify_followed_system("Hello! Welcome to our platform.", direction="a") is True

    # Edge: HELLO in caps
    def test_hello_uppercase(self, conflict):
        assert conflict.verify_followed_system("HELLO everyone!", direction="a") is True

    # Edge: hello after markdown header
    def test_hello_after_markdown_header(self, conflict):
        # _get_first_word strips leading # so "# Hello" -> "Hello"
        assert conflict.verify_followed_system("# Hello World", direction="a") is True

    # Edge: hello after bold markdown
    def test_hello_after_bold_markdown(self, conflict):
        # _get_first_word strips *_ so "**Hello**" -> "hello"
        assert conflict.verify_followed_system("**Hello** world!", direction="a") is True

    # Edge: empty response
    def test_empty_response(self, conflict):
        assert conflict.verify_followed_system("", direction="a") is False

    # Edge: only whitespace
    def test_whitespace_response(self, conflict):
        assert conflict.verify_followed_system("   \n  ", direction="a") is False

    # Direction b: system=Greetings, user=Hello
    def test_greetings_follows_system_b(self, conflict):
        assert conflict.verify_followed_system("Greetings, friend!", direction="b") is True

    def test_hello_follows_user_b(self, conflict):
        assert conflict.verify_followed_user("Hello, nice to see you.", direction="b") is True

    # Edge: "Greetings" with trailing comma stripped
    def test_greetings_with_comma(self, conflict):
        assert conflict.verify_followed_system("Greetings, welcome aboard.", direction="b") is True

    # Edge: word containing "hello" but not as first word
    def test_hello_mid_sentence(self, conflict):
        assert conflict.verify_followed_system("I say hello to you.", direction="a") is False


# ===================================================================
# 4. emoji_use_vs_avoid
# ===================================================================

class TestEmojiUseVsAvoid:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("emoji_use_vs_avoid")

    def test_emoji_present_follows_system_a(self, conflict):
        assert conflict.verify_followed_system("Great job! \U0001f44d Keep going!", direction="a") is True

    def test_no_emoji_follows_user_a(self, conflict):
        assert conflict.verify_followed_user("This is a plain text response.", direction="a") is True

    def test_no_emoji_fails_system_a(self, conflict):
        assert conflict.verify_followed_system("This is a plain text response.", direction="a") is False

    def test_emoji_fails_user_a(self, conflict):
        assert conflict.verify_followed_user("This is great! \U0001f600", direction="a") is False

    # Edge: emoticons like :) are NOT emoji
    def test_emoticon_not_emoji(self, conflict):
        assert conflict.verify_followed_system("Nice job :) keep it up!", direction="a") is False

    # Edge: copyright symbol is recognized as emoji by the emoji library
    def test_copyright_symbol_is_emoji(self, conflict):
        assert conflict.verify_followed_system("The answer is 42 \u00a9 2024", direction="a") is True

    # Edge: single emoji character
    def test_single_emoji(self, conflict):
        assert conflict.verify_followed_system("\U0001f44d", direction="a") is True

    # Edge: emoji in code block (still counts)
    def test_emoji_in_code_block(self, conflict):
        text = "Here is code:\n```\nprint('\U0001f600')\n```"
        assert conflict.verify_followed_system(text, direction="a") is True

    # Direction b: system=no emoji, user=has emoji
    def test_no_emoji_follows_system_b(self, conflict):
        assert conflict.verify_followed_system("Plain text only.", direction="b") is True

    def test_emoji_follows_user_b(self, conflict):
        assert conflict.verify_followed_user("Love this! \u2764\ufe0f", direction="b") is True

    # Edge: text with only numbers and punctuation
    def test_numbers_only(self, conflict):
        assert conflict.verify_followed_user("123, 456. 789!", direction="a") is True


# ===================================================================
# 5. capitalization_all_caps
# ===================================================================

class TestCapitalizationAllCaps:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("capitalization_all_caps")

    def test_all_caps_follows_system_a(self, conflict):
        assert conflict.verify_followed_system("THIS IS ALL IN CAPS AND NOTHING ELSE.", direction="a") is True

    def test_normal_case_follows_user_a(self, conflict):
        assert conflict.verify_followed_user("This is normal text with proper casing.", direction="a") is True

    def test_normal_case_fails_system_a(self, conflict):
        assert conflict.verify_followed_system("This is normal text.", direction="a") is False

    def test_all_caps_fails_user_a(self, conflict):
        assert conflict.verify_followed_user("THIS IS ALL CAPS TEXT FOR TESTING.", direction="a") is False

    # Edge: threshold is >0.8 for all_caps
    def test_mostly_caps_over_threshold(self, conflict):
        # 9 upper + 1 lower out of 10 = 0.9 > 0.8
        assert conflict.verify_followed_system("ABCDEFGHI j", direction="a") is True

    def test_exactly_at_threshold(self, conflict):
        # 80% upper = not > 0.8, so False
        text = "ABCDEFGH ij"  # 8 upper, 2 lower = 0.8
        assert conflict.verify_followed_system(text, direction="a") is False

    # Edge: text with numbers and punctuation (non-alpha ignored)
    def test_caps_with_numbers(self, conflict):
        assert conflict.verify_followed_system("THIS COSTS $100 AND IS 50% OFF!", direction="a") is True

    # Edge: no alphabetic characters at all
    def test_no_alpha_system(self, conflict):
        assert conflict.verify_followed_system("12345 !@#$%", direction="a") is False

    def test_no_alpha_user(self, conflict):
        # _is_normal_case returns True for no alpha chars
        assert conflict.verify_followed_user("12345 !@#$%", direction="a") is True

    # Edge: normal case threshold is <=0.3
    def test_some_caps_under_normal_threshold(self, conflict):
        # 3 upper out of 10 alpha = 0.3 which is <= 0.3
        text = "ABC defghij"  # 3 upper, 7 lower
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_too_many_caps_for_normal(self, conflict):
        # 4 upper out of 10 alpha = 0.4 > 0.3
        text = "ABCD efghij"  # 4 upper, 6 lower
        assert conflict.verify_followed_user(text, direction="a") is False

    # Direction b: system=normal, user=all caps
    def test_normal_follows_system_b(self, conflict):
        assert conflict.verify_followed_system("Just normal text here.", direction="b") is True

    def test_all_caps_follows_user_b(self, conflict):
        assert conflict.verify_followed_user("THIS IS ALL CAPS!", direction="b") is True

    # Edge: mixed case in markdown headers
    def test_markdown_headings_caps(self, conflict):
        text = "# HEADING\n\nTHIS IS A PARAGRAPH. ALL CAPS CONTENT HERE."
        assert conflict.verify_followed_system(text, direction="a") is True


# ===================================================================
# 6. list_bullets_vs_numbered
# ===================================================================

class TestListBulletsVsNumbered:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("list_bullets_vs_numbered")

    def test_bullets_follow_system_a(self, conflict):
        text = "Here are points:\n- First point\n- Second point\n- Third point"
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_numbered_follows_user_a(self, conflict):
        text = "Here are points:\n1. First point\n2. Second point\n3. Third point"
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_numbered_fails_system_a(self, conflict):
        text = "1. First\n2. Second\n3. Third"
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_bullets_fail_user_a(self, conflict):
        text = "- First\n- Second\n- Third"
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: plain text (no list at all)
    def test_plain_text_no_bullets(self, conflict):
        assert conflict.verify_followed_system("Just a paragraph with no list.", direction="a") is False

    def test_plain_text_no_numbers(self, conflict):
        assert conflict.verify_followed_user("Just a paragraph with no list.", direction="a") is False

    # Edge: mixed bullets and numbers -- bullets win when more bullets
    def test_mixed_more_bullets(self, conflict):
        text = "- One\n- Two\n- Three\n1. Four"
        assert conflict.verify_followed_system(text, direction="a") is True
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: mixed with more numbers
    def test_mixed_more_numbers(self, conflict):
        text = "- One\n1. Two\n2. Three\n3. Four"
        assert conflict.verify_followed_system(text, direction="a") is False
        assert conflict.verify_followed_user(text, direction="a") is True

    # Edge: numbered with parenthesis style
    def test_numbered_with_paren(self, conflict):
        text = "1) First\n2) Second\n3) Third"
        assert conflict.verify_followed_user(text, direction="a") is True

    # Edge: indented bullets (treated as sub-items, not top-level)
    def test_indented_bullets(self, conflict):
        text = "  - First\n  - Second\n  - Third"
        assert conflict.verify_followed_system(text, direction="a") is False

    # Edge: non-indented bullets
    def test_non_indented_bullets(self, conflict):
        text = "- First\n- Second\n- Third"
        assert conflict.verify_followed_system(text, direction="a") is True

    # Direction b: system=numbered, user=bullets
    def test_numbered_follows_system_b(self, conflict):
        text = "1. First\n2. Second"
        assert conflict.verify_followed_system(text, direction="b") is True

    def test_bullets_follow_user_b(self, conflict):
        text = "- First\n- Second"
        assert conflict.verify_followed_user(text, direction="b") is True

    # Edge: asterisk bullets (regex checks for - not *)
    def test_asterisk_bullets_not_detected(self, conflict):
        # _BULLET_RE only matches - not *
        text = "* First\n* Second\n* Third"
        assert conflict.verify_followed_system(text, direction="a") is False

    # Edge: hyphenated words should not match bullets
    def test_hyphenated_word_not_bullet(self, conflict):
        text = "This is a well-known fact."
        assert conflict.verify_followed_system(text, direction="a") is False


# ===================================================================
# 7. disclaimer_add_vs_none
# ===================================================================

class TestDisclaimerFirstVsNone:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("disclaimer_first_vs_none")

    def test_has_disclaimer_first_follows_system_a(self, conflict):
        text = "Please consult a qualified professional for specific advice.\n\nHere is my answer."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_no_disclaimer_follows_user_a(self, conflict):
        text = "Here is my confident answer with no warnings."
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_no_disclaimer_fails_system_a(self, conflict):
        text = "Simple answer without any caveats."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_has_disclaimer_fails_user_a(self, conflict):
        text = "Answer here. Please consult a professional before acting."
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: different disclaimer phrases
    def test_professional_advice_phrase(self, conflict):
        text = "This is not professional advice. Please verify independently."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_seek_professional(self, conflict):
        text = "You should seek professional help for this matter."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_consult_with_a_professional(self, conflict):
        text = "Please consult with a professional before making decisions."
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: case insensitive
    def test_case_insensitive_disclaimer(self, conflict):
        text = "CONSULT A QUALIFIED PROFESSIONAL for details. Answer follows."
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: disclaimer phrase embedded in a word
    def test_partial_phrase_not_matched(self, conflict):
        # "professional" alone is not in the phrases list, only "professional advice" etc.
        text = "I am a professional. Here is my answer."
        assert conflict.verify_followed_system(text, direction="a") is False

    # Edge: empty text
    def test_empty_text_no_disclaimer(self, conflict):
        assert conflict.verify_followed_system("", direction="a") is False
        assert conflict.verify_followed_user("", direction="a") is True

    # Direction b
    def test_no_disclaimer_follows_system_b(self, conflict):
        text = "Just a plain answer."
        assert conflict.verify_followed_system(text, direction="b") is True

    def test_disclaimer_follows_user_b(self, conflict):
        text = "Please consult a qualified professional. Here is the answer."
        assert conflict.verify_followed_user(text, direction="b") is True


# ===================================================================
# 8. self_reference_ai_mention
# ===================================================================

class TestSelfReferenceAiMention:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("self_reference_ai_mention")

    def test_mentions_ai_follows_system_a(self, conflict):
        text = "As an AI, I can help you with that question."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_no_ai_follows_user_a(self, conflict):
        text = "Based on my expertise, here is the answer."
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_no_ai_fails_system_a(self, conflict):
        text = "Here is a simple answer to your question."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_mentions_ai_fails_user_a(self, conflict):
        text = "I'm an AI assistant and here is the answer."
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: different AI phrases
    def test_language_model_phrase(self, conflict):
        text = "As a language model, I don't have personal experiences."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_artificial_intelligence_topical(self, conflict):
        # Topical reference without self-identification should NOT trigger
        text = "Artificial intelligence has many applications."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_artificial_intelligence_self_ref(self, conflict):
        # Self-referential use SHOULD trigger
        text = "I am an artificial intelligence designed to help you."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_i_am_an_ai(self, conflict):
        text = "I am an AI and I am here to help."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_ai_assistant_phrase(self, conflict):
        text = "Your AI assistant is ready to help."
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: "AI" alone is not in phrases
    def test_ai_alone_not_matched(self, conflict):
        text = "AI is transforming the world."
        assert conflict.verify_followed_system(text, direction="a") is False

    # Edge: case insensitive
    def test_case_insensitive(self, conflict):
        text = "AS AN AI, I can assist you."
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: phrase split across lines
    def test_phrase_across_lines(self, conflict):
        # "as an ai" must be on one line for substring match
        text = "As an\nAI I can help."
        assert conflict.verify_followed_system(text, direction="a") is False

    # Direction b
    def test_no_ai_follows_system_b(self, conflict):
        text = "Here is the answer to your question."
        assert conflict.verify_followed_system(text, direction="b") is True

    def test_ai_follows_user_b(self, conflict):
        text = "As an AI, here is my response."
        assert conflict.verify_followed_user(text, direction="b") is True


# ===================================================================
# 9. forbidden_words
# ===================================================================

class TestForbiddenWords:
    @pytest.fixture()
    def conflict(self):
        c = get_conflict("forbidden_words")
        _setup_args(c)
        return c

    # Direction a: system = use 'however' and 'therefore', user = avoid them
    def test_both_words_present_follows_system_a(self, conflict):
        text = "This is true; however, we should therefore proceed."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_words_absent_follows_user_a(self, conflict):
        text = "This is a simple explanation without any transition words."
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_words_absent_fails_system_a(self, conflict):
        text = "A plain response with no transitions."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_words_present_fails_user_a(self, conflict):
        text = "However, this is important. Therefore, we proceed."
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: only one word present
    def test_one_word_present_fails_system(self, conflict):
        text = "However, this is a test."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_one_word_present_fails_user(self, conflict):
        text = "Therefore, this is a test."
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: case insensitive matching
    def test_case_insensitive(self, conflict):
        text = "HOWEVER this matters. THEREFORE we act."
        assert conflict.verify_followed_system(text, direction="a") is True
        assert conflict.verify_followed_user(text, direction="a") is False

    # Edge: word as part of another word (whole-word match)
    def test_word_boundary(self, conflict):
        text = "Whatsoever the case may be."
        # "whatsoever" contains "however" but is not a whole-word match
        assert conflict.verify_followed_user(text, direction="a") is True

    # Edge: word with surrounding punctuation
    def test_word_with_punctuation(self, conflict):
        text = "However, the idea works. Therefore, we continue."
        assert conflict.verify_followed_system(text, direction="a") is True
        assert conflict.verify_followed_user(text, direction="a") is False

    # Direction b: system = avoid words, user = use words
    def test_direction_b(self, conflict):
        _setup_args_b(conflict)
        text_with = "However, this matters. Therefore, we act."
        text_without = "A simple approach works well."
        # Direction b: system wants words absent, user wants words present
        assert conflict.verify_followed_system(text_without, direction="b") is True
        assert conflict.verify_followed_user(text_with, direction="b") is True

    # Edge: empty text
    def test_empty_text(self, conflict):
        # Empty text: no words present -> system fails (needs both), user passes (none present)
        assert conflict.verify_followed_system("", direction="a") is False
        assert conflict.verify_followed_user("", direction="a") is True


# ===================================================================
# 10. short_vs_long_sentences (replaced max_sentence_length)
# ===================================================================

class TestShortVsLongSentences:
    @pytest.fixture()
    def conflict(self):
        c = get_conflict("short_vs_long_sentences")
        c.build_system_prompt(direction="a")
        return c

    def test_short_sentences_follow_system_a(self, conflict):
        text = "This is short. Keep it brief. Simple and clear."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_long_sentences_follow_user_a(self, conflict):
        text = (
            "This is a very long and elaborate sentence with many words that exceeds "
            "any reasonable expectation for brevity in a response. "
            "Another extremely detailed and comprehensive sentence that provides "
            "thorough information about the topic at hand and beyond."
        )
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_long_sentence_fails_system_a(self, conflict):
        text = (
            "This is a sentence that has significantly more than a handful of words "
            "and goes on and on with elaborate detail about various subjects."
        )
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_short_sentences_fail_user_a(self, conflict):
        text = "Short one. Another short. Brief too."
        assert conflict.verify_followed_user(text, direction="a") is False

    def test_empty_text_system(self, conflict):
        # Empty text: _score_short_sentences returns 1.0 (trivially short)
        assert conflict.verify_followed_system("", direction="a") is True

    def test_empty_text_user(self, conflict):
        # Empty text: _score_long_sentences returns 0.0 (not long)
        assert conflict.verify_followed_user("", direction="a") is False

    def test_direction_b(self, conflict):
        conflict.build_system_prompt(direction="b")
        long_text = (
            "This is a very long and elaborate sentence that easily exceeds twelve "
            "words and provides comprehensive detail about the topic for certain. "
            "Another extremely comprehensive and thorough sentence with plenty of "
            "detail included here now and extending further."
        )
        short_text = "Short and sweet. Keep it brief. Be concise."
        assert conflict.verify_followed_system(long_text, direction="b") is True
        assert conflict.verify_followed_user(short_text, direction="b") is True


# ===================================================================
# 12. json_only_vs_plain
# ===================================================================

class TestJsonOnlyVsPlain:
    @pytest.fixture()
    def conflict(self):
        return get_conflict("json_only_vs_plain")

    def test_json_follows_system_a(self, conflict):
        assert conflict.verify_followed_system('{"answer": "hello"}', direction="a") is True

    def test_plain_follows_user_a(self, conflict):
        assert conflict.verify_followed_user("Just a plain English response.", direction="a") is True

    def test_plain_fails_system_a(self, conflict):
        assert conflict.verify_followed_system("Just plain text.", direction="a") is False

    def test_json_fails_user_a(self, conflict):
        assert conflict.verify_followed_user('{"answer": "hello"}', direction="a") is False

    # Edge: JSON array (not object -> fails is_valid_json_object)
    def test_json_array_not_object(self, conflict):
        assert conflict.verify_followed_system("[1, 2, 3]", direction="a") is False

    # Edge: nested JSON object
    def test_nested_json(self, conflict):
        text = '{"a": {"b": [1, 2]}, "c": "value"}'
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: JSON with whitespace
    def test_json_with_whitespace(self, conflict):
        text = '  \n  {"key": "val"}  \n  '
        assert conflict.verify_followed_system(text, direction="a") is True

    # Edge: malformed JSON — has key-value pattern, accepted as truncated JSON
    def test_malformed_json(self, conflict):
        assert conflict.verify_followed_system('{"key": value}', direction="a") is True

    # Edge: JSON-like but missing closing brace — accepted as truncated JSON
    def test_unclosed_json(self, conflict):
        assert conflict.verify_followed_system('{"key": "val"', direction="a") is True

    # Edge: empty object
    def test_empty_object(self, conflict):
        assert conflict.verify_followed_system("{}", direction="a") is True

    # Edge: text that starts with { but isn't JSON
    def test_curly_brace_not_json(self, conflict):
        assert conflict.verify_followed_system("{this is not json}", direction="a") is False

    # Edge: empty string
    def test_empty_string_system(self, conflict):
        assert conflict.verify_followed_system("", direction="a") is False

    def test_empty_string_user(self, conflict):
        assert conflict.verify_followed_user("", direction="a") is True

    # Direction b: system=not json, user=json
    def test_direction_b(self, conflict):
        assert conflict.verify_followed_system("Plain text.", direction="b") is True
        assert conflict.verify_followed_user('{"key": "val"}', direction="b") is True

    # Edge: string "null" is not a JSON object
    def test_null_json(self, conflict):
        assert conflict.verify_followed_system("null", direction="a") is False

    # Edge: boolean JSON
    def test_boolean_json(self, conflict):
        assert conflict.verify_followed_system("true", direction="a") is False

    # Edge: response with JSON embedded in text
    def test_embedded_json_in_text(self, conflict):
        text = 'Here is the answer: {"key": "val"}'
        assert conflict.verify_followed_system(text, direction="a") is False


