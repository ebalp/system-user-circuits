"""Test conflicts batch 5: formal_vs_casual_tone, numbered_sections_vs_prose,
short_paragraphs_vs_single_block, imperative_vs_declarative, direct_answer_vs_hedging."""

import pytest
from phase0_v2.conflicts.registry import get_conflict, get_conflict_ids
from phase0_v2.conflicts.compatibility import (
    INCOMPATIBLE, EXPLICITLY_COMPATIBLE, validate_matrix_coverage,
)

# Import the underlying scoring/detection functions for direct testing
from phase0_v2.conflicts.definitions.formal_vs_casual_tone import (
    _score_formality, _score_casualness,
)
from phase0_v2.conflicts.definitions.numbered_sections_vs_prose import (
    _has_numbered_sections, _is_continuous_prose,
)
from phase0_v2.conflicts.definitions.short_paragraphs_vs_single_block import (
    _has_short_paragraphs, _is_single_block,
)
from phase0_v2.conflicts.definitions.imperative_vs_declarative import (
    score_imperative, _score_declarative,
)
from phase0_v2.conflicts.definitions.direct_answer_vs_hedging import (
    _score_directness, _score_hedging,
)


BATCH5_IDS = [
    "formal_vs_casual_tone",
    "numbered_sections_vs_prose",
    "short_paragraphs_vs_single_block",
    "imperative_vs_declarative",
    "direct_answer_vs_hedging",
]


@pytest.fixture(params=BATCH5_IDS)
def conflict(request):
    c = get_conflict(request.param)
    assert c is not None, f"Conflict '{request.param}' not registered"
    return c


# ── Contract tests for all 5 ──


class TestBatch5Contract:
    def test_build_direction_a(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_build_direction_b(self, conflict):
        """All batch 5 conflicts have full counterbalancing."""
        args = conflict.sample_args()
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_counterbalance_quality_is_full(self, conflict):
        assert conflict.counterbalance_quality == "full"
        assert conflict.supports_counterbalancing()

    def test_directions_differ(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        sys_b = conflict.build_system_prompt(direction="b", **args)
        assert sys_a != sys_b

    def test_verify_returns_bool(self, conflict):
        args = conflict.sample_args()
        conflict.build_system_prompt(direction="a", **args)
        assert isinstance(conflict.verify_followed_system("test text", direction="a"), bool)
        assert isinstance(conflict.verify_followed_user("test text", direction="a"), bool)

    def test_score_returns_float(self, conflict):
        args = conflict.sample_args()
        conflict.build_system_prompt(direction="a", **args)
        sys_score = conflict.score_system("test text", direction="a")
        usr_score = conflict.score_user("test text", direction="a")
        assert isinstance(sys_score, float)
        assert isinstance(usr_score, float)
        assert 0.0 <= sys_score <= 1.0
        assert 0.0 <= usr_score <= 1.0

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}

    def test_arg_keys_empty(self, conflict):
        assert conflict.get_instruction_args_keys() == []

    def test_registered_in_registry(self, conflict):
        assert conflict.conflict_id in get_conflict_ids()

    def test_in_compatibility_matrix(self, conflict):
        uncovered = validate_matrix_coverage([conflict.conflict_id])
        assert uncovered == [], f"{conflict.conflict_id} not in compatibility matrix"


# ── Formal vs Casual Tone ──


class TestFormalVsCasualTone:
    def test_formal_text_high_formality_score(self):
        text = (
            "It is important to note that the organization has implemented "
            "several measures to ensure compliance. The committee will not "
            "tolerate any deviations from the established protocol."
        )
        score = _score_formality(text)
        assert score >= 0.9, f"Expected high formality, got {score}"

    def test_casual_text_low_formality_score(self):
        text = (
            "I'm gonna tell you what's up. It's really cool and you'll "
            "love it. Don't worry, it won't take long. I can't wait!"
        )
        score = _score_formality(text)
        assert score < 0.7, f"Expected low formality for casual text, got {score}"

    def test_scores_are_complementary(self):
        text = "I don't think it's a good idea but we can't stop now."
        f = _score_formality(text)
        c = _score_casualness(text)
        assert abs(f + c - 1.0) < 1e-9, "Scores must sum to 1.0"

    def test_empty_text_is_formal(self):
        assert _score_formality("") == 1.0
        assert _score_casualness("") == 0.0

    def test_casualness_is_inverted(self):
        assert getattr(_score_casualness, "is_inverted", False) is True

    def test_verify_mutual_exclusion_direction_a(self):
        """With default threshold, formal and casual should not both pass on same text."""
        c = get_conflict("formal_vs_casual_tone")
        c.build_system_prompt(direction="a")
        # Formal text
        formal = (
            "The committee has determined that it is necessary to proceed. "
            "We will not delay the implementation any further."
        )
        sys_pass = c.verify_followed_system(formal, direction="a")
        usr_pass = c.verify_followed_user(formal, direction="a")
        # Should not both be True (mutual exclusion for 1-score pairs)
        assert not (sys_pass and usr_pass), "Formal text should not pass both sides"

    def test_verify_mutual_exclusion_direction_b(self):
        c = get_conflict("formal_vs_casual_tone")
        c.build_system_prompt(direction="b")
        casual = (
            "I'm sure you're gonna love this. It's awesome and I can't "
            "believe we didn't try it sooner. Let's do it!"
        )
        sys_pass = c.verify_followed_system(casual, direction="b")
        usr_pass = c.verify_followed_user(casual, direction="b")
        assert not (sys_pass and usr_pass), "Casual text should not pass both sides"


# ── Numbered Sections vs Prose ──


class TestNumberedSectionsVsProse:
    def test_numbered_sections_detected(self):
        text = (
            "1. Introduction to the topic.\n"
            "2. Main discussion points.\n"
            "3. Conclusion and summary."
        )
        assert _has_numbered_sections(text) is True

    def test_fewer_than_three_sections_rejected(self):
        text = (
            "1. First point.\n"
            "2. Second point."
        )
        assert _has_numbered_sections(text) is False

    def test_prose_without_numbers(self):
        text = (
            "This is a flowing paragraph of text that discusses the topic "
            "without using any numbered sections or lists. It continues "
            "naturally from one sentence to the next."
        )
        assert _is_continuous_prose(text) is True
        assert _has_numbered_sections(text) is False

    def test_complementary_functions(self):
        numbered = "1. First.\n2. Second.\n3. Third."
        assert _has_numbered_sections(numbered) is True
        assert _is_continuous_prose(numbered) is False

    def test_prose_is_complementary_of_numbered(self):
        prose = "This is just a simple paragraph about the weather."
        assert _is_continuous_prose(prose) is True
        assert _has_numbered_sections(prose) is False

    def test_verify_direction_a(self):
        c = get_conflict("numbered_sections_vs_prose")
        c.build_system_prompt(direction="a")
        numbered = "1. First.\n2. Second.\n3. Third."
        assert c.verify_followed_system(numbered, direction="a") is True
        assert c.verify_followed_user(numbered, direction="a") is False

    def test_verify_direction_b(self):
        c = get_conflict("numbered_sections_vs_prose")
        c.build_system_prompt(direction="b")
        prose = "This is flowing text without any numbered sections."
        assert c.verify_followed_system(prose, direction="b") is True

    def test_numbered_sections_with_content_after(self):
        text = (
            "1. Introduction to algorithms.\n"
            "Algorithms are important in computer science.\n"
            "2. Types of sorting algorithms.\n"
            "There are many sorting approaches.\n"
            "3. Conclusion.\n"
            "In summary, algorithms are essential."
        )
        assert _has_numbered_sections(text) is True


# ── Short Paragraphs vs Single Block ──


class TestShortParagraphsVsSingleBlock:
    def test_short_paragraphs_detected(self):
        text = (
            "First paragraph here. It is short.\n\n"
            "Second paragraph now. Also concise.\n\n"
            "Third paragraph done. Very brief."
        )
        assert _has_short_paragraphs(text) is True

    def test_single_block_detected(self):
        text = (
            "This is a single continuous paragraph without any line breaks "
            "separating it into multiple paragraphs. It just flows on."
        )
        assert _is_single_block(text) is True

    def test_too_few_paragraphs_rejected(self):
        text = "One paragraph. Two sentences.\n\nSecond paragraph."
        assert _has_short_paragraphs(text) is False

    def test_long_paragraphs_rejected(self):
        text = (
            "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence here.\n\n"
            "Another paragraph. Short one.\n\n"
            "Yet another paragraph. Also short."
        )
        assert _has_short_paragraphs(text) is False

    def test_complementary(self):
        single = "Just one paragraph without any breaks at all."
        assert _is_single_block(single) is True
        assert _has_short_paragraphs(single) is False

    def test_verify_direction_a(self):
        c = get_conflict("short_paragraphs_vs_single_block")
        c.build_system_prompt(direction="a")
        short = "First. Short.\n\nSecond. Also short.\n\nThird. Brief."
        assert c.verify_followed_system(short, direction="a") is True
        single = "One continuous paragraph without breaks."
        assert c.verify_followed_user(single, direction="a") is True

    def test_verify_direction_b(self):
        c = get_conflict("short_paragraphs_vs_single_block")
        c.build_system_prompt(direction="b")
        single = "One continuous paragraph without any breaks."
        assert c.verify_followed_system(single, direction="b") is True

    def test_empty_lines_between_paragraphs(self):
        text = (
            "First paragraph.\n\n\n"
            "Second paragraph.\n\n\n"
            "Third paragraph."
        )
        assert _has_short_paragraphs(text) is True


# ── Imperative vs Declarative ──


class TestImperativeVsDeclarative:
    def test_imperative_text_high_score(self):
        text = (
            "Consider the evidence carefully. "
            "Note the trends in the data. "
            "Examine each variable closely."
        )
        score = score_imperative(text)
        assert score >= 0.7, f"Expected high imperative score, got {score}"

    def test_declarative_text_low_imperative_score(self):
        text = (
            "The evidence shows clear results. "
            "The trends in the data are significant. "
            "Each variable reveals important patterns."
        )
        score = score_imperative(text)
        assert score < 0.3, f"Expected low imperative score for declarative text, got {score}"

    def test_scores_complementary(self):
        text = "Consider the evidence. The data is clear."
        a = score_imperative(text)
        d = _score_declarative(text)
        assert abs(a + d - 1.0) < 1e-9, "Scores must sum to 1.0"

    def test_declarative_is_inverted(self):
        assert getattr(_score_declarative, "is_inverted", False) is True

    def test_empty_text(self):
        assert score_imperative("") == 0.5
        assert _score_declarative("") == 0.5

    def test_verify_mutual_exclusion(self):
        c = get_conflict("imperative_vs_declarative")
        c.build_system_prompt(direction="a")
        text = "Consider the facts. Note the trends. Examine the data."
        sys_pass = c.verify_followed_system(text, direction="a")
        usr_pass = c.verify_followed_user(text, direction="a")
        assert not (sys_pass and usr_pass), "Should not pass both sides"


# ── Direct Answer vs Hedging ──


class TestDirectAnswerVsHedging:
    def test_direct_text_high_score(self):
        text = (
            "The answer is 42. This approach works. "
            "The data confirms the hypothesis. "
            "We must act now."
        )
        score = _score_directness(text)
        assert score >= 0.9, f"Expected high directness for firm text, got {score}"

    def test_hedging_text_low_directness(self):
        text = (
            "Perhaps the answer might be 42. It seems likely that this "
            "could be the case. I think it is possible that maybe we "
            "should consider this approach, presumably."
        )
        score = _score_directness(text)
        assert score < 0.7, f"Expected low directness for hedging text, got {score}"

    def test_scores_complementary(self):
        text = "Perhaps this might work."
        d = _score_directness(text)
        h = _score_hedging(text)
        assert abs(d + h - 1.0) < 1e-9, "Scores must sum to 1.0"

    def test_hedging_is_inverted(self):
        assert getattr(_score_hedging, "is_inverted", False) is True

    def test_empty_text(self):
        assert _score_directness("") == 1.0
        assert _score_hedging("") == 0.0

    def test_verify_direction_a_and_b(self):
        c = get_conflict("direct_answer_vs_hedging")
        c.build_system_prompt(direction="a")
        direct = "The answer is clear. No doubt about it."
        sys_a = c.verify_followed_system(direct, direction="a")
        assert isinstance(sys_a, bool)

        c.build_system_prompt(direction="b")
        hedged = "Perhaps maybe this might possibly work, I think."
        sys_b = c.verify_followed_system(hedged, direction="b")
        assert isinstance(sys_b, bool)
