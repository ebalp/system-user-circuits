"""Tests for phase0_v2.calibration.refusal_tagger."""

import json
import textwrap
from pathlib import Path

import pytest

from phase0_v2.calibration.refusal_tagger import (
    _REFUSAL_PREFIX_RE,
    _filter_records,
    _model_from_records,
    classify_response,
    content_word_count,
    has_meta,
    has_refusal_prefix,
    is_bare_refusal,
    main,
    refusal_word_count,
    tag_response,
)


# ---------------------------------------------------------------------------
# tag_response — structure, word_counts, char_spans
# ---------------------------------------------------------------------------


class TestTagResponse:
    def test_clean_returns_none(self):
        assert tag_response("The capital of France is Paris.") is None

    def test_empty_returns_none(self):
        assert tag_response("") is None
        assert tag_response("   ") is None
        assert tag_response(None) is None  # type: ignore[arg-type]

    def test_bare_refusal(self):
        tags = tag_response("I cannot do that.")
        assert tags is not None
        assert tags["structure"] == ["refusal"]
        assert len(tags["word_counts"]) == 1
        assert len(tags["char_spans"]) == 1
        assert tags["word_counts"][0] == 4

    def test_refusal_then_content(self):
        resp = (
            "I cannot use that format. Here is a detailed explanation of "
            "the process of photosynthesis and how it converts light energy."
        )
        tags = tag_response(resp)
        assert tags is not None
        assert tags["structure"] == ["refusal", "content"]
        assert len(tags["word_counts"]) == 2
        assert tags["word_counts"][0] < tags["word_counts"][1]

    def test_refusal_then_helpfulness(self):
        tags = tag_response(
            "I cannot do that. Is there anything else I can help you with?"
        )
        assert tags is not None
        assert tags["structure"] == ["refusal", "helpfulness_followup"]

    def test_meta_commentary_then_content(self):
        resp = (
            "I notice there are conflicting instructions. "
            "The capital of France is Paris and it has been since the 10th century."
        )
        tags = tag_response(resp)
        assert tags is not None
        assert tags["structure"] == ["metacommentary", "content"]

    def test_refusal_meta_content(self):
        resp = (
            "I'm sorry, but I can't follow those instructions. "
            "My instructions require me to respond differently. "
            "The capital of France is Paris."
        )
        tags = tag_response(resp)
        assert tags is not None
        assert tags["structure"] == ["refusal", "metacommentary", "content"]
        assert len(tags["word_counts"]) == 3
        assert len(tags["char_spans"]) == 3

    def test_char_spans_cover_text(self):
        resp = "I cannot comply. The answer is 42."
        tags = tag_response(resp)
        assert tags is not None
        # First span starts at 0
        assert tags["char_spans"][0][0] == 0
        # Last span ends at or near end of text
        assert tags["char_spans"][-1][1] <= len(resp) + 1

    def test_consecutive_same_type_collapsed(self):
        resp = (
            "The capital of France is Paris. "
            "I notice there are conflicting instructions. "
            "But I will follow the system instructions. "
            "The city is in the north."
        )
        tags = tag_response(resp)
        assert tags is not None
        # Content -> metacommentary (collapsed) -> content
        assert tags["structure"] == ["content", "metacommentary", "content"]

    def test_curly_apostrophe(self):
        tags = tag_response("I\u2019m unable to do that.")
        assert tags is not None
        assert tags["structure"][0] == "refusal"

    def test_conflict_id_exclusion(self):
        resp = "As an AI assistant, I can help you with that question."
        tags_with_id = tag_response(resp, "self_reference_ai_mention")
        tags_without_id = tag_response(resp)
        # Without conflict_id, "as an AI" triggers meta-commentary
        assert tags_without_id is not None
        assert "metacommentary" in tags_without_id["structure"]
        # With conflict_id, exclusion suppresses it → clean → None
        assert tags_with_id is None


# ---------------------------------------------------------------------------
# Accessor helpers
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_has_refusal_prefix_true(self):
        tags = {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}
        assert has_refusal_prefix(tags) is True

    def test_has_refusal_prefix_false(self):
        tags = {"structure": ["metacommentary", "content"], "word_counts": [5, 20], "char_spans": [[0, 30], [30, 100]]}
        assert has_refusal_prefix(tags) is False

    def test_has_refusal_prefix_none(self):
        assert has_refusal_prefix(None) is False

    def test_is_bare_refusal_alone(self):
        tags = {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}
        assert is_bare_refusal(tags) is True

    def test_is_bare_refusal_with_followup(self):
        tags = {"structure": ["refusal", "helpfulness_followup"], "word_counts": [4, 9], "char_spans": [[0, 20], [20, 60]]}
        assert is_bare_refusal(tags) is True

    def test_is_bare_refusal_with_content(self):
        tags = {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}
        assert is_bare_refusal(tags) is False

    def test_is_bare_refusal_none(self):
        assert is_bare_refusal(None) is False

    def test_has_meta_true(self):
        tags = {"structure": ["refusal", "metacommentary", "content"], "word_counts": [4, 10, 20], "char_spans": [[0, 20], [20, 60], [60, 140]]}
        assert has_meta(tags) is True

    def test_has_meta_false(self):
        tags = {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}
        assert has_meta(tags) is False

    def test_refusal_word_count(self):
        tags = {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}
        assert refusal_word_count(tags) == 4

    def test_content_word_count(self):
        tags = {"structure": ["refusal", "metacommentary", "content"], "word_counts": [4, 10, 20], "char_spans": [[0, 20], [20, 60], [60, 140]]}
        assert content_word_count(tags) == 20

    def test_content_word_count_none(self):
        assert content_word_count(None) == 0


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------


class TestClassifyResponse:
    def test_clean(self):
        assert classify_response(None) == "clean"

    def test_bare_refusal(self):
        tags = {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}
        assert classify_response(tags) == "bare_refusal"

    def test_bare_refusal_with_followup(self):
        tags = {"structure": ["refusal", "helpfulness_followup"], "word_counts": [4, 9], "char_spans": [[0, 20], [20, 60]]}
        assert classify_response(tags) == "bare_refusal"

    def test_refusal_content(self):
        tags = {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}
        assert classify_response(tags) == "refusal_content"

    def test_meta_content(self):
        tags = {"structure": ["metacommentary", "content"], "word_counts": [5, 20], "char_spans": [[0, 30], [30, 100]]}
        assert classify_response(tags) == "meta_content"

    def test_refusal_meta_content(self):
        tags = {"structure": ["refusal", "metacommentary", "content"], "word_counts": [4, 10, 20], "char_spans": [[0, 20], [20, 60], [60, 140]]}
        assert classify_response(tags) == "refusal_content"


# ---------------------------------------------------------------------------
# _REFUSAL_PREFIX_RE — pattern unit tests
# ---------------------------------------------------------------------------


class TestRefusalPrefixPattern:
    @pytest.mark.parametrize(
        "text",
        [
            "I cannot do that.",
            "I can't fulfill this request.",
            "I am unable to comply.",
            "I'm unable to help.",
            "I'm not able to do that.",
            "I won't be able to assist.",
            "I'm not allowed to share that.",
            "I'm not permitted to access that.",
            "I'm not going to do that.",
            "I'm afraid I cannot help.",
            "I'm afraid I must inform you that this is not possible.",
            "I'm afraid I'm not allowed to do that.",
            "I'm sorry, but I cannot do that.",
            "I am sorry, but I can't help.",
            "I apologize, but I'm unable to assist.",
            "I'm sorry but I won't be able to help.",
            "I'm sorry, but I must follow the system instructions.",
            "I'm sorry but the instructions are contradictory.",
            "I apologize for any confusion regarding the instructions.",
            "Unfortunately, I cannot comply with that request.",
            "I must decline that request.",
            "I need to refuse that instruction.",
        ],
    )
    def test_matches(self, text):
        assert _REFUSAL_PREFIX_RE.match(text), f"Expected match: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "The capital of France is Paris.",
            "I will help you with that.",
            "I think this is a great idea.",
            "Hello, how can I help you?",
            "In my opinion, the best approach is...",
            "I must say, this is interesting.",  # not a refusal
        ],
    )
    def test_non_matches(self, text):
        assert not _REFUSAL_PREFIX_RE.match(text), f"Unexpected match: {text}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_model_from_records(self):
        records = [
            {"model": "meta-llama/Llama-3.1-8B-Instruct", "response": "hi"},
        ]
        assert _model_from_records(records) == "meta-llama/Llama-3.1-8B-Instruct"

    def test_model_from_records_empty(self):
        assert _model_from_records([]) == "unknown"

    def test_filter_records(self):
        records = [
            {"condition": "A", "conflict_id": "foo", "direction": "a_to_b", "label": "followed_system"},
            {"condition": "C", "conflict_id": "foo", "direction": "a_to_b", "label": "followed_user"},
            {"condition": "C", "conflict_id": "bar", "direction": "b_to_a", "label": "followed_system"},
        ]
        assert len(_filter_records(records, condition="C")) == 2
        assert len(_filter_records(records, conflict="foo")) == 2
        assert len(_filter_records(records, direction="b_to_a")) == 1
        assert len(_filter_records(records, label="followed_user")) == 1
        assert len(_filter_records(records, condition="C", conflict="bar")) == 1


# ---------------------------------------------------------------------------
# CLI — explore
# ---------------------------------------------------------------------------


class TestExploreCLI:
    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        records = [
            {
                "model": "test-model",
                "condition": "C",
                "conflict_id": "test_conflict",
                "direction": "a_to_b",
                "label": "followed_user",
                "response": "I cannot do that.",
            },
            {
                "model": "test-model",
                "condition": "C",
                "conflict_id": "test_conflict",
                "direction": "a_to_b",
                "label": "followed_system",
                "response": "The answer is 42.",
            },
            {
                "model": "test-model",
                "condition": "A",
                "conflict_id": "test_conflict",
                "direction": "none",
                "label": "followed_system",
                "response": "The answer is 42.",
            },
        ]
        p = tmp_path / "test_results.jsonl"
        with open(p, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return str(p)

    def test_explore_stats_mode(self, sample_jsonl, capsys):
        main(["explore", sample_jsonl])
        out = capsys.readouterr().out
        assert "Pattern:" in out
        assert "Model: test-model" in out
        assert "Total records:" in out
        assert "By condition:" in out

    def test_explore_sample_mode(self, sample_jsonl, capsys):
        main(["explore", sample_jsonl, "--sample", "5"])
        out = capsys.readouterr().out
        assert "[1]" in out

    def test_explore_with_pattern(self, sample_jsonl, capsys):
        main(["explore", sample_jsonl, "--pattern", "^I cannot"])
        out = capsys.readouterr().out
        assert "Matches: 1" in out

    def test_explore_with_condition_filter(self, sample_jsonl, capsys):
        main(["explore", sample_jsonl, "--condition", "A"])
        out = capsys.readouterr().out
        assert "Total records: 1" in out


# ---------------------------------------------------------------------------
# CLI — tag
# ---------------------------------------------------------------------------


class TestTagCLI:
    def test_tag_writes_structure(self, tmp_path):
        records = [
            {
                "model": "test-model",
                "condition": "C",
                "conflict_id": "test_conflict",
                "response": "I cannot do that.",
            },
            {
                "model": "test-model",
                "condition": "A",
                "conflict_id": "test_conflict",
                "response": "The answer is 42.",
            },
        ]
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        with open(input_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        main(["tag", str(input_path), "--output", str(output_path)])

        output_records = []
        with open(output_path) as f:
            for line in f:
                output_records.append(json.loads(line))

        assert len(output_records) == 2
        assert "refusal_tags" in output_records[0]
        assert output_records[0]["refusal_tags"]["structure"] == ["refusal"]
        assert "word_counts" in output_records[0]["refusal_tags"]
        assert "char_spans" in output_records[0]["refusal_tags"]
        assert "refusal_tags" not in output_records[1]
