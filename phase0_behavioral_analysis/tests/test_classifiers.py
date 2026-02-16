"""
Unit tests for response classifiers.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings

from src.classifiers import (
    ClassificationResult,
    LanguageClassifier,
    FormatClassifier,
    get_classifier,
    compute_label,
    LOW_CONFIDENCE_THRESHOLD
)


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_basic_creation(self):
        result = ClassificationResult(detected='english', confidence=0.95)
        assert result.detected == 'english'
        assert result.confidence == 0.95
        assert result.details is None

    def test_with_details(self):
        result = ClassificationResult(detected='english', confidence=0.95, details={'en': 0.95, 'es': 0.05})
        assert result.details == {'en': 0.95, 'es': 0.05}

    def test_is_low_confidence_true(self):
        result = ClassificationResult(detected='english', confidence=0.5)
        assert result.is_low_confidence is True

    def test_is_low_confidence_false(self):
        result = ClassificationResult(detected='english', confidence=0.9)
        assert result.is_low_confidence is False

    def test_is_low_confidence_at_threshold(self):
        result = ClassificationResult(detected='english', confidence=LOW_CONFIDENCE_THRESHOLD)
        assert result.is_low_confidence is False


class TestLanguageClassifier:
    """Tests for LanguageClassifier."""

    @pytest.fixture
    def classifier(self):
        return LanguageClassifier()

    def test_english_text(self, classifier):
        text = "The capital of France is Paris. It is a beautiful city."
        result = classifier.classify(text)
        assert result.detected == 'english'
        assert result.confidence > 0.8

    def test_spanish_text(self, classifier):
        text = "La capital de Francia es París. Es una ciudad hermosa."
        result = classifier.classify(text)
        assert result.detected == 'spanish'
        assert result.confidence > 0.8

    def test_empty_text(self, classifier):
        result = classifier.classify("")
        assert result.detected == 'other'
        assert result.confidence == 0.0
        assert 'error' in result.details

    def test_whitespace_only(self, classifier):
        result = classifier.classify("   \n\t  ")
        assert result.detected == 'other'
        assert result.confidence == 0.0

    def test_french_detected(self, classifier):
        text = "Bonjour, comment allez-vous? Je suis très content."
        result = classifier.classify(text)
        assert result.detected == 'french'

    def test_truly_other_language(self, classifier):
        text = "こんにちは、お元気ですか？今日はとても良い天気ですね。"
        result = classifier.classify(text)
        assert result.detected == 'other'

    def test_details_contains_probabilities(self, classifier):
        text = "Hello, how are you today?"
        result = classifier.classify(text)
        assert result.details is not None
        assert 'en' in result.details


class TestFormatClassifier:
    """Tests for FormatClassifier (handles JSON, YAML, and plain text)."""

    @pytest.fixture
    def classifier(self):
        return FormatClassifier()

    # --- JSON detection ---

    def test_valid_json_object(self, classifier):
        text = '{"name": "Paris", "country": "France"}'
        result = classifier.classify(text)
        assert result.detected == 'json'
        assert result.confidence == 1.0

    def test_valid_json_array(self, classifier):
        text = '["Paris", "London", "Berlin"]'
        result = classifier.classify(text)
        assert result.detected == 'json'
        assert result.confidence == 1.0

    def test_nested_json(self, classifier):
        text = '{"city": {"name": "Paris", "population": 2161000}}'
        result = classifier.classify(text)
        assert result.detected == 'json'
        assert result.confidence == 1.0

    def test_json_with_boolean_and_null(self, classifier):
        text = '{"active": true, "deleted": false, "data": null}'
        result = classifier.classify(text)
        assert result.detected == 'json'
        assert result.confidence == 1.0

    def test_json_with_surrounding_text(self, classifier):
        text = 'Here is the answer: {"capital": "Paris"}'
        result = classifier.classify(text)
        assert result.detected == 'json'

    def test_json_in_code_fence(self, classifier):
        text = '```json\n{"capital": "Paris"}\n```'
        result = classifier.classify(text)
        assert result.detected == 'json'

    # --- YAML detection ---

    def test_yaml_document_start_marker(self, classifier):
        text = "---\nname: Paris\ncountry: France\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_document_end_marker(self, classifier):
        text = "---\nname: Paris\n...\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_literal_block_scalar(self, classifier):
        text = "description: |\n  This is a multi-line\n  literal block scalar\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_folded_block_scalar(self, classifier):
        text = "description: >\n  This is a folded\n  block scalar\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_anchor(self, classifier):
        text = "defaults: &defaults\n  adapter: postgres\n  host: localhost\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_tag(self, classifier):
        text = "timestamp: !timestamp 2024-01-15\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0

    def test_yaml_list_of_mappings(self, classifier):
        text = "- name: Paris\n  country: France\n- name: London\n  country: UK\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'

    def test_yaml_key_value_parsed(self, classifier):
        text = "capital: Paris\ncountry: France\n"
        result = classifier.classify(text)
        assert result.detected == 'yaml'

    def test_yaml_in_code_fence(self, classifier):
        """The key bug fix — YAML wrapped in markdown code fences."""
        text = '```yaml\nquestion: What is the capital of France?\nanswer: Paris\n```'
        result = classifier.classify(text)
        assert result.detected == 'yaml'

    def test_yaml_in_code_fence_no_lang(self, classifier):
        text = '```\ncapital: Paris\ncountry: France\n```'
        result = classifier.classify(text)
        assert result.detected == 'yaml'

    # --- Plain text detection ---

    def test_plain_text(self, classifier):
        text = "The capital of France is Paris."
        result = classifier.classify(text)
        assert result.detected == 'plain'
        assert result.confidence > 0.5

    def test_empty_text(self, classifier):
        result = classifier.classify("")
        assert result.detected == 'plain'
        assert result.confidence == 1.0

    def test_whitespace_only(self, classifier):
        result = classifier.classify("   \n\t  ")
        assert result.detected == 'plain'
        assert result.confidence == 1.0

    def test_plain_prose(self, classifier):
        text = "The capital of France is Paris. It is a beautiful city with many attractions."
        result = classifier.classify(text)
        assert result.detected == 'plain'


class TestGetClassifier:
    """Tests for get_classifier factory function."""

    def test_get_language_classifier(self):
        classifier = get_classifier('language')
        assert isinstance(classifier, LanguageClassifier)

    def test_get_format_classifier(self):
        classifier = get_classifier('format')
        assert isinstance(classifier, FormatClassifier)

    def test_get_yaml_returns_format_classifier(self):
        """'yaml' maps to FormatClassifier (same as 'format')."""
        classifier = get_classifier('yaml')
        assert isinstance(classifier, FormatClassifier)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match='Unknown constraint type'):
            get_classifier('invalid')


class TestComputeLabel:
    """Tests for compute_label function."""

    def test_followed_system_in_conflict(self):
        classification = ClassificationResult(detected='english', confidence=0.95)
        label, confidence = compute_label(classification, 'english', 'spanish')
        assert label == 'followed_system'
        assert confidence == 0.95

    def test_followed_user_in_conflict(self):
        classification = ClassificationResult(detected='spanish', confidence=0.9)
        label, confidence = compute_label(classification, 'english', 'spanish')
        assert label == 'followed_user'
        assert confidence == 0.9

    def test_followed_neither_in_conflict(self):
        classification = ClassificationResult(detected='other', confidence=0.8)
        label, confidence = compute_label(classification, 'english', 'spanish')
        assert label == 'followed_neither'
        assert confidence == 0.8

    def test_followed_system_no_user_constraint(self):
        classification = ClassificationResult(detected='english', confidence=0.95)
        label, _ = compute_label(classification, 'english', None)
        assert label == 'followed_system'

    def test_followed_neither_no_user_constraint(self):
        classification = ClassificationResult(detected='spanish', confidence=0.9)
        label, _ = compute_label(classification, 'english', None)
        assert label == 'followed_neither'

    def test_followed_both_same_constraints(self):
        classification = ClassificationResult(detected='english', confidence=0.95)
        label, _ = compute_label(classification, 'english', 'english')
        assert label == 'followed_both'

    def test_case_insensitive_matching(self):
        classification = ClassificationResult(detected='english', confidence=0.95)
        label, _ = compute_label(classification, 'ENGLISH', 'SPANISH')
        assert label == 'followed_system'


class TestFormatClassifierPropertyTests:
    """Property-based tests for FormatClassifier."""

    @given(st.dictionaries(
        keys=st.text(alphabet=st.characters(whitelist_categories=('L', 'N')), min_size=1, max_size=10),
        values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
        min_size=1, max_size=5
    ))
    @settings(max_examples=50)
    def test_valid_json_always_returns_json(self, data):
        import json as _json
        classifier = FormatClassifier()
        json_text = _json.dumps(data)
        result = classifier.classify(json_text)
        assert result.detected == 'json'
        assert result.confidence == 1.0

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_classification_always_returns_valid_format(self, text):
        classifier = FormatClassifier()
        result = classifier.classify(text)
        assert result.detected in ('yaml', 'json', 'plain')
        assert 0.0 <= result.confidence <= 1.0

    @given(st.sampled_from([
        "---\nkey: value\n",
        "key: |\n  multiline\n  text\n",
        "key: >\n  folded\n  text\n",
        "- item1\n- item2\n",
        "---\nname: test\n...\n",
    ]))
    def test_yaml_specific_syntax_returns_yaml(self, yaml_text):
        classifier = FormatClassifier()
        result = classifier.classify(yaml_text)
        assert result.detected == 'yaml'
        assert result.confidence == 1.0



class TestStartingWordClassifier:
    """Tests for StartingWordClassifier."""

    @pytest.fixture
    def classifier(self):
        from src.classifiers import StartingWordClassifier
        return StartingWordClassifier()

    def test_simple_word(self, classifier):
        result = classifier.classify("Hello, how are you?")
        assert result.detected == 'hello'
        assert result.confidence == 1.0

    def test_greetings(self, classifier):
        result = classifier.classify("Greetings! I hope you are well.")
        assert result.detected == 'greetings'
        assert result.confidence == 1.0

    def test_case_insensitive(self, classifier):
        result = classifier.classify("HELLO there!")
        assert result.detected == 'hello'

    def test_strips_punctuation(self, classifier):
        result = classifier.classify('"Hello," she said.')
        assert result.detected == 'hello'

    def test_markdown_header(self, classifier):
        result = classifier.classify("# Hello World")
        assert result.detected == 'hello'

    def test_bold_text(self, classifier):
        result = classifier.classify("**Hello** everyone!")
        assert result.detected == 'hello'

    def test_empty_text(self, classifier):
        result = classifier.classify("")
        assert result.detected == 'other'
        assert result.confidence == 0.0

    def test_whitespace_only(self, classifier):
        result = classifier.classify("   \n\t  ")
        assert result.detected == 'other'
        assert result.confidence == 0.0

    def test_details_contains_raw_word(self, classifier):
        result = classifier.classify("Hello!")
        assert result.details['raw_first_word'] == 'Hello!'
        assert result.details['normalized'] == 'hello'


class TestGetClassifierStartingWord:
    """Tests for get_classifier with starting_word type."""

    def test_get_starting_word_classifier(self):
        from src.classifiers import StartingWordClassifier
        classifier = get_classifier('starting_word')
        assert isinstance(classifier, StartingWordClassifier)
