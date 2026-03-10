"""Tests for sentence_connector_density conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.sentence_connector_density import (
    _score_connector_density,
    _score_no_connectors,
    SentenceConnectorDensityConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("sentence_connector_density")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "sentence_connector_density"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_arg_keys_empty(self, conflict):
        assert conflict.arg_keys == [] or conflict.get_instruction_args_keys() == []

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b")
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_templates_no_placeholders(self, conflict):
        """Templates should have no unfilled placeholders."""
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_threshold_set(self, conflict):
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The no-connectors scorer should have is_inverted=True."""
        assert getattr(_score_no_connectors, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(_score_connector_density, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreConnectorDensity:
    def test_heavy_connectors(self):
        """Text with many transition words should score high."""
        text = (
            "However, the situation changed dramatically. Furthermore, new evidence "
            "emerged. Moreover, experts agreed on the findings. Additionally, the "
            "public responded positively. Consequently, the policy was adopted."
        )
        score = _score_connector_density(text)
        assert score >= 0.8

    def test_no_connectors(self):
        """Plain text without transition words should score near zero."""
        text = (
            "The cat sat on the mat. The dog ran in the yard. "
            "Birds flew over the trees. The sun was bright."
        )
        score = _score_connector_density(text)
        assert score == 0.0

    def test_mixed_connectors(self):
        """Some sentences with connectors, some without."""
        text = (
            "The project started in January. However, delays occurred in March. "
            "The team worked hard. Therefore, the deadline was met."
        )
        score = _score_connector_density(text)
        assert 0.4 <= score <= 0.6  # 2 out of 4 sentences

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "However, the data shows improvement. The trend is clear. Moreover, costs decreased."
        direct = _score_connector_density(text)
        inverted = _score_no_connectors(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "However. Moreover. Thus.", "The cat sat."]:
            score = _score_connector_density(text)
            assert 0.0 <= score <= 1.0

    def test_multi_word_connectors(self):
        """Multi-word phrases like 'in addition' should be detected."""
        text = "The first point is clear. In addition, the second point strengthens the argument."
        score = _score_connector_density(text)
        assert score >= 0.4  # at least 1 of 2 sentences

    def test_case_insensitive(self):
        """Connectors should be detected regardless of case."""
        text = "HOWEVER, this is important. MOREOVER, it is urgent."
        score = _score_connector_density(text)
        assert score >= 0.8


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_connector_density("") == 0.0
        assert _score_no_connectors("") == 1.0

    def test_single_sentence_with_connector(self):
        score = _score_connector_density("However, the situation improved.")
        assert score == 1.0

    def test_single_sentence_without_connector(self):
        score = _score_connector_density("The situation improved.")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = SentenceConnectorDensityConflict()
        # Text with many connectors should pass system check in direction a
        with_connectors = (
            "However, the data is clear. Moreover, the trend continues. "
            "Furthermore, experts agree. Additionally, the public supports this."
        )
        assert c.verify_followed_system(with_connectors, direction="a")
        # Text without connectors should fail system check in direction a
        without_connectors = "The cat sat on the mat. The dog ran in the yard."
        assert not c.verify_followed_system(without_connectors, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = SentenceConnectorDensityConflict()
        # Text without connectors should pass system check in direction b
        without_connectors = "The cat sat on the mat. The dog ran in the yard."
        assert c.verify_followed_system(without_connectors, direction="b")
        # Text with many connectors should fail system check in direction b
        with_connectors = (
            "However, the data is clear. Moreover, the trend continues. "
            "Furthermore, experts agree."
        )
        assert not c.verify_followed_system(with_connectors, direction="b")

    def test_also_detected(self):
        """'also' is a common connector and should be detected."""
        text = "The team also considered alternative approaches."
        score = _score_connector_density(text)
        assert score == 1.0

    def test_yet_detected(self):
        """'yet' as connector should be detected."""
        text = "The results were promising. Yet the challenges remained significant."
        score = _score_connector_density(text)
        assert score >= 0.4
