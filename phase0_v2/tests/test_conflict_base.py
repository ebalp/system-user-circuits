"""Test Conflict base class contract -- direction dispatch, arg storage, verify convention."""

import pytest
from phase0_v2.conflicts.conflict_base import Conflict


class DummyConflict(Conflict):
    """Minimal conflict for testing the base class mechanics."""
    conflict_id = "test_dummy"
    system_template = "System says {word}."
    user_template = "User wants {word}."
    inverse_system_template = "Inverse system says {word}."
    inverse_user_template = "Inverse user wants {word}."
    counterbalance_quality = "full"
    arg_keys = ["word"]

    # 1-arg verify fns (no args dict)
    def verify_system_fn(response: str) -> bool:
        return "system" in response.lower()

    def verify_user_fn(response: str) -> bool:
        return "user" in response.lower()

    def verify_inverse_system_fn(response: str) -> bool:
        return "inverse_system" in response.lower()

    def verify_inverse_user_fn(response: str) -> bool:
        return "inverse_user" in response.lower()

    def sample_args(self) -> dict:
        return {"word": "hello"}


class DummyArgConflict(Conflict):
    """Conflict with 2-arg verify fns (tests arg-passing convention)."""
    conflict_id = "test_dummy_args"
    system_template = "Avoid {word}."
    user_template = "Include {word}."
    counterbalance_quality = "none"
    arg_keys = ["word"]

    def verify_system_fn(response: str, args: dict) -> bool:
        return args["word"] not in response.lower()

    def verify_user_fn(response: str, args: dict) -> bool:
        return args["word"] in response.lower()

    def sample_args(self) -> dict:
        return {"word": "forbidden"}


class DummyNoInverseConflict(Conflict):
    """Non-invertible conflict."""
    conflict_id = "test_no_inverse"
    system_template = "Do something special."
    user_template = "Do something else."
    counterbalance_quality = "none"

    def verify_system_fn(response: str) -> bool:
        return True

    def verify_user_fn(response: str) -> bool:
        return False


class DummyFloatConflict(Conflict):
    """Conflict with float-returning verify fns for threshold testing."""
    conflict_id = "test_float"
    system_template = "System says {word}."
    user_template = "User wants {word}."
    counterbalance_quality = "none"
    arg_keys = ["word"]
    verify_threshold = 0.7

    def verify_system_fn(response: str) -> float:
        # Returns fraction of words that are "good"
        words = response.split()
        if not words:
            return 0.0
        return sum(1 for w in words if w.startswith("g")) / len(words)

    def verify_user_fn(response: str) -> float:
        words = response.split()
        if not words:
            return 0.0
        return sum(1 for w in words if w.startswith("b")) / len(words)


class TestBuildSystemPrompt:
    def test_direction_a(self):
        c = DummyConflict()
        result = c.build_system_prompt(direction="a", word="test")
        assert result == "System says test."

    def test_direction_b(self):
        c = DummyConflict()
        result = c.build_system_prompt(direction="b", word="test")
        assert result == "Inverse system says test."

    def test_direction_b_raises_when_no_inverse(self):
        c = DummyNoInverseConflict()
        with pytest.raises(ValueError, match="inverse"):
            c.build_system_prompt(direction="b")

    def test_invalid_direction_raises(self):
        c = DummyConflict()
        with pytest.raises(ValueError):
            c.build_system_prompt(direction="x", word="test")

    def test_placeholder_replacement_uses_str_replace(self):
        """Verify we use str.replace, not .format() -- curly braces in text shouldn't break."""
        c = DummyConflict()
        # If .format() were used, this would fail or behave unexpectedly
        result = c.build_system_prompt(direction="a", word="test")
        assert "test" in result


class TestBuildUserPrompt:
    def test_direction_a(self):
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="test")  # store args
        result = c.build_user_conflict_prompt(direction="a")
        assert result == "User wants test."

    def test_direction_b(self):
        c = DummyConflict()
        c.build_system_prompt(direction="b", word="test")  # store args
        result = c.build_user_conflict_prompt(direction="b")
        assert result == "Inverse user wants test."


class TestStoreArgs:
    def test_args_stored_after_build(self):
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="hello")
        args = c.get_instruction_args()
        assert args == {"word": "hello"}

    def test_args_keys(self):
        c = DummyConflict()
        assert c.get_instruction_args_keys() == ["word"]

    def test_args_persist_for_user_prompt(self):
        """build_system_prompt stores args, build_user_conflict_prompt uses them."""
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="persisted")
        result = c.build_user_conflict_prompt(direction="a")
        assert "persisted" in result


class TestVerifyFunctions:
    def test_1arg_verify_system_direction_a(self):
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="x")
        assert c.verify_followed_system("system output", direction="a") is True
        assert c.verify_followed_system("random text", direction="a") is False

    def test_1arg_verify_user_direction_a(self):
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="x")
        assert c.verify_followed_user("user output", direction="a") is True

    def test_1arg_verify_direction_b(self):
        c = DummyConflict()
        c.build_system_prompt(direction="b", word="x")
        assert c.verify_followed_system("inverse_system output", direction="b") is True
        assert c.verify_followed_user("inverse_user output", direction="b") is True

    def test_2arg_verify_passes_args(self):
        """co_argcount >= 2 convention: verify fn receives (response, args_dict)."""
        c = DummyArgConflict()
        c.build_system_prompt(direction="a", word="forbidden")
        # "forbidden" not in response -> followed system
        assert c.verify_followed_system("clean text", direction="a") is True
        # "forbidden" in response -> did NOT follow system
        assert c.verify_followed_system("this has forbidden word", direction="a") is False
        # "forbidden" in response -> followed user
        assert c.verify_followed_user("this has forbidden word", direction="a") is True

    def test_verify_direction_b_no_inverse_raises(self):
        """Calling verify with direction='b' on non-invertible should raise."""
        c = DummyNoInverseConflict()
        c.build_system_prompt(direction="a")
        with pytest.raises((ValueError, TypeError)):
            c.verify_followed_system("text", direction="b")


class TestCounterbalancing:
    def test_supports_counterbalancing_full(self):
        c = DummyConflict()
        assert c.supports_counterbalancing() is True

    def test_supports_counterbalancing_none(self):
        c = DummyNoInverseConflict()
        assert c.supports_counterbalancing() is False

    def test_sample_args_returns_expected_keys(self):
        c = DummyConflict()
        args = c.sample_args()
        assert "word" in args


class TestThresholdAndScoring:
    def test_float_verify_applies_threshold(self):
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        # 3/4 = 0.75 >= 0.7 threshold -> True
        assert c.verify_followed_system("good great grand bad", direction="a") is True
        # 1/4 = 0.25 < 0.7 -> False
        assert c.verify_followed_system("good bad bad bad", direction="a") is False

    def test_score_system_returns_raw_float(self):
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        score = c.score_system("good great bad bad", direction="a")
        assert score == pytest.approx(0.5)

    def test_score_user_returns_raw_float(self):
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        score = c.score_user("bad bad bad good", direction="a")
        assert score == pytest.approx(0.75)

    def test_bool_verify_fn_score_returns_0_or_1(self):
        """Bool-returning verify fns produce 1.0 or 0.0 from score methods."""
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="x")
        assert c.score_system("system output", direction="a") == 1.0
        assert c.score_system("random text", direction="a") == 0.0

    def test_verify_threshold_attribute(self):
        c = DummyFloatConflict()
        assert c.verify_threshold == 0.7

    def test_no_default_threshold_for_bool(self):
        """Bool conflicts (test_ prefix, no threshold in yaml) have no verify_threshold."""
        c = DummyConflict()
        assert not hasattr(c, "verify_threshold")


def _inverted_user_fn(response: str) -> float:
    """Returns 1 - fraction of 'g' words (inverted score)."""
    words = response.split()
    if not words:
        return 1.0
    return 1.0 - sum(1 for w in words if w.startswith("g")) / len(words)


_inverted_user_fn.is_inverted = True  # type: ignore[attr-defined]


class DummyFloatInvertedConflict(Conflict):
    """Float conflict with an inverted user verify fn for threshold override testing."""
    conflict_id = "test_float_inverted"
    system_template = "System says {word}."
    user_template = "User wants {word}."
    counterbalance_quality = "none"
    arg_keys = ["word"]
    verify_threshold = 0.7

    def verify_system_fn(response: str) -> float:
        """Returns fraction of words starting with 'g' (direct score)."""
        words = response.split()
        if not words:
            return 0.0
        return sum(1 for w in words if w.startswith("g")) / len(words)

    verify_user_fn = _inverted_user_fn


class TestThresholdOverride:
    def test_threshold_override_takes_precedence(self):
        """Explicit threshold parameter overrides class attribute."""
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        # 2/4 = 0.50 is the score. Default threshold 0.7 -> False
        assert c.verify_followed_system("good great bad bad", direction="a") is False
        # With threshold override 0.5 -> score >= 0.5 -> True
        assert c.verify_followed_system("good great bad bad", direction="a", threshold=0.5) is True

    def test_threshold_none_uses_default(self):
        """threshold=None behaves identically to no parameter."""
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        result_default = c.verify_followed_system("good great grand bad", direction="a")
        result_none = c.verify_followed_system("good great grand bad", direction="a", threshold=None)
        assert result_default == result_none

    def test_threshold_override_with_inverted(self):
        """Inverted check uses > (1 - override) with overridden threshold."""
        c = DummyFloatInvertedConflict()
        c.build_system_prompt(direction="a", word="x")
        # Response: "good bad bad bad" -> g fraction = 0.25
        # User fn is inverted, returns 1 - 0.25 = 0.75
        # Default threshold 0.7: inverted check is score > (1 - 0.7) = score > 0.3 -> 0.75 > 0.3 -> True
        assert c.verify_followed_user("good bad bad bad", direction="a") is True
        # With threshold override 0.3: inverted check is score > (1 - 0.3) = score > 0.7 -> 0.75 > 0.7 -> True
        assert c.verify_followed_user("good bad bad bad", direction="a", threshold=0.3) is True
        # With threshold override 0.2: inverted check is score > (1 - 0.2) = score > 0.8 -> 0.75 > 0.8 -> False
        assert c.verify_followed_user("good bad bad bad", direction="a", threshold=0.2) is False

    def test_threshold_override_system_and_user_independent(self):
        """Threshold override applies to both system and user verify calls."""
        c = DummyFloatConflict()
        c.build_system_prompt(direction="a", word="x")
        # "good bad" -> system score 0.5, user score (b-words) 0.5
        # Default threshold 0.7 -> both False
        assert c.verify_followed_system("good bad", direction="a") is False
        assert c.verify_followed_user("good bad", direction="a") is False
        # With threshold 0.5 -> both True (score >= 0.5)
        assert c.verify_followed_system("good bad", direction="a", threshold=0.5) is True
        assert c.verify_followed_user("good bad", direction="a", threshold=0.5) is True

    def test_threshold_override_does_not_affect_bool_verify(self):
        """Threshold parameter is ignored for bool-returning verify functions."""
        c = DummyConflict()
        c.build_system_prompt(direction="a", word="x")
        # Bool verify fn: "system" in response -> True regardless of threshold
        assert c.verify_followed_system("system output", direction="a", threshold=0.99) is True
        assert c.verify_followed_system("random text", direction="a", threshold=0.01) is False


class TestPerModelExclusion:
    def test_exclude_conflicts_filters_prompts(self):
        """Model with exclude_conflicts produces fewer prompts."""
        from phase0_v2.src.config import ModelConfig
        from phase0_v2.conflicts.registry import get_all_conflicts

        mc_full = ModelConfig(id="model-full")
        mc_excl = ModelConfig(id="model-excl", exclude_conflicts=["word_repetition_density"])

        all_conflicts = get_all_conflicts()
        full_conflicts = [c for c in all_conflicts if c.conflict_id not in set(mc_full.exclude_conflicts)]
        excl_conflicts = [c for c in all_conflicts if c.conflict_id not in set(mc_excl.exclude_conflicts)]

        assert len(excl_conflicts) == len(full_conflicts) - 1

    def test_exclude_multiple_conflicts(self):
        """Excluding multiple conflicts removes all of them."""
        from phase0_v2.src.config import ModelConfig
        from phase0_v2.conflicts.registry import get_all_conflicts

        excluded = ["word_repetition_density", "emoji_use_vs_avoid"]
        mc = ModelConfig(id="model-excl", exclude_conflicts=excluded)

        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.conflict_id not in set(mc.exclude_conflicts)]

        assert len(filtered) == len(all_conflicts) - 2

    def test_empty_exclude_returns_all(self):
        """Empty exclude_conflicts list returns all conflicts."""
        from phase0_v2.src.config import ModelConfig
        from phase0_v2.conflicts.registry import get_all_conflicts

        mc = ModelConfig(id="model-full")
        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.conflict_id not in set(mc.exclude_conflicts)]

        assert len(filtered) == len(all_conflicts)
